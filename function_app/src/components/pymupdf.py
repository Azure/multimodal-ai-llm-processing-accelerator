import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_type

from ..helpers.data_loading import pymupdf_pdf_page_to_img_pil
from ..helpers.image import pil_img_to_base64

with LazyImport("Run 'pip install fitz'") as fitz_import:
    # PyMuPDF is imported as fitz
    from fitz import Document as PyMuPDFDocument

logger = logging.getLogger(__name__)


VALID_PYMUPDF_MIME_TYPES = {"application/pdf"}


@component
class PyMuPDFConverter:
    """
    Converts PDF files to Document (page text) and ByteStream (page image)
    objects.
    """

    def __init__(
        self,
        to_img_dpi: int = 200,
        correct_img_rotation: bool = True,
    ):
        """
        Create an PyMuPDF converter component.

        :param to_img_dpi:
            Sets the DPI of the image output. Larger values result in larger
            output images.
        :param correct_img_rotation:
            If True, rotates the image based on the page rotation of the PDF (if
            encoded within the PDF).
        """
        fitz_import.check()

        self._to_img_dpi = to_img_dpi
        self._correct_img_rotation = correct_img_rotation

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, converter=self.converter.to_dict())

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        converter_class = deserialize_type(data["init_parameters"]["converter"]["type"])
        data["init_parameters"]["converter"] = converter_class.from_dict(
            data["init_parameters"]["converter"]
        )
        return default_from_dict(cls, data)

    def convert(
        self, pymupdf_page: PyMuPDFDocument
    ) -> Tuple[List[Document], List[ByteStream]]:
        """Extract text from the PDF and return Document and ByteStream objects
        containing the content."""
        documents = list()
        images = list()
        for page in pymupdf_page.pages():
            # Text
            documents.append(
                Document(
                    content=page.get_text(),
                    meta={"page_number": page.number},
                )
            )
            # Image
            current_rotation = 0 if self._correct_img_rotation else page.rotation
            rotation_factor = page.rotation if self._correct_img_rotation else 0
            base64_img = pil_img_to_base64(
                pymupdf_pdf_page_to_img_pil(page, self._to_img_dpi, rotation_factor)
            )
            bytestream = ByteStream.from_base64_image(
                base64_img,
                meta={
                    "page_number": page.number,
                    "original_rotation": page.rotation,
                    "current_rotation": current_rotation,
                },
            )
            images.append(bytestream)
        return documents, images

    @component.output_types(documents=List[Document], images=List[ByteStream])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts PDF files to Documents and Images.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []
        images = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue
            try:
                fitz_pdf = fitz.open(stream=io.BytesIO(bytestream.data), filetype="pdf")
                documents_result, images_result = self.convert(fitz_pdf)
            except Exception as e:
                logger.warning(
                    "Could not read source and convert it to Document, skipping. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            for document in documents_result:
                merged_metadata = {**bytestream.meta, **metadata, **document.meta}
                document.meta = merged_metadata
                documents.append(document)

            for image in images_result:
                merged_metadata = {**bytestream.meta, **metadata, **image.meta}
                image.meta = merged_metadata
                images.append(image)

        return {"documents": documents, "images": images}
