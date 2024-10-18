import io
import os
from typing import Dict, Optional, Union

import fitz
import requests
from fitz import Document as PyMuPDFDocument
from PIL import Image
from PIL.Image import Image as PILImage

from ..components.utils import base64_bytes_to_buffer


def load_pymupdf_pdf(
    pdf_bytes: Optional[bytes] = None,
    pdf_path: Optional[Union[str, os.PathLike]] = None,
    pdf_url: Optional[str] = None,
) -> "fitz.Document":
    """
    Loads a PDF file using PyMuPDF (fitz).

    :param pdf_bytes: Bytes object representing the PDF, defaults to None
    :type pdf_bytes: Optional[bytes], optional
    :param pdf_path: Path to local PDF, defaults to None
    :type pdf_path: Optional[Union[str, os.PathLike]], optional
    :param pdf_url: URL path to PDF, defaults to None
    :type pdf_url: Optional[str], optional
    :raises ValueError: Raised when neither `pdf_path` nor `pdf_url` are
        provided
    :return: The loaded fitz/PyMuPDF Document object
    :rtype: fitz.Document
    """
    num_sources_provided = sum(
        [1 for source in [pdf_bytes, pdf_path, pdf_url] if source is not None]
    )
    if num_sources_provided > 1:
        raise ValueError(
            "Only one source for the PDF can be given. Please provide only one of `pdf_bytes`, `pdf_path`, or `pdf_url`."
        )
    if pdf_bytes is not None:
        return fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    elif pdf_path is not None:
        return fitz.open(pdf_path)
    elif pdf_url is not None:
        r = requests.get(pdf_url)
        data = r.content
        return fitz.open(stream=data, filetype="pdf")
    else:
        raise ValueError(
            "Either `pdf_bytes`, `pdf_path` or `pdf_url` must be provided."
        )


def load_visual_obj_bytes_to_pil_imgs_dict(
    media_bytes: bytes,
    mime_type: str,
    starting_idx: int = 1,
    pdf_img_dpi: int = 100,
) -> Dict[int, PILImage]:
    """
    Loads a byte string representing a media object and convert it to a
    dictionary of PIL images. This dictionary will map page indices to the
    corresponding PIL images.

    :param media_bytes: Bytes object representing the media object.
    :type media_bytes: bytes
    :param mime_type: MIME type of the media object.
    :type mime_type: str
    :param starting_idx: Starting index of the output dictionary, defaults to 1
    :type starting_idx: int, optional
    :param pdf_img_dpi: DPI to use when converting PDF files to images,
        defaults to 100
    :type pdf_img_dpi: int, optional
    :raises ValueError: In cases where the media MIME type is not supported.
    :return: Dictionary of page index to PIL Image.
    :rtype: Dict[int, PILImage]
    """
    if mime_type.startswith("image"):
        name = mime_type.replace("/", ".")
        req_body_buffer = base64_bytes_to_buffer(media_bytes, name=name)
        return {1: Image.open(req_body_buffer)}
    elif mime_type == "application/pdf":
        pdf = load_pymupdf_pdf(media_bytes)
        return extract_pdf_page_images(
            pdf, img_dpi=pdf_img_dpi, starting_idx=starting_idx
        )
    else:
        raise ValueError(f"Unsupported media type: {mime_type}")


def pymupdf_pdf_page_to_img_pil(
    pymupdf_page: fitz.Page, img_dpi: int, rotation: int
) -> Image.Image:
    """
    Converts a PyMuPDF page to a PIL Image.

    :param pymupdf_page: PyMuPDF page object to convert.
    :type pymupdf_page: fitz.Page
    :param img_dpi: DPI to use when converting the image.
    :type img_dpi: int
    :param rotation: Rotation to apply to the image (counter-clockwise). If
        rotation is applied, the image will be expanded to fit the new size.
    :type rotation: int
    :return: PIL Image object
    :rtype: Image.Image
    """
    pix = pymupdf_page.get_pixmap(dpi=img_dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img.rotate(rotation, expand=True)


def extract_pdf_page_images(
    pdf: PyMuPDFDocument, img_dpi: int = 100, starting_idx: int = 1
) -> Dict[int, PILImage]:
    """
    Extracts all images from a PDF document and returns them as a dictionary.

    :param pdf: PDF document to extract images from.
    :type pdf: fitz.Document
    :param img_dpi: DPI to use when extracting images, defaults to 100
    :type img_dpi: int, optional
    :param starting_idx: Index to start numbering the pages from, defaults to 1
    :type starting_idx: int, optional
    :return: Dictionary of page index to PIL Image
    :rtype: Dict[int, PIL.Image.Image]
    """
    page_imgs: Dict[int, PILImage] = dict()
    for page_idx, page in enumerate(pdf.pages()):
        page_imgs[page_idx + starting_idx] = pymupdf_pdf_page_to_img_pil(
            page, img_dpi=img_dpi, rotation=False
        )
    return page_imgs
