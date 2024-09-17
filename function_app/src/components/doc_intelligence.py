import copy
import hashlib
import io
import itertools
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Literal, Optional, Union

import fitz
import networkx as nx
import pandas as pd
import PIL.Image as Image
import requests
from azure.ai.documentintelligence._model_base import rest_field
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    Document,
    DocumentBarcode,
    DocumentFigure,
    DocumentFootnote,
    DocumentFormula,
    DocumentKeyValueElement,
    DocumentLanguage,
    DocumentLine,
    DocumentList,
    DocumentPage,
    DocumentParagraph,
    DocumentSection,
    DocumentSelectionMark,
    DocumentSpan,
    DocumentTable,
    DocumentWord,
    ParagraphRole,
)
from fitz import Page as PyMuPDFPage
from haystack.dataclasses import ByteStream as HaystackByteStream
from haystack.dataclasses import Document as HaystackDocument
from pydantic import BaseModel, Field
from src.components.pymupdf import pymupdf_pdf_page_to_img_pil
from src.helpers.image import pil_img_to_base64

logger = logging.getLogger(__name__)

VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/heif",
}


class EnhancedDocumentContentBase(BaseModel):
    ordered_idx: Optional[str] = Field(
        None,
        description="Index of the content in relation to all other content. This is used to order the content logically.",
    )
    page_number: int = Field(..., description="Page number of the content")

    class Config:
        arbitrary_types_allowed = True


class EnhancedDocumentParagraph(EnhancedDocumentContentBase):
    text: str = Field(description="Text content in the paragraph")
    role: Optional[ParagraphRole] = Field(
        description="Role of the paragraph in the document"
    )
    di_document_paragraph: DocumentParagraph = Field(
        description="Base Paragraph extracted from the document"
    )


class EnhancedDocumentTable(EnhancedDocumentContentBase):
    table_md: str = Field(description="Markdown representation of the table")
    table_df: Optional[pd.DataFrame] = Field(
        None, description="Pandas Dataframe representation of the table"
    )
    di_document_table: DocumentTable = Field(
        description="Base DI table extracted from the document"
    )


# class EnhancedDocumentFigure(EnhancedDocumentContentBase):
#     figure: Image.Image = Field(description="Figure extracted from the document")
#     caption: Optional[str] = Field(
#         None, description="Caption of the figure extracted from the document"
#     )
#     footnotes: Optional[List[DocumentFootnote]] = Field(
#         None, description="Footnote of the figure extracted from the document"
#     )
#     di_document_table: DocumentTable = Field(
#         description="Base DI table extracted from the document"
#     )


def camel_to_snake(name):
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


class EnhancedDocumentMixin:
    def _get_passthrough_fields(self, orig_obj: Any) -> dict[str:Any]:
        return {
            camel_to_snake(k): v
            for k, v in orig_obj.as_dict().items()
            if camel_to_snake(k) in orig_obj.__class__.__dict__["_attr_to_rest_field"]
        }


class EnhancedDocumentFigure(DocumentFigure, EnhancedDocumentMixin):
    """
    An object representing a figure in the document, enhanced with additional
     useful fields.

    :ivar base_figure: The original figure object as returned by the
     Document Intelligence API. Required.
    :vartype image: ~azure.ai.documentintelligence.models.DocumentFigure
    :ivar image: PIL Image. Required.
    :vartype image: Image.Image
    """

    image: Optional[Image.Image] = rest_field(type=Image.Image, name="image")

    def __init__(self, base_figure: DocumentFigure, image: Optional[Image.Image]):
        # Pass through fields from original object
        passthrough_fields = self._get_passthrough_fields(base_figure)
        super().__init__(**passthrough_fields)
        # Add new enhanced fields
        self.image = image

    @property
    def base64_img(self) -> bytes:
        if not self.image:
            raise ValueError("Image not set")
        return self.image.tobytes()


class EnhancedAnalyzeResult(AnalyzeResult, EnhancedDocumentMixin):
    """
    An object representing a figure in the document, enhanced with additional
     useful fields.

    :ivar base_obj: The original figure object as returned by the
     Document Intelligence API. Required.
    :vartype image: ~azure.ai.documentintelligence.models.DocumentFigure
    :ivar image: PIL Image. Required.
    :vartype image: Image.Image
    """

    def __init__(
        self,
        base_obj: AnalyzeResult,
        paragraphs: Optional[List[EnhancedDocumentParagraph]] = None,
        tables: Optional[List[EnhancedDocumentTable]] = None,
        figures: Optional[List[EnhancedDocumentFigure]] = None,
    ):
        # Pass through fields from original object, excluding the new enhanced fields
        passthrough_fields = self._get_passthrough_fields(
            base_obj, exclude_fields=["paragraphs", "tables", "figures"]
        )
        super().__init__(**passthrough_fields)
        # Add new enhanced fields
        self.paragraphs = paragraphs
        self.tables = tables
        self.figures = figures


def is_span_in_span(span: DocumentSpan, parent_span: DocumentSpan) -> bool:
    return span.offset >= parent_span.offset and (span.offset + span.length) <= (
        parent_span.offset + parent_span.length
    )


def get_all_formulas(result: AnalyzeResult) -> List[DocumentFormula]:
    return list(
        itertools.chain.from_iterable(
            page.formulas for page in result.pages if page.formulas
        )
    )


def get_formula(
    all_formulas: List[DocumentFormula], span: dict[str, int]
) -> DocumentFormula:
    matching_formulas = [
        formula for formula in all_formulas if is_span_in_span(formula.span, span)
    ]
    if not matching_formulas:
        raise ValueError(f"No formula found for span {span}")
    if len(matching_formulas) > 1:
        raise NotImplementedError(
            "Support for multiple matched formulas is not yet available."
        )
    return matching_formulas[0]


def crop_img(img: Image.Image, crop_poly: list[float]):
    top_left = (min(crop_poly[::2]), min(crop_poly[1::2]))
    bottom_right = (max(crop_poly[::2]), max(crop_poly[1::2]))
    return img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))


def normalize_xy_coords(
    polygon: list[float],
    existing_scale: tuple[float, float],
    new_scale: tuple[float, float],
):
    """
    Normalizes a polygon from one scale to a new one

    :param polygon: List of coordinates of the polygon (x0, y0, x1, y1, etc.)
    :type polygon: list[float]
    :param existing_dimensions: Dimensions of the existing scale (width, height)
    :type existing_dimensions: tuple[float, float]
    :param new_dimensions: Dimensions of the new scale (width, height)
    :type new_dimensions: tuple[float, float]
    :return: Normalized polygon scaled to the new dimensions.
    :rtype: _type_
    """
    x_coords = polygon[::2]
    x_coords = [x / existing_scale[0] * new_scale[0] for x in x_coords]
    y_coords = polygon[1::2]
    y_coords = [y / existing_scale[1] * new_scale[1] for y in y_coords]
    return list(itertools.chain.from_iterable(zip(x_coords, y_coords)))


def get_section_heirarchy(
    section_direct_children_mapper: dict,
) -> dict[int, tuple[int]]:
    """
    This function takes a mapper of sections to their direct children and
    returns a full heirarchy of all sections.

    :param section_direct_children_mapper: Mapper of section ID to a list of all
        direct children, including the section itself.
    :type section_direct_children_mapper: dict
    :return: _description_
    :rtype: dict[int, tuple[int]]
    """
    all_section_ids = set(section_direct_children_mapper.keys())
    full_heirarchy = dict()
    missing_ids = set(section_direct_children_mapper.keys())
    while missing_ids:
        next_id = min(missing_ids)
        branch_heirarchy = _get_section_heirarchy(
            section_direct_children_mapper, next_id
        )
        full_heirarchy.update(branch_heirarchy)
        missing_ids = all_section_ids - set(full_heirarchy.keys())
    return dict(sorted(full_heirarchy.items()))


def _get_section_heirarchy(
    section_direct_children_mapper: dict, current_id: int
) -> dict:
    """Recursive function to get the heirarchy of a section"""
    output = {current_id: (current_id,)}
    for child_id in section_direct_children_mapper[current_id]:
        if current_id != child_id:
            children_heirarchy = _get_section_heirarchy(
                section_direct_children_mapper, child_id
            )
            for parent, children_tup in children_heirarchy.items():
                output[parent] = (current_id,) + children_tup
    return output


def get_element_heirarchy_mapper(
    result: "AnalyzeResult",
) -> Dict[str, Dict[int, tuple[int]]]:
    """
    Gets a mapping of each element contained to a heirarchy of its parent
    sections. The result will only contain elements contained by a parent section.

    :param result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type result: AnalyzeResult
    :return: A dictionary containing a key for each element type, which each
        contain a mapping of element ID to a tuple of parent section IDs.
    :rtype: dict[str, dict[int, tuple[int]]]
    """
    # Get section mapper, mapping sections to their direct children
    sections = result.sections
    section_direct_children_mapper = {
        section_id: [section_id] for section_id in range(len(sections))
    }
    for section_id, section in enumerate(sections):
        for elem in section.elements:
            if "section" in elem:
                child_id = int(elem.split("/")[-1])
                if child_id != section_id:
                    section_direct_children_mapper[section_id].append(child_id)

    # Now recursively work through the mapping to develop a multi-level heirarchy for every section
    section_heirarchy = get_section_heirarchy(section_direct_children_mapper)

    # Now map each element to the section parents
    elem_to_section_parent_mapper = defaultdict(dict)
    for section_id, section in enumerate(sections):
        for elem in section.elements:
            if "section" not in elem:
                _, elem_type, elem_type_id = elem.split("/")
                elem_to_section_parent_mapper[elem_type][int(elem_type_id)] = (
                    section_heirarchy.get(section_id, ())
                )
    # Return sorted dict
    for key in elem_to_section_parent_mapper:
        elem_to_section_parent_mapper[key] = dict(
            sorted(elem_to_section_parent_mapper[key].items())
        )
    # Now add the section heiarchy
    elem_to_section_parent_mapper["sections"] = section_heirarchy
    return dict(elem_to_section_parent_mapper)


def _convert_section_heirarchy_to_incremental_numbering(
    section_heirarchy: Dict[int, tuple[int]]
) -> Dict[int, tuple[int]]:
    """
    Converts a section heirarchy mapping from a tuple of element IDs to an
    incremental numbering scheme (so that each new section starts from 1).

    :param section_heirarchy: A section heirarchy mapping section ID to a tuple
        of parent section IDs.
    :type section_heirarchy: dict[int, tuple[int]]
    :return: A mapping of section ID to tuple of incremental parent section IDs.
    :rtype: dict[int, tuple[int]]
    """
    # Create a mapping of section level -> tuple of parent section IDs -> dict of section ID: incremental ID
    section_to_section_incremental_id = defaultdict(dict)
    for section_id, parent_section_heirarchy in section_heirarchy.items():
        parent_section_heirarchy = parent_section_heirarchy[:-1]
        parent_level = len(parent_section_heirarchy) - 1
        if parent_level >= 0:
            if (
                parent_section_heirarchy
                not in section_to_section_incremental_id[parent_level]
            ):
                section_to_section_incremental_id[parent_level][
                    parent_section_heirarchy
                ] = {section_id: 1}
            else:
                section_to_section_incremental_id[parent_level][
                    parent_section_heirarchy
                ][section_id] = (
                    len(
                        section_to_section_incremental_id[parent_level][
                            parent_section_heirarchy
                        ]
                    )
                    + 1
                )

    # Now iterate through the mapping to create a final mapping of section ID -> incremental ID
    section_to_incremental_id_mapper = dict()
    for section_id, parent_section_heirarchy in section_heirarchy.items():
        if section_id > 0:
            parent_section_heirarchy = parent_section_heirarchy[:-1]
            incremental_id_tup = tuple()
            # Get iDs for parents
            if len(parent_section_heirarchy) > 1:
                for section_level in range(len(parent_section_heirarchy) - 1):
                    section_incremental_id = section_to_section_incremental_id[
                        section_level
                    ][parent_section_heirarchy[: section_level + 1]][
                        parent_section_heirarchy[section_level + 1]
                    ]
                    incremental_id_tup += (section_incremental_id,)
            # Get current incremental ID
            section_incremental_id = section_to_section_incremental_id[
                len(parent_section_heirarchy) - 1
            ][parent_section_heirarchy][section_id]
            incremental_id_tup += (section_incremental_id,)
            section_to_incremental_id_mapper[section_id] = incremental_id_tup
    return section_to_incremental_id_mapper


def convert_element_heirarchy_to_incremental_numbering(
    elem_heirarchy_mapper: dict[str, Dict[int, tuple[int]]]
) -> dict[str, Dict[int, tuple[int]]]:
    """
    Converts a mapping of elements to their parent sections to an incremental
    numbering scheme.

    :param elem_heirarchy_mapper: A mapper containing keys for each element
        type, which inID to a tuple of parent.
    :type elem_heirarchy_mapper: dict[str, dict[int, tuple[int]]]
    :return: A dictionary containing a key for each element type, which each
        contain a mapping of element ID to a tuple of incremental parent section
        IDs.
    :rtype: dict[int, int]
    """
    if "sections" not in elem_heirarchy_mapper:
        # Sections are not present in the heirarchy, meaning no sections were returned from the API. Return an empty dict
        return dict()
    section_to_heirarchy_mapper = elem_heirarchy_mapper["sections"]
    # print(section_to_heirarchy_mapper)
    section_to_incremental_id_mapper = (
        _convert_section_heirarchy_to_incremental_numbering(section_to_heirarchy_mapper)
    )
    # print(section_to_incremental_id_mapper)

    # Now iterate through all elements and create a mapping of element ID -> incremental heirarchy
    element_to_incremental_id_mapper = defaultdict(dict)
    element_to_incremental_id_mapper["sections"] = section_to_incremental_id_mapper
    for elem_type, elem_heirarchy in elem_heirarchy_mapper.items():
        if elem_type != "sections":
            for elem_id, parent_section_heirarchy in elem_heirarchy.items():
                # print(elem_type, elem_heirarchy)
                # print(elem_id, parent_section_heirarchy)
                parent_section = parent_section_heirarchy[-1]
                element_to_incremental_id_mapper[elem_type][elem_id] = (
                    section_to_incremental_id_mapper.get(parent_section, None)
                )

    return dict(element_to_incremental_id_mapper)


# def get_element_heirarchy_mapping(sections: List[DocumentSection]) -> Dict[str, tuple[int]]:
#     # Get section mapper, mapping sections to their direct children
#     section_direct_children_mapper = {section_id + 1: [section_id + 1] for section_id in range(len(sections) - 1)}
#     for section_id, section in enumerate(sections[1:]):
#         for elem in section.elements:
#             if "section" in elem:
#                 child_id = int(elem.split("/")[-1])
#                 if child_id != section_id:
#                     section_direct_children_mapper[section_id].append(child_id)
#     # print(section_direct_children_mapper)

#     # Now recursively work through the mapping to develop a multi-level heirarchy for every section
#     section_heirarchy = get_section_heirarchy(section_direct_children_mapper)
#     # print(section_heirarchy)

#     # Now map each element to the section parents
#     elem_to_section_parent_mapper = defaultdict(dict)
#     for section_id, section in enumerate(sections):
#         for elem in section.elements:
#             if "section" not in elem:
#                 _, elem_type, elem_type_id = elem.split("/")
#                 elem_to_section_parent_mapper[elem_type][int(elem_type_id)] = section_heirarchy.get(section_id, ())    # Return sorted dict
#     return dict(sorted(elem_to_section_parent_mapper.items()))


def get_span_to_ordered_idx_mapper(
    result: "AnalyzeResult",
) -> Dict[tuple[int, int], int]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
    :returns: A dictionary with the element ID as key and the order as value.
    """
    all_spans = list()
    for attr in [
        "pages",
        "paragraphs",
        "tables",
        "figures",
        "lists",
        "key_value_pairs",
        "documents",
    ]:
        elements = getattr(result, attr) or list()
        all_spans.extend(
            [get_min_and_max_span_offsets(elem.spans) for elem in elements]
        )
    for page in result.pages:
        for attr in ["barcodes", "lines", "words", "selection_marks"]:
            elements = getattr(page, attr) or list()
            for elem in elements:
                if hasattr(elem, "spans"):
                    all_spans.append(get_min_and_max_span_offsets(elem.spans))
                elif hasattr(elem, "span"):
                    all_spans.append(get_min_and_max_span_offsets([elem.span]))

    # Sort by lowest start offset, then largest end offset
    all_spans = sorted(all_spans, key=lambda x: (x[0], 0 - x[1]))

    span_to_ordered_idx_mapper: Dict[str, int] = {
        full_span: idx for idx, full_span in enumerate(all_spans)
    }
    return span_to_ordered_idx_mapper


def get_min_and_max_span_offsets(spans: List[DocumentSpan]) -> tuple[int, int]:
    min_offset = min([span.offset for span in spans])
    max_offset = max([span.offset + span.length for span in spans])
    return (min_offset, max_offset)


class AzureDocumentIntelligenceResultProcessor:
    """
    Convert files to documents using Azure's Document Intelligence service.

    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    In order to be able to use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. Follow the steps described in the [Azure documentation]
    (https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api)
    to set up your resource.

    Usage example:
    ```python
    from haystack.components.converters import AzureOCRDocumentConverter
    from haystack.utils import Secret

    converter = AzureOCRDocumentConverter(endpoint="<url>", api_key=Secret.from_token("<your-api-key>"))
    results = converter.run(sources=["path/to/doc_with_images.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(
        self,
        preceding_context_len: int = 3,
        following_context_len: int = 3,
        merge_multiple_column_headers: bool = True,
        include_imgs: bool = True,
        img_dpi: int = 200,
    ):
        """
        Create an AzureDocumentIntelligenceResultProcessor component.

        :param preceding_context_len: Number of lines before a table to extract as preceding context
            (will be returned as part of metadata).
        :param following_context_len: Number of lines after a table to extract as subsequent context (
            will be returned as part of metadata).
        :param merge_multiple_column_headers: Some tables contain more than one row as a column header
            (i.e., column description).
            This parameter lets you choose, whether to merge multiple column header rows to a single row.
        TODO
        """
        self.preceding_context_len = preceding_context_len
        self.following_context_len = following_context_len
        self.merge_multiple_column_headers = merge_multiple_column_headers
        self.include_imgs = include_imgs
        self.img_dpi = img_dpi

    def _load_fitz_pdf(
        self,
        pdf_path: Optional[Union[str, os.PathLike]] = None,
        pdf_url: Optional[str] = None,
    ) -> "fitz.Document":
        """
        _summary_

        :param pdf_path: Path to local PDF, defaults to None
        :type pdf_path: Optional[Union[str, os.PathLike]], optional
        :param pdf_url: URL path to PDF, defaults to None
        :type pdf_url: Optional[str], optional
        :raises ValueError: Raised when neither `pdf_path` nor `pdf_url` are
         provided
        :return: The loaded fitz/PyMuPDF Document object
        :rtype: fitz.Document
        """
        if pdf_path and pdf_url:
            raise ValueError(
                "Both `pdf_path` and `pdf_url` cannot be provided at the same time."
            )
        if pdf_path is not None:
            return fitz.open(pdf_path)
        elif pdf_url is not None:
            r = requests.get(pdf_url)
            data = r.content
            return fitz.open(stream=data, filetype="pdf")
        else:
            raise ValueError("Either `pdf_path` or `pdf_url` must be provided.")

    def process_analyze_result(
        self,
        result: AnalyzeResult,
        pdf_path: Optional[Union[str, os.PathLike]] = None,
        pdf_url: Optional[str] = None,
    ):
        """
        Processes the result of an analyze operation and returns the extracted text and tables.

        :param result: The result of an analyze operation.
        :returns: A dictionary containing the extracted text and tables.
        """
        # Get images of every page
        pdf_page_imgs: Dict[int, Image.Image] = dict()
        if self.include_imgs:
            pdf = self._load_fitz_pdf(pdf_path=pdf_path, pdf_url=pdf_url)
            for page in pdf.pages():
                pdf_page_imgs[page.number + 1] = pymupdf_pdf_page_to_img_pil(
                    next(iter(pdf.pages())), img_dpi=self.img_dpi, rotation=False
                )

        # Get mapper of element to heirarchy
        elem_heirarchy_mapper = get_element_heirarchy_mapper(result.sections)
        span_to_ordered_idx_mapper = get_span_to_ordered_idx_mapper(result)

        # Extract the text and tables
        paragraphs = self._convert_paragraphs(
            result, elem_heirarchy_mapper, span_to_ordered_idx_mapper
        )
        figures = self._convert_figures(
            result, pdf_page_imgs, elem_heirarchy_mapper, span_to_ordered_idx_mapper
        )
        tables = self._convert_tables(
            result, elem_heirarchy_mapper, span_to_ordered_idx_mapper
        )

        return EnhancedAnalyzeResult(
            base_obj=result,
            paragraphs=paragraphs,
            tables=tables,
            figures=figures,
        )

    def _convert_figures(
        self,
        result: "AnalyzeResult",
        page_imgs: Dict[int, Image.Image],
        elem_heirarchy_mapper: Dict[str, tuple[int]],
        span_to_ordered_idx_mapper: Dict[tuple[int], int],
    ) -> List[EnhancedDocumentFigure]:
        """TODO"""
        figures = []
        for fig_id, figure in enumerate(result.figures):
            enhanced_figure = self._convert_figure(
                result=result,
                figure=figure,
                page_imgs=page_imgs,
                ordered_idx=span_to_ordered_idx_mapper[
                    get_min_and_max_span_offsets(figure.spans)
                ],
                section_heirarchy=elem_heirarchy_mapper["figures"][fig_id],
            )
            figures.append(enhanced_figure)
        return figures

    def _convert_figure(
        self,
        result: AnalyzeResult,
        figure: DocumentFigure,
        page_imgs: Optional[Dict[int, Image.Image]],
        ordered_idx: int,
        section_heirarchy: tuple[int],
    ) -> List[EnhancedDocumentFigure]:
        """TODO"""
        if page_imgs:
            page_numbers = [region.page_number for region in figure.bounding_regions]
            if len(page_numbers) > 1:
                logger.warning(
                    f"Figure spans multiple pages. Only the first page will be used. Page numbers: {page_numbers}"
                )
            page_number = page_numbers[0]
            page_img = page_imgs[page_number]
            di_page = next(
                iter([page for page in result.pages if page.page_number == page_number])
            )
            di_page_dimensions = (di_page.width, di_page.height)
            figure_img = crop_img(
                page_img,
                normalize_xy_coords(
                    figure.bounding_regions[0].polygon,
                    di_page_dimensions,
                    page_img.size,
                ),
            )
        else:
            figure_img = None

        enhanced_figure = EnhancedDocumentFigure(
            base_obj=figure,
            image=figure_img,
            ordered_idx=ordered_idx,
            section_heirarchy=section_heirarchy,
        )
        return enhanced_figure

    def _convert_tables(
        self,
        result: "AnalyzeResult",
        elem_heirarchy_mapper: Dict[str, tuple[int]],
        span_to_ordered_idx_mapper: Dict[tuple[int], int],
    ) -> List[EnhancedDocumentTable]:
        """
        Converts the tables extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :returns: List of Documents containing the tables extracted from the AnalyzeResult object.
        """
        all_formulas = get_all_formulas(result)
        tables = []
        for table_id, table in enumerate(result.tables):
            enhanced_table = self._convert_table(
                table=table,
                all_formulas=all_formulas,
                ordered_idx=span_to_ordered_idx_mapper[
                    get_min_and_max_span_offsets(table.spans)
                ],
                section_heirarchy=elem_heirarchy_mapper["figures"][table_id],
            )
            tables.append(enhanced_table)
        return tables

    def _convert_table(
        self,
        table: DocumentTable,
        all_formulas: List[DocumentFormula],
        ordered_idx: int,
        section_heirarchy: tuple[int],
    ) -> EnhancedDocumentTable:
        sorted_cells = sorted(
            table.cells, key=lambda cell: (cell.row_index, cell.column_index)
        )
        cell_dict = defaultdict(list)

        num_header_rows = 0
        num_index_cols = 0
        for cell in sorted_cells:
            if cell.kind == "columnHeader":
                num_header_rows = max(num_header_rows, cell.row_index + 1)
            if cell.kind == "rowHeader":
                num_index_cols = max(num_index_cols, cell.column_index + 1)
            # Get text content
            cell_content = cell.content
            if ":formula:" in cell_content:
                matched_formula = get_formula(all_formulas, cell.spans[0])
                cell_content = cell_content.replace(":formula:", matched_formula.value)
            cell_content = cell_content.replace(":selected:", "").replace(
                ":unselected:", ""
            )
            cell_dict[cell.row_index].append(cell_content)
        table_df = pd.DataFrame.from_dict(cell_dict, orient="index")
        # Set index and columns
        if num_index_cols > 0:
            table_df.set_index(list(range(0, num_index_cols)), inplace=True)
        if num_header_rows > 0:
            table_df = table_df.T
            table_df.set_index(list(range(0, num_header_rows)), inplace=True)
            table_df = table_df.T
        index = num_index_cols > 0
        table_fmt = "github" if num_header_rows > 0 else "rounded_grid"
        table_md = table_df.to_markdown(index=index, tablefmt=table_fmt)
        # TODO: Set headers and columns by checking # of rows and columns of kind columnHeader / rowHeader, then set index and columns
        page_number = min(
            region.page_number
            for cell in sorted_cells
            for region in cell.bounding_regions
        )
        # Construct model without validating the ID., which will be added later
        return EnhancedDocumentTable.model_construct(
            page_number=page_number,
            span=table.spans[0],
            table_md=table_md,
            table_df=table_df,
            ordered_idx=ordered_idx,
            section_heirarchy=section_heirarchy,
            di_document_table=table,
        )

    def _convert_paragraphs(
        self,
        result: "AnalyzeResult",
        elem_heirarchy_mapper: Dict[str, tuple[int]],
        span_to_ordered_idx_mapper: Dict[tuple[int], int],
    ) -> List[EnhancedDocumentParagraph]:
        """
        This converts the `AnalyzeResult` object ... TODO.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :returns: TODO
        """
        table_spans = self._collect_table_spans(result=result)
        figure_spans = self._collect_figure_spans(result=result)

        paragraphs_to_pages: Dict[int, list] = defaultdict(list)
        if not result.paragraphs:
            logger.warning("No text paragraphs were detected by the OCR conversion.")
            return []
        for paragraph_idx, paragraph in enumerate(result.paragraphs):
            print(paragraph_idx, paragraph.content)
            # Check if paragraph is part of a table or figure and if so skip
            if any(
                (
                    self._check_if_in_table(table_spans, line_or_paragraph=paragraph),
                    self._check_if_in_figure(figure_spans, line_or_paragraph=paragraph),
                )
            ):
                continue
            if paragraph.bounding_regions:
                # If paragraph spans multiple pages we group it with the first page number
                page_numbers = [b.page_number for b in paragraph.bounding_regions]
            else:
                # If page_number is not available we put the paragraph onto an existing page
                current_last_page_number = (
                    sorted(paragraphs_to_pages.keys())[-1] if paragraphs_to_pages else 1
                )
                page_numbers = [current_last_page_number]
            # Get section heirarchy. For non-normal content like headers and footers, this will be None
            section_heirarchy = elem_heirarchy_mapper["paragraphs"].get(
                paragraph_idx, tuple()
            )
            # Contruct the EnhancedDocumentParagraph model
            enhanced_paragraph = EnhancedDocumentParagraph(
                text=paragraph.content,
                role=paragraph.role,
                page_number=page_numbers[0],
                span=paragraph.spans[0],
                ordered_idx=span_to_ordered_idx_mapper[
                    get_min_and_max_span_offsets(paragraph.spans)
                ],
                section_heirarchy=section_heirarchy,
                di_document_paragraph=paragraph,
            )
            paragraphs_to_pages[page_numbers[0]].append(enhanced_paragraph)

            # max_page_number: int = max(paragraphs_to_pages)
            # for page_idx in range(1, max_page_number + 1):
            #     # We add empty strings for missing pages so the preprocessor can still extract the correct page number
            #     # from the original PDF.
            #     page_text = paragraphs_to_pages.get(page_idx, "")
            #     paragraphs.append(page_text)
        return list(itertools.chain.from_iterable(paragraphs_to_pages.values()))

    def _collect_table_spans(self, result: "AnalyzeResult") -> List["DocumentSpan"]:
        """
        Collect the spans of all tables by page number.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
        :returns: A dictionary with the page number as key and a list of table spans as value.
        """
        tables: List[DocumentTable] = result.tables
        return list(itertools.chain.from_iterable([table.spans for table in tables]))

    def _check_if_in_table(
        self,
        table_spans: List["DocumentSpan"],
        line_or_paragraph: Union["DocumentLine", "DocumentParagraph"],
    ) -> bool:
        """
        Check if a line or paragraph is part of a table.

        :param table_spans_on_page: A list of table spans on the current page.
        :param line_or_paragraph: The line or paragraph to check.
        :returns: True if the line or paragraph is part of a table, False otherwise.
        """
        for table_span in table_spans:
            if (
                table_span.offset
                <= line_or_paragraph.spans[0].offset
                <= table_span.offset + table_span.length
            ):
                return True
        return False

    def _collect_figure_spans(self, result: "AnalyzeResult") -> List["DocumentSpan"]:
        """
        Collect the spans of all figures by page number.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
        :returns: A dictionary with the page number as key and a list of figure spans as value.
        """
        figures: List[DocumentFigure] = result.figures
        return list(itertools.chain.from_iterable([figure.spans for figure in figures]))

    def _check_if_in_figure(
        self,
        figure_spans: List["DocumentSpan"],
        line_or_paragraph: Union["DocumentLine", "DocumentParagraph"],
    ) -> bool:
        """
        Check if a line or paragraph is part of a figure.

        :param figure_spans_on_page: A list of figure spans on the current page.
        :param line_or_paragraph: The line or paragraph to check.
        :returns: True if the line or paragraph is part of a figure, False otherwise.
        """
        for figure_span in figure_spans:
            if (
                figure_span.offset
                <= line_or_paragraph.spans[0].offset
                <= figure_span.offset + figure_span.length
            ):
                return True
        return False


@dataclass
class BaseFormatConfig:
    include: bool

    def format_page_element(
        self, page_element: DocumentPage, element_id: str, **kwargs
    ) -> List[HaystackDocument]:
        raise NotImplementedError("Subclasses must implement this method.")


class DocumentPageImgExporterBase(ABC):

    @abstractmethod
    def export_page_img(self, page: PyMuPDFPage) -> Image.Image:
        """
        Method for exporting a Page as an image.
        """
        pass


class DefaultDocumentPageImgExporter(DocumentPageImgExporterBase):
    def __init__(self, img_dpi: int = 200):
        self.img_dpi = img_dpi

    def export_page_img(self, page: PyMuPDFPage) -> Image.Image:
        return pymupdf_pdf_page_to_img_pil(page, img_dpi=self.img_dpi, rotation=0)


@dataclass
class PageFormatConfig(BaseFormatConfig):
    text_format: str = "Page: {page_number}"
    page_text_order: Literal["start", "end", "none"] = "start"
    page_img_order: Literal["start", "end", "none"] = "end"

    def format_page_element(
        self,
        page_element: DocumentPage,
        element_id: str,
        current_location: Literal["start", "end"],
        page_img: Optional[Image.Image] = None,
    ) -> List[HaystackDocument]:
        documents: List[HaystackDocument] = list()
        if self.include:
            # Text content
            if self.page_text_order == current_location:
                documents.append(
                    HaystackDocument(
                        id=f"{element_id}_text",
                        content=self.text_format.format(
                            page_number=page_element.page_number
                        ),
                        meta={
                            "element_id": element_id,
                            "page_text_order": self.page_text_order,
                        },
                    )
                )
            if self.page_img_order == current_location:
                if page_img is None:
                    raise ValueError(
                        "page_img must be provided if page_img_order is not 'none'"
                    )
                img_bytestream = HaystackByteStream(
                    data=pil_img_to_base64(page_img),
                    mime_type="image/jpeg",
                )
                documents.append(
                    HaystackDocument(
                        id=f"{element_id}_img",
                        blob=img_bytestream,
                        meta={
                            "element_id": element_id,
                            "page_img_order": self.page_img_order,
                        },
                    )
                )
        return documents


class DocumentTableLoaderBase(ABC):

    @abstractmethod
    def load_di_document_table(table: DocumentTable) -> tuple[str, pd.DataFrame]:
        """
        Method for loading a DocumentTable object into a tuple of markdown
        and DataFrame
        """
        pass


class DefaultDocumentTableLoader(DocumentTableLoaderBase):
    def __init__(self, all_formulas: List[DocumentFormula]):
        self.all_formulas = all_formulas

    def load_di_document_table(self, table: DocumentTable) -> tuple[str, pd.DataFrame]:
        sorted_cells = sorted(
            table.cells, key=lambda cell: (cell.row_index, cell.column_index)
        )
        cell_dict = defaultdict(list)

        num_header_rows = 0
        num_index_cols = 0
        for cell in sorted_cells:
            if cell.kind == "columnHeader":
                num_header_rows = max(num_header_rows, cell.row_index + 1)
            if cell.kind == "rowHeader":
                num_index_cols = max(num_index_cols, cell.column_index + 1)
            # Get text content
            cell_content = cell.content
            if ":formula:" in cell_content:
                matched_formula = get_formula(self.all_formulas, cell.spans[0])
                cell_content = cell_content.replace(":formula:", matched_formula.value)
            cell_content = cell_content.replace(":selected:", "").replace(
                ":unselected:", ""
            )
            cell_dict[cell.row_index].append(cell_content)
        table_df = pd.DataFrame.from_dict(cell_dict, orient="index")
        # Set index and columns
        if num_index_cols > 0:
            table_df.set_index(list(range(0, num_index_cols)), inplace=True)
        if num_header_rows > 0:
            table_df = table_df.T
            table_df.set_index(list(range(0, num_header_rows)), inplace=True)
            table_df = table_df.T
        index = num_index_cols > 0
        table_fmt = "github" if num_header_rows > 0 else "rounded_grid"
        table_md = table_df.to_markdown(index=index, tablefmt=table_fmt)
        return table_md, table_df


@dataclass
class TableFormatConfig(BaseFormatConfig):
    """
    Sets the config for how tables are formatted.

    Args:
    TODO

    The following fields can be used in `text_format`:
    - {table}: The table in markdown format.
    - {caption}: The caption of the table, if any. Typically a title or
        description.
    - {footnotes}: The footnotes of the table, if any.
    """

    table_loader: DocumentTableLoaderBase
    text_format: str = "{caption}\n{table}\n{footnotes}"
    content_outputs: tuple[Literal["text", "dataframe"]] = "text"

    def format_page_element(
        self, page_element: DocumentTable, element_id: str
    ) -> List[HaystackDocument]:
        documents = list()
        if self.include:
            table_md, table_df = self.table_loader.load_di_document_table(page_element)
            footnotes = "\n".join(
                [footnote.content for footnote in page_element.footnotes]
            )
            for output_type in self.content_outputs:
                if output_type == "text":
                    documents.append(
                        HaystackDocument(
                            id=f"{element_id}_text",
                            content=self.text_format.format(
                                table=table_md,
                                caption=page_element.caption,
                                footnotes=footnotes,
                            ),
                            meta={"element_id": element_id},
                        )
                    )
                elif output_type == "dataframe":
                    documents.append(
                        HaystackDocument(
                            id=f"{element_id}_dataframe",
                            content=table_df,
                            meta={"element_id": element_id},
                        )
                    )
        return documents


class DocumentFigureExporterBase(ABC):

    @abstractmethod
    def export_figure_img(
        figure: DocumentFigure,
    ) -> Image.Image:
        """
        Method for loading a DocumentFigure object into a tuple of image objects
        (text_content, PIL.Image.Image and bytes64)
        """
        pass


class DefaultDocumentFigureExporter(DocumentFigureExporterBase):
    def __init__(self):
        pass

    def export_figure_img(
        self,
        figure: DocumentFigure,
        result: AnalyzeResult,
        page_imgs: Optional[Dict[int, Image.Image]],
    ) -> Image.Image:
        page_numbers = [region.page_number for region in figure.bounding_regions]
        if len(page_numbers) > 1:
            logger.warning(
                f"Figure spans multiple pages. Only the first page will be used. Page numbers: {page_numbers}"
            )
        page_number = page_numbers[0]
        page_img = page_imgs[page_number]
        di_page = next(
            iter([page for page in result.pages if page.page_number == page_number])
        )
        di_page_dimensions = (di_page.width, di_page.height)
        figure_img = crop_img(
            page_img,
            normalize_xy_coords(
                figure.bounding_regions[0].polygon,
                di_page_dimensions,
                page_img.size,
            ),
        )
        return figure_img


@dataclass
class FigureFormatConfig(BaseFormatConfig):
    """
    Sets the config for how figures are formatted.

    Args:
    TODO

    The following fields can be used in `text_format`:
    - {table}: The table in markdown format.
    - {caption}: The caption of the table, if any. Typically a title or
        description.
    - {footnotes}: The footnotes of the table, if any.
    """

    figure_exporter: DocumentFigureExporterBase
    before_text_format: str = "{caption}\n{table}\n{footnotes}"
    image_order: Literal["start", "middle", "end", "none"] = "end"
    after_text_format: str = "{caption}\n{table}\n{footnotes}"
    content_outputs: tuple[Literal["before_text", "image", "after_text"]] = (
        "before_text",
        "image",
        "after_text",
    )

    def format_page_element(
        self, page_element: DocumentTable, element_id: str
    ) -> List[HaystackDocument]:
        documents = list()
        if self.include:
            table_md, table_df = self.table_loader.load_di_document_table(page_element)
            footnotes = "\n".join(
                [footnote.content for footnote in page_element.footnotes]
            )
            for output_type in self.content_outputs:
                if output_type == "text":
                    documents.append(
                        HaystackDocument(
                            id=f"{element_id}_text",
                            content=self.text_format.format(
                                table=table_md,
                                caption=page_element.caption,
                                footnotes=footnotes,
                            ),
                            meta={"element_id": element_id},
                        )
                    )
                elif output_type == "dataframe":
                    documents.append(
                        HaystackDocument(
                            id=f"{element_id}_dataframe",
                            content=table_df,
                            meta={"element_id": element_id},
                        )
                    )
        return documents
