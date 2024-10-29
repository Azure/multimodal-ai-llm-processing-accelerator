import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from azure.ai.documentintelligence._model_base import Model as DocumentModelBase
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    Document,
    DocumentBarcode,
    DocumentBarcodeKind,
    DocumentFigure,
    DocumentFormula,
    DocumentKeyValuePair,
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
from azure.ai.formrecognizer import AnalyzeResult as FRAnalyzeResult
from azure.ai.formrecognizer import DocumentBarcode as FRDocumentBarcode
from azure.ai.formrecognizer import DocumentFormula as FRDocumentFormula
from azure.ai.formrecognizer import DocumentKeyValuePair as FRDocumentKeyValuePair
from azure.ai.formrecognizer import DocumentLine as FRDocumentLine
from azure.ai.formrecognizer import DocumentPage as FRDocumentPage
from azure.ai.formrecognizer import DocumentParagraph as FRDocumentParagraph
from azure.ai.formrecognizer import DocumentSelectionMark as FRDocumentSelectionMark
from azure.ai.formrecognizer import DocumentSpan as FRDocumentSpan
from azure.ai.formrecognizer import DocumentTable as FRDocumentTable
from azure.ai.formrecognizer import DocumentWord as FRDocumentWord
from fitz import Document as PyMuPDFDocument
from haystack.dataclasses import ByteStream as HaystackByteStream
from haystack.dataclasses import Document as HaystackDocument
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from PIL.Image import Image as PILImage
from pylatexenc.latex2text import LatexNodes2Text

from ..components.pymupdf import pymupdf_pdf_page_to_img_pil
from ..helpers.image import (
    TransformedImage,
    crop_img,
    get_flat_poly_lists_convex_hull,
    pil_img_to_base64,
    rotate_img_pil,
    rotate_polygon,
    scale_flat_poly_list,
)

logger = logging.getLogger(__name__)

VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/heif",
}

LATEX_NODES_TO_TEXT = LatexNodes2Text()


def is_span_in_span(
    span: Union[DocumentSpan, FRDocumentSpan],
    parent_span: Union[DocumentSpan, FRDocumentSpan],
) -> bool:
    """
    Checks if a given span is contained by a parent span.

    :param span: Span to check.
    :type span: DocumentSpan
    :param parent_span: Parent span to check against.
    :type parent_span: DocumentSpan
    :return: Whether the span is contained by the parent span.
    :rtype: bool
    """
    return span.offset >= parent_span.offset and (span.offset + span.length) <= (
        parent_span.offset + parent_span.length
    )


def get_all_formulas(analyze_result: AnalyzeResult) -> List[DocumentFormula]:
    """
    Returns all formulas from the Document Intelligence result.

    :param analyze_result: AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A list of all formulas in the result.
    :rtype: List[DocumentFormula]
    """
    return list(
        itertools.chain.from_iterable(
            page.formulas for page in analyze_result.pages if page.formulas
        )
    )


def get_all_barcodes(analyze_result: AnalyzeResult) -> List[DocumentBarcode]:
    """
    Returns all barcodes from the Document Intelligence result.

    :param analyze_result: AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A list of all barcodes in the result.
    :rtype: List[DocumentBarcode]
    """
    return list(
        itertools.chain.from_iterable(
            page.barcodes for page in analyze_result.pages if page.barcodes
        )
    )


def get_formulas_in_spans(
    all_formulas: List[DocumentFormula],
    spans: List[Union[DocumentSpan, FRDocumentSpan]],
) -> List[DocumentFormula]:
    """
    Get all formulas contained within a list of given spans.

    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param span: The span to match.
    :type span: DocumentSpan
    :return: The formulas contained within the given span.
    :rtype: List[DocumentFormula]
    """
    matching_formulas = list()
    for span in spans:
        matching_formulas.extend(
            [formula for formula in all_formulas if is_span_in_span(formula.span, span)]
        )
    return matching_formulas


def get_barcodes_in_spans(
    all_barcodes: List[DocumentBarcode],
    spans: List[Union[DocumentSpan, FRDocumentSpan]],
) -> List[DocumentBarcode]:
    """
    Get all barcodes contained within a list of given spans.

    :param all_barcodes: A list of all barcodes in the document.
    :type all_barcodes: List[DocumentBarcode]
    :param span: The span to match.
    :type span: DocumentSpan
    :return: The barcodes contained within the given span.
    :rtype: List[DocumentBarcode]
    """
    matching_barcodes = list()
    for span in spans:
        matching_barcodes.extend(
            [barcode for barcode in all_barcodes if is_span_in_span(barcode.span, span)]
        )
    return matching_barcodes


def latex_to_text(latex_str: str) -> str:
    """
    Converts a string containing LaTeX to plain text.

    :param latex_str: The LaTeX string to convert.
    :type latex_str: str
    :return: The plain text representation of the LaTeX string.
    :rtype: str
    """
    return LATEX_NODES_TO_TEXT.latex_to_text(latex_str).strip()


def substitute_content_formulas(
    content: str, matching_formulas: list[DocumentFormula]
) -> str:
    """
    Substitute formulas in the content with their actual values.

    :param content: The content to substitute formulas in.
    :type content: str
    :param matching_formulas: The formulas to substitute.
    :type matching_formulas: list[DocumentFormula]
    :returns: The content with the formulas substituted.
    :rtype: str
    """
    # Get all string indices of :formula: in the content
    match_bounds = [
        (match.start(), match.end()) for match in re.finditer(r":formula:", content)
    ]
    if not len(match_bounds) == len(matching_formulas):
        raise ValueError(
            "The number of formulas to substitute does not match the number of :formula: placeholders in the content."
        )
    # Regex may throw issues with certain characters (e.g. double backslashes), so do the replacement manually
    new_content = ""
    last_idx = 0
    for match_bounds, matching_formula in zip(match_bounds, matching_formulas):
        if match_bounds[0] == last_idx:
            new_content += latex_to_text(matching_formula.value)
            last_idx = match_bounds[1]
        else:
            new_content += content[last_idx : match_bounds[0]]
            new_content += latex_to_text(matching_formula.value)
            last_idx = match_bounds[1]
    new_content += content[last_idx:]
    return new_content


def substitute_content_barcodes(
    content: str, matching_barcodes: list[DocumentBarcode]
) -> str:
    """
    Substitute barcodes in the content with their actual values.

    :param content: The content to substitute barcodes in.
    :type content: str
    :param matching_barcodes: The barcodes to substitute.
    :type matching_barcodes: list[DocumentBarcode]
    :returns: The content with the barcodes substituted.
    :rtype: str
    """
    # Get all string indices of :barcode: in the content
    match_bounds = [
        (match.start(), match.end()) for match in re.finditer(r":barcode:", content)
    ]
    if not len(match_bounds) == len(matching_barcodes):
        raise ValueError(
            "The number of barcodes to substitute does not match the number of :barcode: placeholders in the content."
        )
    # Regex may throw issues with certain characters (e.g. double backslashes), so do the replacement manually
    new_content = ""
    last_idx = 0
    for match_bounds, matching_barcode in zip(match_bounds, matching_barcodes):
        kind_str = (
            matching_barcode.kind.value
            if isinstance(matching_barcode.kind, DocumentBarcodeKind)
            else matching_barcode.kind
        )
        if match_bounds[0] == last_idx:
            new_content += f"*Barcode value:* {matching_barcode.value} (*Barcode kind:* {kind_str})"
            last_idx = match_bounds[1]
        else:
            new_content += content[last_idx : match_bounds[0]]
            new_content += f"*Barcode value:* {matching_barcode.value} (*Barcode kind:* {kind_str})"
            last_idx = match_bounds[1]
    new_content += content[last_idx:]
    return new_content


def replace_content_formulas_and_barcodes(
    content: str,
    content_spans: List[Union[DocumentSpan, FRDocumentSpan]],
    all_formulas: List[DocumentFormula],
    all_barcodes: List[DocumentBarcode],
) -> str:
    """
    Replace formulas in the content with their actual values.

    :param content: The content to replace formulas in.
    :type content: str
    :param content_spans: The spans of the content.
    :type content_spans: List[DocumentSpan]
    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param all_barcodes: A list of all barcodes in the document.
    :type all_barcodes: List[DocumentBarcode]
    :returns: The content with the formulas and barcdoes replaced.
    :rtype: str
    """
    if ":formula:" in content:
        matching_formulas = get_formulas_in_spans(all_formulas, content_spans)
        content = substitute_content_formulas(content, matching_formulas)
    if ":barcode:" in content:
        matching_barcodes = get_barcodes_in_spans(all_barcodes, content_spans)
        content = substitute_content_barcodes(content, matching_barcodes)
    return content


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
    """
    Recursive function to get the heirarchy of a section.

    :param section_direct_children_mapper: Mapper of section ID to a list of all
        direct children, including the section itself.
    :type section_direct_children_mapper: dict
    :param current_id: The ID of the current section to get the heirarchy for.
    :type current_id: int
    :return: A dictionary containing the heirarchy of the current section.
    :rtype: dict
    """
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
    analyze_result: "AnalyzeResult",
) -> Dict[str, Dict[int, tuple[int]]]:
    """
    Gets a mapping of each element contained to a heirarchy of its parent
    sections. The result will only contain elements contained by a parent section.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A dictionary containing a key for each element type, which each
        contain a mapping of element ID to a tuple of parent section IDs.
    :rtype: dict[str, dict[int, tuple[int]]]
    """
    if not getattr(analyze_result, "sections", None):
        # Sections do not appear in result (it might be the read model result).
        # Return an empty dict
        return dict()
    # Get section mapper, mapping sections to their direct children
    sections = analyze_result.sections
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
) -> Dict[str, Dict[int, tuple[int]]]:
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
    section_to_incremental_id_mapper = (
        _convert_section_heirarchy_to_incremental_numbering(section_to_heirarchy_mapper)
    )

    # Now iterate through all elements and create a mapping of element ID -> incremental heirarchy
    element_to_incremental_id_mapper = defaultdict(dict)
    element_to_incremental_id_mapper["sections"] = section_to_incremental_id_mapper
    for elem_type, elem_heirarchy in elem_heirarchy_mapper.items():
        if elem_type != "sections":
            for elem_id, parent_section_heirarchy in elem_heirarchy.items():
                parent_section = parent_section_heirarchy[-1]
                element_to_incremental_id_mapper[elem_type][elem_id] = (
                    section_to_incremental_id_mapper.get(parent_section, None)
                )

    return dict(element_to_incremental_id_mapper)


def get_document_element_spans(
    document_element: DocumentModelBase,
) -> list[Union[DocumentSpan, FRDocumentSpan]]:
    """
    Get the spans of a document element.

    :param document_element: The document element to get the spans of.
    :type document_element: Model
    :raises NotImplementedError: Raised when the document element does not
        contain a span field.
    :return: The spans of the document element.
    :rtype: list[DocumentSpan]
    """
    if isinstance(document_element, DocumentKeyValuePair) or isinstance(
        document_element, FRDocumentKeyValuePair
    ):
        return document_element.key.spans + (
            document_element.value.spans if document_element.value else list()
        )
    if hasattr(document_element, "span"):
        return [document_element.span] or list()
    elif hasattr(document_element, "spans"):
        return document_element.spans or list()
    else:
        raise NotImplementedError(
            f"Class {document_element.__class__.__name__} does not contain span field."
        )


@dataclass
class SpanBounds:
    """
    Dataclass representing the outer bounds of a span.
    """

    offset: int
    end: int

    def __hash__(self):
        return hash((self.offset, self.end))


def span_bounds_to_document_span(span_bounds: SpanBounds) -> DocumentSpan:
    """Converts a SpanBounds object to a DocumentSpan object."""
    return DocumentSpan(
        offset=span_bounds.offset, length=span_bounds.end - span_bounds.offset
    )


def document_span_to_span_bounds(
    span: Union[DocumentSpan, FRDocumentSpan]
) -> SpanBounds:
    """Converts a DocumentSpan object to a SpanBounds object."""
    return SpanBounds(span.offset, span.offset + span.length)


def get_span_bounds_to_ordered_idx_mapper(
    analyze_result: Union[AnalyzeResult, FRAnalyzeResult],
) -> Dict[SpanBounds, int]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :returns: A dictionary mapping span bounds to the ordered index within the
        document.
    :rtype: dict
    """
    all_spans: List[SpanBounds] = list()
    for attr in [
        "pages",
        "paragraphs",
        "tables",
        "figures",
        "lists",
        "key_value_pairs",
        "documents",
    ]:
        elements = getattr(analyze_result, attr) or list()
        all_spans.extend([get_min_and_max_span_bounds(elem.spans) for elem in elements])
    for page in analyze_result.pages:
        for attr in ["barcodes", "lines", "words", "selection_marks"]:
            elements = getattr(page, attr) or list()
            for elem in elements:
                if hasattr(elem, "spans"):
                    all_spans.append(get_min_and_max_span_bounds(elem.spans))
                elif hasattr(elem, "span"):
                    all_spans.append(get_min_and_max_span_bounds([elem.span]))

    # Sort by lowest start offset, then largest end offset
    all_spans = sorted(all_spans, key=lambda x: (x.offset, 0 - x.end))

    span_bounds_to_ordered_idx_mapper: Dict[SpanBounds, int] = {
        full_span: idx for idx, full_span in enumerate(all_spans)
    }
    return span_bounds_to_ordered_idx_mapper


class PageSpanCalculator:
    """
    Calculator for determining the page number of a span based on its position.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    """

    def __init__(self, analyze_result: AnalyzeResult):
        self.page_span_bounds: Dict[int, SpanBounds] = self._get_page_span_bounds(
            analyze_result
        )
        self._doc_end_span = self.page_span_bounds[max(self.page_span_bounds)].end

    def determine_span_start_page(self, span_start_offset: int) -> int:
        """
        Determines the page on which a span starts.

        :param span_start_offset: Span starting offset.
        :type span_position: int
        :raises ValueError: Raised when the span_start_offset is greater than
            the last page's end span.
        :return: The page number on which the span starts.
        :rtype: int
        """
        if span_start_offset > self._doc_end_span:
            raise ValueError(
                f"span_start_offset {span_start_offset} is greater than the last page's end span ({self._doc_end_span})."
            )
        page_numbers = [
            k
            for k, v in self.page_span_bounds.items()
            if v.offset <= span_start_offset and v.end >= span_start_offset
        ]
        return min(page_numbers)

    def _get_page_span_bounds(
        self, analyze_result: AnalyzeResult
    ) -> Dict[int, SpanBounds]:
        """
        Gets the span bounds for each page.

        :param analyze_result: AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :raises ValueError: Raised when a gap exists between the span bounds of
            two pages.
        :return: Dictionary with page number as key and tuple of start and end.
        :rtype: Dict[int, SpanBounds]
        """
        page_span_bounds: Dict[int, SpanBounds] = dict()
        page_start_span = 0  # Set first page start to 0
        for page in analyze_result.pages:
            max_page_bound = get_min_and_max_span_bounds(page.spans).end
            max_word_bound = (
                get_min_and_max_span_bounds([page.words[-1].span]).end
                if page.words
                else -1
            )
            max_bound_across_elements = max(max_page_bound, max_word_bound)
            page_span_bounds[page.page_number] = SpanBounds(
                offset=page_start_span,
                end=max_bound_across_elements,
            )
            page_start_span = max_bound_across_elements + 1
        # Check no spans are missed
        last_end_span = -1
        for page_num, span_bounds in page_span_bounds.items():
            if span_bounds.offset != last_end_span + 1:
                raise ValueError(
                    f"Gap exists between span bounds of pages {page_num-1} and {page_num}"
                )
            last_end_span = span_bounds.end

        return page_span_bounds


def get_min_and_max_span_bounds(
    spans: List[Union[DocumentSpan, FRDocumentSpan]]
) -> SpanBounds:
    """
    Get the minimum and maximum offsets of a list of spans.

    :param spans: List of spans to get the offsets for.
    :type spans: List[DocumentSpan]
    :return: Tuple containing the minimum and maximum offsets.
    :rtype: SpanBounds
    """
    min_offset = min([span.offset for span in spans])
    max_offset = max([span.offset + span.length for span in spans])
    return SpanBounds(offset=min_offset, end=max_offset)


## TODO: Create mapper of section ID to section heirarchy, containing more info about the section
# @dataclass
# class SectionHeirarchyInfo:
#     section_id: int
#     section: DocumentSection
#     section_heirarchy: tuple[int]
#     section_heirarchy_incremental_id: tuple[int]

# def get_section_heirarchy_mapper(sections: List[DocumentSection]) -> Dict[int, SectionHeirarchyInfo]:
#     section_span_info_mapper = dict()
#     for section_id, section in enumerate(sections):
#         span_bounds = get_min_and_max_span_bounds(section.spans)
#         section_span_info_mapper[span_bounds] = SectionHeirarchyInfo(
#             section_id=section_id,
#             section=section,
#             section_heirarchy=elem_heirarchy_mapper["sections"].get(section_id, tuple()),
#             section_heirarchy_incremental_id=section_to_incremental_id_mapper.get(section_id, tuple())
#         )
#     return section_span_info_mapper

# For Example usage: section_heirarchy_mapper = get_section_heirarchy_mapper(analyze_result.sections)


@dataclass
class ElementInfo:
    """
    Dataclass containing information about a document element.
    """

    element_id: str
    element: Union[
        DocumentSection,
        DocumentPage,
        DocumentParagraph,
        DocumentLine,
        DocumentWord,
        DocumentFigure,
        DocumentTable,
        DocumentFormula,
        DocumentBarcode,
        DocumentKeyValuePair,
        Document,
        DocumentSelectionMark,
        FRDocumentBarcode,
        FRDocumentFormula,
        FRDocumentKeyValuePair,
        FRDocumentLine,
        FRDocumentPage,
        FRDocumentParagraph,
        FRDocumentSelectionMark,
        FRDocumentTable,
        FRDocumentWord,
    ]
    full_span_bounds: SpanBounds
    spans: List[Union[DocumentSpan, FRDocumentSpan]]
    start_page_number: int
    section_heirarchy_incremental_id: tuple[int]


def get_element_page_number(
    element: Union[
        DocumentPage,
        FRDocumentPage,
        DocumentSection,
        DocumentParagraph,
        FRDocumentParagraph,
        DocumentTable,
        FRDocumentTable,
        DocumentFigure,
        DocumentList,
        DocumentKeyValuePair,
        FRDocumentKeyValuePair,
        Document,
    ]
) -> int:
    """
    Get the page number of a document element.

    :param element: The document element to get the page number of.
    :type element: Union[DocumentPage, DocumentSection, DocumentParagraph,
        DocumentTable, DocumentFigure, DocumentList, DocumentKeyValueElement,
        Document]
    :return: The page number of the document element.
    :rtype: int
    """
    if isinstance(element, DocumentPage) or isinstance(element, FRDocumentPage):
        return element.page_number
    return element.bounding_regions[0].page_number


def get_element_span_info_list(
    analyze_result: "AnalyzeResult",
    page_span_calculator: PageSpanCalculator,
    section_to_incremental_id_mapper: Dict[int, tuple[int]],
) -> Dict[Union[DocumentSpan, FRDocumentSpan], ElementInfo]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :param page_span_calculator: The PageSpanCalculator object to determine the
        page location of a span.
    :type page_span_calculator: PageSpanCalculator
    :param section_to_incremental_id_mapper: A dict containing a mapping of span
        to section heirarchical ID for each object type.
    :type section_to_incremental_id_mapper: Dict[int, tuple[int]]
    :returns: A dictionary with the element ID as key and the order as value.
    """
    # Note: Formula and Barcode elements are not processed by this function as
    # they are processed automatically by the `substitute_content_formulas` and
    # `substitute_content_barcodes` functions.
    # Get info for every unique element
    element_span_info_list = list()
    for attr in [
        "pages",
        "sections",
        "paragraphs",
        "tables",
        "figures",
        "lists",
        "key_value_pairs",
        "documents",
    ]:
        elements: List[
            DocumentPage,
            FRDocumentPage,
            DocumentSection,
            DocumentParagraph,
            FRDocumentParagraph,
            DocumentTable,
            FRDocumentTable,
            DocumentFigure,
            DocumentList,
            DocumentKeyValuePair,
            FRDocumentKeyValuePair,
            Document,
        ] = (
            getattr(analyze_result, attr, list()) or list()
        )
        for elem_idx, element in enumerate(elements):
            spans = get_document_element_spans(element)
            full_span_bounds = get_min_and_max_span_bounds(spans)
            element_span_info_list.append(
                ElementInfo(
                    element_id=f"/{attr}/{elem_idx}",
                    element=element,
                    full_span_bounds=full_span_bounds,
                    spans=spans,
                    start_page_number=page_span_calculator.determine_span_start_page(
                        full_span_bounds.offset
                    ),
                    section_heirarchy_incremental_id=section_to_incremental_id_mapper.get(
                        attr, {}
                    ).get(
                        elem_idx, None
                    ),
                )
            )
    page_sub_element_counter = defaultdict(int)
    for page_num, page in enumerate(analyze_result.pages):
        for attr in ["lines", "words", "selection_marks"]:
            elements: List[
                DocumentLine,
                FRDocumentLine,
                DocumentWord,
                FRDocumentWord,
                DocumentSelectionMark,
                FRDocumentSelectionMark,
            ] = (
                getattr(page, attr) or list()
            )
            for element in elements:
                element_idx = page_sub_element_counter[attr]
                page_sub_element_counter[attr] += 1
                spans = get_document_element_spans(element)
                full_span_bounds = get_min_and_max_span_bounds(spans)
                element_span_info_list.append(
                    ElementInfo(
                        element_id=f"/pages/{page_num}/{attr}/{element_idx}",
                        element=element,
                        full_span_bounds=full_span_bounds,
                        spans=spans,
                        start_page_number=page.page_number,
                        section_heirarchy_incremental_id=None,  # Page elements are not referred to by sections
                    )
                )

    return element_span_info_list


def order_element_info_list(
    element_span_info_list: List[ElementInfo],
) -> List[ElementInfo]:
    """
    Returns an ordered list of element spans based on the element type, page
    number and the start and end of the span. Ordering is done so that higher
    priority elements have their content processed first, ensuring that any
    lower priority elements are ignored if their content is already contained.

    For example, captions, footnotes and text content of a figure should be
    processed before the raw lines/words (to provide more logical outputs and
    avoid duplication of the content).

    :param element_span_info_list: List of ElementInfo objects to order.
    :type element_span_info_list: List[ElementInfo]
    :return: A reordered list of ElementInfo objects.
    :rtype: List[ElementInfo]
    """
    # Assign a priority to each element type so that higher priority elements are ordered first
    # e.g. pages should appear before any contained content, and paragraphs should always be processed before lines or words
    priority_mapper = {
        DocumentPage: 0,
        FRDocumentPage: 0,
        DocumentSection: 1,
        DocumentTable: 2,
        FRDocumentTable: 2,
        DocumentFigure: 2,
        Document: 2,
        DocumentKeyValuePair: 2,
        FRDocumentKeyValuePair: 2,
        DocumentParagraph: 3,
        FRDocumentParagraph: 3,
        DocumentLine: 4,
        FRDocumentLine: 4,
        DocumentSelectionMark: 4,
        FRDocumentSelectionMark: 4,
        DocumentWord: 5,
        FRDocumentWord: 5,
    }

    return sorted(
        element_span_info_list,
        key=lambda x: (
            x.start_page_number,
            x.full_span_bounds.offset,
            priority_mapper[type(x.element)],
            0 - x.full_span_bounds.end,
        ),
    )


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


class SelectionMarkFormatter(ABC):
    """
    Base formatter class for Selection Mark elements.
    """

    @abstractmethod
    def format_content(self, content: str) -> str:
        pass


class DefaultSelectionMarkFormatter(SelectionMarkFormatter):
    """
    Default formatter for Selection Mark elements. This class provides a method
    for formatting the content of a Selection Mark element.

    :param selected_replacement: The text used to replace any selected
        text placeholders within the document (anywhere a ':selected:'
        placeholder exists), defaults to "[X]"
    :type selected_replacement: str, optional
    :param unselected_replacement: The text used to replace any unselected
        text placeholders within the document (anywhere an ':unselected:'
        placeholder exists), defaults to "[ ]"
    :type unselected_replacement: str, optional
    """

    def __init__(
        self, selected_replacement: str = "[X]", unselected_replacement: str = "[ ]"
    ):

        self._selected_replacement = selected_replacement
        self._unselected_replacement = unselected_replacement

    def format_content(self, content: str) -> str:
        """
        Formats text content, replacing any selection mark placeholders with
        the selected or unselected replacement text.

        :param content: Text content to format.
        :type content: str
        :return: Formatted text content.
        :rtype: str
        """
        return content.replace(":selected:", self._selected_replacement).replace(
            ":unselected:", self._unselected_replacement
        )


class DocumentElementProcessor(ABC):
    """
    Base processor class for all Document elements extracted by Document
    Intelligence.
    """

    expected_elements = []

    def _validate_element_type(self, element_info: ElementInfo):
        """
        Validates that the Document element contained by the ElementInfo object
        is of the expected type.

        :param element_info: Information about the Document element.
        :type element_info: ElementInfo
        :raises ValueError: If the element of the ElementInfo object is not of
            the expected type.
        """
        if not self.expected_elements:
            raise ValueError("expected_element has not been set for this processor.")

        if not any(
            isinstance(element_info.element, expected_element)
            for expected_element in self.expected_elements
        ):
            raise ValueError(
                f"Element is incorrect type - {type(element_info.element)}. It should be one of `{self.expected_elements}` types."
            )

    def _format_str_contains_placeholders(
        self, format_str: str, expected_placeholders: List[str]
    ) -> bool:
        """
        Returns True if the format string contains any of the expected placeholders.

        :param format_str: Formatting string to check.
        :type format_str: str
        :param expected_placeholders: A list of expected placeholders.
        :type expected_placeholders: List[str]
        :return: Whether the string contains any of the expected placeholders.
        :rtype: bool
        """
        return any([placeholder in format_str for placeholder in expected_placeholders])


class DocumentSectionProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentSection elements.
    """

    expected_elements = [DocumentSection]

    @abstractmethod
    def convert_section(
        self,
        element_info: ElementInfo,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for processing section elements.
        """
        pass


class DefaultDocumentSectionProcessor(DocumentSectionProcessor):
    """
    The default processor for DocumentSection elements. This class provides
    methods for converting a DocumentSection element into Haystack Documents
    that contain section information.

    The text_format string can contain placeholders for the following:
    - {section_incremental_id}: The heirarchical ID of the section
        (e.g. (1.2.4)).

    :param text_format: A text format string which defines how the content
        should be extracted from the element. If set to None, no text content
        will be extracted from the element.
    :type text_format: str, optional
    :param max_heirarchy_depth: The maximum depth of the heirarchy to include in
        the section incremental ID. If a section has an ID that is deeper than
        max_heirarchy_depth, it's section ID will be ignored. If None, all
        section ID levels will be included when printing section IDs.
    :type max_heirarchy_depth: int, optional
    """

    def __init__(
        self,
        text_format: Optional[str] = None,
        max_heirarchy_depth: Optional[int] = 3,
    ):
        self.text_format = text_format
        # If max_depth is None, ensure it is set to a high number
        if max_heirarchy_depth < 0:
            raise ValueError("max_depth must be a positive integer or None.")
        self.max_heirarchy_depth = (
            max_heirarchy_depth if max_heirarchy_depth is not None else 999
        )

    def convert_section(
        self,
        element_info: ElementInfo,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentSection element into a Haystack Document.

        :param element_info: Information about the DocumentSection element.
        :type element_info: ElementInfo
        :param section_heirarchy: The heirarchical ID of the section.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the section information.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        # Section incremental ID can be empty for the first section representing
        # the entire document. Only output text if values exist, and only if max_depth
        # is not exceeded. We want to avoid printing the same section ID -
        # e.g. (3, 1) should become 3.1 but (3, 1, 1) should be ignored if
        # max_depth is only 2.
        if self.text_format and element_info.section_heirarchy_incremental_id:
            if (
                len(element_info.section_heirarchy_incremental_id)
                <= self.max_heirarchy_depth
            ):
                section_incremental_id_text = ".".join(
                    str(i)
                    for i in element_info.section_heirarchy_incremental_id[
                        : self.max_heirarchy_depth
                    ]
                )
            else:
                section_incremental_id_text = ""
            formatted_text = self.text_format.format(
                section_incremental_id=section_incremental_id_text
            )
            formatted_if_null = self.text_format.format(section_incremental_id="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=formatted_text,
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()


class DocumentPageProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentPage elements.
    """

    expected_elements = [DocumentPage, FRDocumentPage]

    @abstractmethod
    def export_page_img(
        self, pdf_page_img: PILImage, di_page: DocumentPage
    ) -> TransformedImage:
        """
        Method for exporting the page image and applying transformations.
        """
        pass

    @abstractmethod
    def convert_page_start(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting content for the beginning of the page.
        """
        pass

    @abstractmethod
    def convert_page_end(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting content for the end of the page.
        """
        pass


class DefaultDocumentPageProcessor(DocumentPageProcessor):
    """
    The default processor for DocumentPage elements. This class provides
    methods for converting a DocumentPage element into Haystack Documents
    that contain the text and page image content.

    The text format string can contain placeholders for the following:
    - {page_number}: The 1-based page number of the page.

    :param page_start_text_formats: Text formats to be used to export text
        content that should be shown at the start of the page. If None, no text
        content will be exported at the start of the page.
    :type page_start_text_formats: List[str], optional
    :param page_end_text_formats: Text formats to be used to export text
        content that should be shown at the end of the page. If None, no text
        content will be exported at the end of the page.
    :type page_end_text_formats: str, optional
    :param page_img_order: Whether to export the page image 'before' or 'after'
        the rest of the page content, or not at all ('None').
    :type page_img_order: Literal["before", "after"], optional
    :param page_img_text_intro: Introduction text to be paired with the page
        image, defaults to "*Page Image:*"
    :type page_img_text_intro: str, optional
    :param img_export_dpi: DPI of the exported page image, defaults to 100
    :type img_export_dpi: int, optional
    :param adjust_rotation: Whether to automatically adjust the image based on
        the page angle detected by Document Intelligence, defaults to True
    :type adjust_rotation: bool, optional
    :param rotated_fill_color: Defines the fill color to be used if the page
        image is rotated and new pixels are added to the image, defaults to
        (255, 255, 255)
    :type rotated_fill_color: Tuple[int], optional
    """

    expected_format_placeholders = ["{page_number}"]

    def __init__(
        self,
        page_start_text_formats: Optional[List[str]] = [
            "\n*Page {page_number} content:*\n"
        ],
        page_end_text_formats: Optional[str] = None,
        page_img_order: Optional[Literal["before", "after"]] = "after",
        page_img_text_intro: Optional[str] = "*Page {page_number} Image:*",
        img_export_dpi: int = 100,
        adjust_rotation: bool = True,
        rotated_fill_color: Tuple[int] = (255, 255, 255),
    ):
        self.page_start_text_formats = page_start_text_formats
        self.page_end_text_formats = page_end_text_formats
        self.page_img_order = page_img_order
        self.page_img_text_intro = page_img_text_intro
        self.img_export_dpi = img_export_dpi
        self.adjust_rotation = adjust_rotation
        self.rotated_fill_color = rotated_fill_color

    def export_page_img(
        self, pdf_page_img: PILImage, di_page: DocumentPage
    ) -> TransformedImage:
        """
        Export the page image and apply transformations.

        :param pdf_page_img: PIL Image of the page.
        :type pdf_page_img: PIL.Image.Image
        :param di_page: DocumentPage object.
        :type di_page: DocumentPage
        :return: Transformed image object containing the original and
            transformed image information.
        :rtype: TransformedImage
        """
        if self.adjust_rotation:
            transformed_img = rotate_img_pil(
                pdf_page_img, di_page.angle or 0, self.rotated_fill_color
            )
        else:
            transformed_img = pdf_page_img
        return TransformedImage(
            image=transformed_img,
            orig_image=pdf_page_img,
            rotation_applied=di_page.angle if self.adjust_rotation else None,
        )

    def convert_page_start(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Exports Haystack Documents for content to be shown at the start of the
        page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param section_heirarchy: Section heirarchy for the page.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the page content.
        :rtype: List[Document]
        """
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "page_location": "start",
            "section_heirarchy": section_heirarchy,
        }
        if self.page_img_order == "before":
            outputs.extend(
                self._export_page_img_docs(element_info, transformed_page_img, meta)
            )
        if self.page_start_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_start_text_formats, meta
                )
            )
        return outputs

    def convert_page_end(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[Document]:
        """
        Exports Haystack Documents for content to be shown at the end of the
        page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param section_heirarchy: Section heirarchy for the page.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the page content.
        :rtype: List[Document]
        """
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "page_location": "end",
            "section_heirarchy": section_heirarchy,
        }
        if self.page_end_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_end_text_formats, meta
                )
            )
        if self.page_img_order == "after":
            outputs.extend(
                self._export_page_img_docs(element_info, transformed_page_img, meta)
            )
        return outputs

    def _export_page_img_docs(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the page image as a HaystackDocument.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param meta: Metadata to include in the output documents.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the image content.
        :rtype: List[HaystackDocument]
        """
        # Get transformed image, copying over image transformation metadata
        img_bytestream = HaystackByteStream(
            data=pil_img_to_base64(transformed_page_img.image),
            mime_type="image/jpeg",
        )
        meta["rotation_applied"] = transformed_page_img.rotation_applied
        # Create output docs
        img_outputs = list()
        if self.page_img_text_intro:
            page_intro_content = self.page_img_text_intro.format(
                page_number=element_info.start_page_number
            )
        else:
            page_intro_content = None
        img_outputs.append(
            HaystackDocument(
                id=f"{element_info.element_id}_img",
                content=page_intro_content,
                blob=img_bytestream,
                meta=meta,
            )
        )
        return img_outputs

    def _export_page_text_docs(
        self, element_info: ElementInfo, text_formats: List[str], meta: Dict[str, Any]
    ) -> List[HaystackDocument]:
        """
        Exports text documents for the page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param text_formats: List of text formats to use.
        :type text_formats: List[str]
        :param meta: Metadata to include in the output documents.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the text content.
        :rtype: List[HaystackDocument]
        """
        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(page_number=element_info.start_page_number)
            formatted_if_null = format.format(page_number="")
            if formatted_text != formatted_if_null or not has_format_placeholders:
                output_strings.append(formatted_text)
        if output_strings:
            # Join outputs together with new lines
            return [
                HaystackDocument(
                    id=f"{element_info.element_id}_text",
                    content="\n".join(output_strings),
                    meta=meta,
                )
            ]
        return list()


def get_heading_hashes(section_heirarchy: Optional[tuple[int]]) -> str:
    """
    Gets the heading hashes for a section heirarchy.

    :param section_heirarchy: Tuple of section IDs representing the section
        heirarchy - For example, (1,1,4).
    :type section_heirarchy: tuple[int], optional
    :return: A string containing the heading hashes.
    :rtype: str
    """
    if section_heirarchy:
        return "#" * len(section_heirarchy)
    return ""


class DocumentParagraphProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentParagraph elements.
    """

    expected_elements = [DocumentParagraph, FRDocumentParagraph]

    @abstractmethod
    def convert_paragraph(
        self,
        element_info: ElementInfo,
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting paragraph content.
        """
        pass


class DefaultDocumentParagraphProcessor(DocumentParagraphProcessor):
    """
    The default processor for DocumentParagraph elements. This class provides a
    method for converting a DocumentParagraph element into a Haystack Document
    object that contains the text content. Since paragraphs can come with
    different roles, the text format string can be set for each possible role.

    The text format string can contain placeholders for the following:
    - {content}: The text content of the element
    - {heading_hashes}: The markdown heading hashes that set the heading
        level of the paragraph. The heading level is based on the section
        heirarchy of the paragraph and calculated automatically.

    :param general_text_format: Text format string for general text content.
    :type general_text_format: str, optional
    :param page_header_format: Text format string for page headers.
    :type page_header_format: str, optional
    :param page_footer_format: Text format string for page footers.
    :type page_footer_format: str, optional
    :param title_format: Text format string for titles.
    :type title_format: str, optional
    :param section_heading_format: Text format string for section headings.
    :type section_heading_format: str, optional
    :param footnote_format: Text format string for footnotes.
    :type footnote_format: str, optional
    :param formula_format: Text format string for formulas.
    :type formula_format: str, optional
    :param page_number_format: Text format string for page numbers.
    :type page_number_format: str, optional
    """

    expected_format_placeholders = ["{content}", "{heading_hashes}"]

    def __init__(
        self,
        general_text_format: Optional[str] = "{content}",
        page_header_format: Optional[str] = None,
        page_footer_format: Optional[str] = None,
        title_format: Optional[str] = "\n{heading_hashes} **{content}**",
        section_heading_format: Optional[str] = "\n{heading_hashes} **{content}**",
        footnote_format: Optional[str] = "*Footnote:* {content}",
        formula_format: Optional[str] = "*Formula:* {content}",
        page_number_format: Optional[str] = None,
    ):
        self.paragraph_format_mapper = {
            None: general_text_format,
            ParagraphRole.PAGE_HEADER: page_header_format,
            ParagraphRole.PAGE_FOOTER: page_footer_format,
            ParagraphRole.TITLE: title_format,
            ParagraphRole.SECTION_HEADING: section_heading_format,
            ParagraphRole.FOOTNOTE: footnote_format,
            ParagraphRole.FORMULA_BLOCK: formula_format,
            ParagraphRole.PAGE_NUMBER: page_number_format,
        }

    def convert_paragraph(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a paragraph element into a Haystack Document object containing
        the text content of the paragraph.

        :param element_info: Element information for the paragraph.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the line.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the paragraph content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        format_mapper = self.paragraph_format_mapper[element_info.element.role]
        heading_hashes = get_heading_hashes(
            element_info.section_heirarchy_incremental_id
        )
        content = replace_content_formulas_and_barcodes(
            element_info.element.content,
            element_info.element.spans,
            all_formulas,
            all_barcodes,
        )
        content = selection_mark_formatter.format_content(content)
        if format_mapper:
            formatted_text = format_mapper.format(
                heading_hashes=heading_hashes, content=content
            )
            formatted_if_null = format_mapper.format(heading_hashes="", content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=formatted_text,
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()


class DocumentLineProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentLine elements.
    """

    expected_elements = [DocumentLine, FRDocumentLine]

    @abstractmethod
    def convert_line(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting line content.
        """
        pass


class DefaultDocumentLineProcessor(DocumentLineProcessor):
    """
    The default processor for DocumentLine elements. This class provides a
    method for converting a DocumentLine element into a Haystack Document object
    that contains the text content.

    The text format string can contain placeholders for the following:
    - {content}: The text content of the element

    :param text_format: A text format string which defines
        the content which should be extracted from the element. If set to None,
        no text content will be extracted from the element.
    :type text_format: str, optional
    """

    def __init__(
        self,
        text_format: Optional[str] = "{content}",
    ):
        self.text_format = text_format

    def convert_line(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a line element into a Haystack Document object containing the
        text content of the line.

        :param element_info: Element information for the line.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the line.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the line content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        content = replace_content_formulas_and_barcodes(
            element_info.element.content,
            element_info.element.spans,
            all_formulas,
            all_barcodes,
        )
        content = selection_mark_formatter.format_content(content)
        if self.text_format:
            formatted_text = self.text_format.format(content=content)
            formatted_if_null = self.text_format.format(content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=formatted_text,
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()


class DocumentWordProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentWord elements.
    """

    expected_elements = [DocumentWord, FRDocumentWord]

    @abstractmethod
    def convert_word(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting word content.
        """
        pass


class DefaultDocumentWordProcessor(DocumentWordProcessor):
    """
    The default processor for DocumentWord elements. This class provides a
    method for converting a DocumentWord element into a Haystack Document object
    that contains the text content.

    The text format string can contain placeholders for the following:
    - {content}: The text content of the element

    :param text_format: A text format string which defines
        the content which should be extracted from the element. If set to None,
        no text content will be extracted from the element.
    :type text_format: str, optional
    """

    def __init__(
        self,
        text_format: Optional[str] = "{content}",
    ):
        self.text_format = text_format

    def convert_word(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a word element into a Haystack Document object containing the
        text content of the word.

        :param element_info: Element information for the word.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the word.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the word content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        content = replace_content_formulas_and_barcodes(
            element_info.element.content,
            [element_info.element.span],
            all_formulas,
            all_barcodes,
        )
        content = selection_mark_formatter.format_content(content)
        if self.text_format:
            formatted_text = self.text_format.format(content=content)
            formatted_if_null = self.text_format.format(content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=self.text_format.format(content=content),
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()


class DocumentKeyValuePairProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentKeyValuePair elements.
    """

    expected_elements = [DocumentKeyValuePair, FRDocumentKeyValuePair]

    @abstractmethod
    def convert_kv_pair(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting Key Value pairs.
        """
        pass


class DefaultDocumentKeyValuePairProcessor(DocumentKeyValuePairProcessor):
    """
    The default processor for DocumentKeyValuePair elements. This class provides
    a method for converting a DocumentKeyValuePair element into a
    Haystack Document object that contains the text content.

    The text format string can contain placeholders for the following:
    - {key_content}: The text content of the key.
    - {value_content}: The text content of the value.

    :param text_format: Text format string for text content.
    :type text_format: str, optional
    """

    expected_format_placeholders = ["{key_content}", "{value_content}"]

    def __init__(
        self,
        text_format: Optional[str] = "*Key Value Pair*: {key_content}: {value_content}",
    ):
        self.text_format = text_format

    def convert_kv_pair(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentKeyValuePair element into a Haystack Document object
        containing the text contents of the element.

        :param element_info: Element information for the KV pair.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the word.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the KV pair content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        key_content = replace_content_formulas_and_barcodes(
            element_info.element.key.content,
            element_info.element.key.spans,
            all_formulas,
            all_barcodes,
        )
        key_content = selection_mark_formatter.format_content(key_content)
        value_content = replace_content_formulas_and_barcodes(
            element_info.element.value.content,
            element_info.element.value.spans,
            all_formulas,
            all_barcodes,
        )
        value_content = selection_mark_formatter.format_content(value_content)
        if self.text_format:
            formatted_text = self.text_format.format(
                key_content=key_content, value_content=value_content
            )
            formatted_if_null = self.text_format.format(
                key_content="", value_content=""
            )
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=formatted_text,
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()


class DocumentTableProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentTable element.
    """

    expected_elements = [DocumentTable, FRDocumentTable]

    @abstractmethod
    def convert_table(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting table elements.
        """
        pass


class DefaultDocumentTableProcessor(DocumentTableProcessor):
    """
    The default processor for DocumentTable elements. This class provides
    methods for converting DocumentTable elements into Haystack Document objects
    for both the text and table/dataframe content of a table.

    Text format strings can contain placeholders for the following:
    - {table_number}: The table number (ordered from the beginning of the
        document).
    - {caption}: The table's caption (if available).
    - {footnotes}: The table's footnotes (if available).

    This processor outputs multiple Documents for each figure:
    1. Before-table text: A list of text content documents that are output
        prior to the table itself.
    2. Table: The content of the table as a Haystack Document containing the
        table's content as both a markdown string (content field) and a
        dataframe (dataframe field).
    3. After-table text: A list of text content documents that are output
        following the table itself.

    It is recommended to output most of the table content (captions, content,
    footnotes etc) prior to the table itself.

    :param before_table_text_formats: A list of text format strings which define
        the content which should be extracted from the element prior to the
        table itself. If None, no text content will be extracted prior to the
        image.
    :type before_table_text_formats: List[str], optional
    :param after_table_text_formats: : A list of text format strings which define
        the content which should be extracted from the element following the
        table itself. If None, no text content will be extracted after the
        table.
    :type after_table_text_formats: List[str], optional
    """

    expected_format_placeholders = ["{table_number}", "{caption}", "{footnotes}"]

    def __init__(
        self,
        before_table_text_formats: Optional[List[str]] = [
            "**Table {table_number} Info**\n",
            "*Table Caption:* {caption}",
            "*Table Footnotes:* {footnotes}",
            "*Table Content:*",
        ],
        after_table_text_formats: Optional[List[str]] = None,
    ):
        self.before_table_text_formats = before_table_text_formats
        self.after_table_text_formats = after_table_text_formats

    def convert_table(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a table element into a list of Haystack Documents containing
        the content of the table.

        :param element_info: Element information for the table.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the table.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the table content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "section_heirarchy": section_heirarchy,
        }
        if self.before_table_text_formats:
            outputs.extend(
                self._export_table_text(
                    element_info,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_before_table_text",
                    self.before_table_text_formats,
                    meta,
                )
            )
        # Convert table content
        table_md, table_df = self._convert_table_content(
            element_info.element, all_formulas, all_barcodes, selection_mark_formatter
        )
        outputs.append(
            HaystackDocument(
                id=f"{element_info.element_id}_table",
                content=table_md,
                dataframe=table_df,
                meta=meta,
            )
        )
        if self.after_table_text_formats:
            outputs.extend(
                self._export_table_text(
                    element_info,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_after_table_text",
                    self.after_table_text_formats,
                    meta,
                )
            )

        return outputs

    def _export_table_text(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the text content for a table element.

        :param element_info: Element information for the table.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param id: ID of the table element.
        :type id: str
        :param text_formats: List of text formats to format with the table's
            content.
        :type text_formats: List[str]
        :param meta: Metadata for the table element.
        :type meta: Dict[str, Any]
        :return: A list of HaystackDocument objects containing the table
            content.
        :rtype: List[HaystackDocument]
        """
        table_number_text = get_element_number(element_info)
        caption_text = (
            selection_mark_formatter.format_content(
                replace_content_formulas_and_barcodes(
                    element_info.element.caption.content,
                    element_info.element.caption.spans,
                    all_formulas,
                    all_barcodes,
                )
            )
            if getattr(element_info.element, "caption", None)
            else ""
        )
        footnotes_text = selection_mark_formatter.format_content(
            "\n".join(
                [
                    replace_content_formulas_and_barcodes(
                        footnote.content, footnote.spans, all_formulas, all_barcodes
                    )
                    for footnote in getattr(element_info.element, "footnotes", None)
                    or []
                ]
            )
        )
        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(
                table_number=table_number_text,
                caption=caption_text,
                footnotes=footnotes_text,
            )
            formatted_if_null = format.format(table_number="", caption="", footnotes="")
            if formatted_text != formatted_if_null or not has_format_placeholders:
                output_strings.append(formatted_text)
        if output_strings:
            return [
                HaystackDocument(
                    id=id,
                    content="\n".join(output_strings),
                    meta=meta,
                ),
            ]
        return list()

    def _convert_table_content(
        self,
        table: DocumentTable,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
    ) -> tuple[str, pd.DataFrame]:
        """
        Converts table content into both a markdown string and a pandas
        DataFrame.

        :param table: A DocumentTable element as extracted by Azure Document
            Intelligence.
        :type table: DocumentTable
        :param all_formulas: A list of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :return: A tuple of the markdown string and the pandas DataFrame.
        :rtype: tuple[str, pd.DataFrame]
        """
        ### Create pandas DataFrame
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
            cell_content = selection_mark_formatter.format_content(
                replace_content_formulas_and_barcodes(
                    cell.content, cell.spans, all_formulas, all_barcodes
                )
            )
            cell_dict[cell.row_index].append(cell_content)
        table_df = pd.DataFrame.from_dict(cell_dict, orient="index")
        ### Create markdown text version
        table_fmt = "github"
        # Set index
        if num_index_cols > 0:
            table_df.set_index(list(range(0, num_index_cols)), inplace=True)
        index = num_index_cols > 0
        # Set headers
        if num_header_rows > 0:
            table_df = table_df.T
            table_df.set_index(list(range(0, num_header_rows)), inplace=True)
            table_df = table_df.T
            table_md = table_df.to_markdown(index=index, tablefmt=table_fmt)
        else:
            # Table has no header rows. We will create a dummy header row that has empty cells
            table_df_temp = table_df.copy()
            table_df_temp.columns = ["<!-- -->"] * table_df.shape[1]
            table_md = table_df_temp.to_markdown(index=index, tablefmt=table_fmt)
        # Add new line to end of table markdown
        return "\n" + table_md + "\n\n", table_df


class DocumentFigureProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentFigure elements.
    """

    expected_elements = [DocumentFigure]

    @abstractmethod
    def convert_figure(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting figure elements.
        """
        pass


def get_element_number(element_info: ElementInfo) -> str:
    return str(int(element_info.element_id.split("/")[-1]) + 1)


class DefaultDocumentFigureProcessor(DocumentFigureProcessor):
    """
    The default processor for DocumentFigure elements. This class provides
    methods for converting DocumentFigure elements into Haystack Document
    objects for both the text and image content of a figure.

    Text format strings can contain placeholders for the following:
    - {figure_number}: The figure number (ordered from the beginning of the
        document).
    - {caption}: The figure caption.
    - {footnotes}: The figure footnotes.
    - {content}: Text content from within the figure itself.

    This processor outputs multiple Documents for each figure:
    1. Before-image text: A list of text content documents that are output
        prior to the image itself.
    2. Image: The image content of the figure, optionally containing text
        content that can give context to the image itself.
    3. After-image text: A list of text content documents that are output
        following the image itself.

    It is recommended to output most of the figure content (captions, content,
    footnotes etc) prior to the image itself.

    :param before_figure_text_formats: A list of text format strings which define
        the content which should be extracted from the element prior to the
        image itself. If None, no text content will be extracted prior to the
        image.
    :type before_figure_text_formats: List[str], optional
    :param output_figure_img: Whether to output a cropped image as part of the
        outputs, defaults to True
    :type output_figure_img: bool
    :param figure_img_text_format: An optional text format str to be attached to
        the Document containing the image.
    :type figure_img_text_format: str, optional
    :param after_figure_text_formats: : A list of text format strings which define
        the content which should be extracted from the element following the
        image itself. If None, no text content will be extracted after the
        image.
    :type after_figure_text_formats: List[str], optional
    """

    expected_format_placeholders = [
        "{figure_number}",
        "{caption}",
        "{footnotes}",
        "{content}",
    ]

    def __init__(
        self,
        before_figure_text_formats: Optional[List[str]] = [
            "**Figure {figure_number} Info**\n",
            "*Figure Caption:* {caption}",
            "*Figure Footnotes:* {footnotes}",
            "*Figure Content:*\n{content}",
        ],
        output_figure_img: bool = True,
        figure_img_text_format: Optional[str] = "\n*Figure Image:*",
        after_figure_text_formats: Optional[List[str]] = None,
    ):
        self.before_figure_text_formats = before_figure_text_formats
        self.output_figure_img = output_figure_img
        self.figure_img_text_format = figure_img_text_format
        self.after_figure_text_formats = after_figure_text_formats

    def convert_figure(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentFigure element into a list of HaystackDocument
        objects, containing the text and image content for the figure.

        :param element_info: ElementInfo object for the figure.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image of the page containing
            the figure.
        :type transformed_page_img: TransformedImage
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: Section heirarchy of the figure.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the figure content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        page_numbers = list(
            sorted(
                {region.page_number for region in element_info.element.bounding_regions}
            )
        )
        if len(page_numbers) > 1:
            logger.warning(
                f"Figure spans multiple pages. Only the first page will be used. Page numbers: {page_numbers}"
            )
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "section_heirarchy": section_heirarchy,
        }
        if self.before_figure_text_formats:
            outputs.extend(
                self._export_figure_text(
                    element_info,
                    analyze_result,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_before_figure_text",
                    self.before_figure_text_formats,
                    meta,
                )
            )
        if self.output_figure_img:
            page_element = analyze_result.pages[
                page_numbers[0] - 1
            ]  # Convert 1-based page number to 0-based index
            figure_img = self.convert_figure_to_img(
                element_info.element, transformed_page_img, page_element
            )
            figure_img_text_docs = self._export_figure_text(
                element_info,
                analyze_result,
                all_formulas,
                all_barcodes,
                selection_mark_formatter,
                id="NOT_REQUIRED",
                text_formats=[self.figure_img_text_format],
                meta={},
            )
            figure_img_text = (
                figure_img_text_docs[0].content if figure_img_text_docs else ""
            )
            outputs.append(
                HaystackDocument(
                    id=f"{element_info.element_id}_img",
                    content=figure_img_text,
                    blob=HaystackByteStream(
                        data=pil_img_to_base64(figure_img),
                        mime_type="image/jpeg",
                    ),
                    meta=meta,
                )
            )
        if self.after_figure_text_formats:
            outputs.extend(
                self._export_figure_text(
                    element_info,
                    analyze_result,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_after_figure_text",
                    self.after_figure_text_formats,
                    meta,
                )
            )

        return outputs

    def _export_figure_text(
        self,
        element_info: ElementInfo,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the text content of a figure element.

        :param element_info: ElementInfo object for the figure.
        :type element_info: ElementInfo
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param id: ID of the figure element.
        :type id: str
        :param text_formats: List of text formats to use.
        :type text_formats: List[str]
        :param meta: Metadata for the figure element.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the text content.
        :rtype: List[HaystackDocument]
        """
        figure_number_text = get_element_number(element_info)
        caption_text = selection_mark_formatter.format_content(
            replace_content_formulas_and_barcodes(
                element_info.element.caption.content,
                element_info.element.caption.spans,
                all_formulas,
                all_barcodes,
            )
            if element_info.element.caption
            else ""
        )
        if element_info.element.footnotes:
            footnotes_text = selection_mark_formatter.format_content(
                "\n".join(
                    [
                        replace_content_formulas_and_barcodes(
                            footnote.content, footnote.spans, all_formulas, all_barcodes
                        )
                        for footnote in element_info.element.footnotes
                    ]
                )
            )
        else:
            footnotes_text = ""
        # Get text content only if it is necessary (it is a slow operation)
        content_text = selection_mark_formatter.format_content(
            self._get_figure_text_content(
                element_info.element,
                analyze_result,
                all_formulas,
                all_barcodes,
                selection_mark_formatter,
            )
            if any("{content}" in format for format in text_formats)
            else ""
        )

        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(
                figure_number=figure_number_text,
                caption=caption_text,
                footnotes=footnotes_text,
                content=content_text,
            )
            formatted_if_null = format.format(
                figure_number="", caption="", footnotes="", content=""
            )
            if formatted_text != formatted_if_null or not has_format_placeholders:
                output_strings.append(formatted_text)
        if output_strings:
            return [
                HaystackDocument(
                    id=id,
                    content="\n".join(output_strings),
                    meta=meta,
                ),
            ]
        return list()

    def _get_figure_text_content(
        self,
        figure_element: DocumentFigure,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
    ) -> str:
        """
        Gets the text content of a figure element. This method automatically
        excludes the caption and footnotes from the text content.

        :param figure_element: Figure element to extract text content from.
        :type figure_element: DocumentFigure
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :return: The text content of the figure element.
        :rtype: str
        """
        # Identify figure content spans
        caption_spans = figure_element.caption.spans if figure_element.caption else []
        footnote_spans = list(
            itertools.chain.from_iterable(
                [footnote.spans for footnote in figure_element.footnotes or []]
            )
        )
        content_spans = [
            span
            for span in figure_element.spans
            if span not in caption_spans and span not in footnote_spans
        ]
        figure_page_numbers = [
            region.page_number for region in figure_element.bounding_regions
        ]

        content_min_max_span_bounds = get_min_and_max_span_bounds(content_spans)

        output_line_strings = list()
        current_line_strings = list()
        matched_spans = list()
        for content_span in content_spans:
            content_min_max_span_bounds = get_min_and_max_span_bounds(content_spans)
            span_perfect_match = False
            for para in analyze_result.paragraphs or []:
                if para.spans[0].offset > content_min_max_span_bounds.end:
                    break
                span_perfect_match = para.spans[0] == content_span
                span_perfect_match = False
                if span_perfect_match or all(
                    is_span_in_span(para_span, content_span) for para_span in para.spans
                ):
                    matched_spans.extend(para.spans)
                    if current_line_strings:
                        output_line_strings.append(" ".join(current_line_strings))
                        current_line_strings = list()
                    output_line_strings.append(
                        selection_mark_formatter.format_content(
                            replace_content_formulas_and_barcodes(
                                para.content, para.spans, all_formulas, all_barcodes
                            )
                        )
                    )
            if span_perfect_match:
                continue
            for page_number in figure_page_numbers:
                for line in analyze_result.pages[page_number - 1].lines or []:
                    # If we have already matched the span, we can skip the rest of the lines
                    if line.spans[0].offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    if any(
                        is_span_in_span(line_span, matched_span)
                        for matched_span in matched_spans
                        for line_span in line.spans
                    ):
                        continue
                    span_perfect_match = line.spans[0] == content_span
                    if span_perfect_match or all(
                        is_span_in_span(line_span, content_span)
                        for line_span in line.spans
                    ):
                        matched_spans.extend(line.spans)
                        if current_line_strings:
                            output_line_strings.append(" ".join(current_line_strings))
                            current_line_strings = list()
                        output_line_strings.append(
                            selection_mark_formatter.format_content(
                                replace_content_formulas_and_barcodes(
                                    line.content, line.spans, all_formulas, all_barcodes
                                )
                            )
                        )
                if span_perfect_match:
                    continue
                for word in analyze_result.pages[page_number - 1].words or []:
                    # If we have already matched the span, we can skip the rest of the lines
                    if word.span.offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    if any(
                        is_span_in_span(word.span, matched_span)
                        for matched_span in matched_spans
                    ):
                        continue
                    span_perfect_match = word.span == content_span
                    if span_perfect_match or is_span_in_span(word.span, content_span):
                        current_line_strings.append(
                            selection_mark_formatter.format_content(
                                replace_content_formulas_and_barcodes(
                                    word.content,
                                    [word.span],
                                    all_formulas,
                                    all_barcodes,
                                )
                            )
                        )
            if current_line_strings:
                output_line_strings.append(" ".join(current_line_strings))
                current_line_strings = list()
        if output_line_strings:
            return "\n".join(output_line_strings)
        return ""

    def convert_figure_to_img(
        self,
        figure: DocumentFigure,
        transformed_page_img: TransformedImage,
        page_element: DocumentPage,
    ) -> PILImage:
        """
        Converts a figure element to a cropped image.

        :param figure: DocumentFigure element to be converted.
        :type figure: DocumentFigure
        :param transformed_page_img: TransformedImage object of the page
            containing the figure.
        :type transformed_page_img: TransformedImage
        :param page_element: Page object which contains the figure.
        :type page_element: DocumentPage
        :return: a PIL Image of the cropped figure.
        :rtype: PILImage
        """
        di_page_dimensions = (page_element.width, page_element.height)
        pil_img_dimensions = (
            transformed_page_img.orig_image.width,
            transformed_page_img.orig_image.height,
        )
        # Adjust DI page coordinates from inch-based to pixel-based
        unique_pages = set([br.page_number for br in figure.bounding_regions])
        if len(unique_pages) > 1:
            logger.warning(
                f"Figure contains bounding regions across multiple pages ({unique_pages}). Only image content from the first page will be used."
            )
        figure_polygons = [
            br.polygon
            for br in figure.bounding_regions
            if br.page_number == page_element.page_number
        ]
        # Scale polygon coordinates from DI page to PIL image
        scaled_polygons = [
            scale_flat_poly_list(
                figure_polygon,
                di_page_dimensions,
                pil_img_dimensions,
            )
            for figure_polygon in figure_polygons
        ]
        pixel_polygon = get_flat_poly_lists_convex_hull(scaled_polygons)
        # If the page image has been transformed, adjust the DI page bounding boxes
        if transformed_page_img.rotation_applied:
            pixel_polygon = rotate_polygon(
                pixel_polygon,
                transformed_page_img.rotation_applied,
                transformed_page_img.orig_image.width,
                transformed_page_img.orig_image.height,
            )
        return crop_img(
            transformed_page_img.image,
            pixel_polygon,
        )


class DocumentIntelligenceProcessor:
    def __init__(
        self,
        page_processor: DocumentPageProcessor = DefaultDocumentPageProcessor(),
        section_processor: DocumentSectionProcessor = DefaultDocumentSectionProcessor(),
        table_processor: DocumentTableProcessor = DefaultDocumentTableProcessor(),
        figure_processor: DocumentFigureProcessor = DefaultDocumentFigureProcessor(),
        key_value_pair_processor: DocumentKeyValuePairProcessor = DefaultDocumentKeyValuePairProcessor(),
        paragraph_processor: DocumentParagraphProcessor = DefaultDocumentParagraphProcessor(),
        line_processor: DocumentLineProcessor = DefaultDocumentLineProcessor(),
        word_processor: DocumentWordProcessor = DefaultDocumentWordProcessor(),
        selection_mark_formatter: SelectionMarkFormatter = DefaultSelectionMarkFormatter(),
    ):
        self._page_processor = page_processor
        self._section_processor = section_processor
        self._table_processor = table_processor
        self._figure_processor = figure_processor
        self._key_value_pair_processor = key_value_pair_processor
        self._paragraph_processor = paragraph_processor
        self._line_processor = line_processor
        self._word_processor = word_processor
        self._selection_mark_formatter = selection_mark_formatter

    def process_analyze_result(
        self,
        analyze_result: Union[AnalyzeResult, FRAnalyzeResult],
        doc_page_imgs: Optional[Dict[int, PILImage]] = None,
        on_error: Literal["ignore", "raise"] = "ignore",
        break_after_element_idx: Optional[int] = None,
    ) -> List[HaystackDocument]:
        """
        Processes the result of a Document Intelligence analyze operation and
        returns the extracted content as a list of Haystack Documents. This
        output is ready for splitting into separate chunks and/or use with LLMs.

        If image outputs are configured for the page or figure processors, the
        source PDF must be provided using either the `pdf_path` or `pdf_url`
        parameters.

        This process maintains the order of the content as it appears in the
        source document, with the configuration of each separate element
        processor responsible for how the content is processed.

        :param analyze_result: The result of an analyze operation.
        :type analyze_result: Union[AnalyzeResult, FormRecognizerAnalyzeResult]
        :param pdf_path: Local path of the PDF, defaults to None
        :type pdf_path: Union[str, os.PathLike], optional
        :param pdf_url: URL path to PDF, defaults to None
        :type pdf_url: str, optional
        :param on_error: How to handle errors, defaults to "ignore"
        :type on_error: Literal["ignore", "raise"], optional
        :param break_after_element_idx: If provided, this will break the
            processing loop after this many items. defaults to None
        :type break_after_element_idx: int, optional
        :returns: A list of Haystack Documents containing the processed content.
        :rtype: List[HaystackDocument]
        """
        if doc_page_imgs is not None:
            # Check page image IDs match the DI page IDs
            di_page_ids = {page.page_number for page in analyze_result.pages}
            if not doc_page_imgs.keys() == di_page_ids:

                raise ValueError(
                    "doc_page_imgs keys do not match DI page numbers. doc_page_imgs Keys: {}, DI Page Numbers: {}".format(
                        f"({min(doc_page_imgs.keys())} -> {max(doc_page_imgs.keys())})",
                        f"{min(di_page_ids)} -> {max(di_page_ids)}",
                    )
                )
        # Get mapper of element to section heirarchy
        elem_heirarchy_mapper = get_element_heirarchy_mapper(analyze_result)
        section_to_incremental_id_mapper = (
            convert_element_heirarchy_to_incremental_numbering(elem_heirarchy_mapper)
        )

        page_span_calculator = PageSpanCalculator(analyze_result)

        # # Get list of all spans in order
        element_span_info_list = get_element_span_info_list(
            analyze_result, page_span_calculator, section_to_incremental_id_mapper
        )
        ordered_element_span_info_list = order_element_info_list(element_span_info_list)

        all_formulas = get_all_formulas(analyze_result)
        all_barcodes = get_all_barcodes(analyze_result)

        # Create outputs
        full_output_list: List[HaystackDocument] = list()

        transformed_page_imgs: dict[TransformedImage] = dict()  # 1-based page numbers
        current_page_info = None
        current_section_heirarchy_incremental_id = None
        # Keep track of all spans already processed on the page. We will use this to
        # skip low-priority elements that may already be contained in higher-priority
        # elements (e.g. ignoring paragraphs/lines/words that already appear in a table
        # or figure, or lines/words that were already part of a paragraph).
        current_page_priority_spans: List[Union[DocumentSpan, FRDocumentSpan]] = list()
        unprocessed_element_counter = defaultdict(int)
        # Work through all elements and add to output
        for element_idx, element_info in enumerate(ordered_element_span_info_list):
            try:
                # Skip lower priority elements if their content is already processed as part of a higher-priority element
                if any(
                    isinstance(element_info.element, element_type)
                    for element_type in [
                        DocumentKeyValuePair,
                        FRDocumentKeyValuePair,
                        DocumentParagraph,
                        FRDocumentParagraph,
                        DocumentLine,
                        FRDocumentLine,
                        DocumentWord,
                        FRDocumentWord,
                    ]
                ):
                    span_already_processed = False
                    for element_span in element_info.spans:
                        if any(
                            is_span_in_span(element_span, processed_span)
                            for processed_span in current_page_priority_spans
                        ):
                            span_already_processed = True
                            break
                    if span_already_processed:
                        continue
                # Output page end outputs if the page has changed
                if (
                    current_page_info is not None
                    and element_info.start_page_number
                    > current_page_info.start_page_number
                ):
                    full_output_list.extend(
                        self._page_processor.convert_page_end(
                            current_page_info,
                            transformed_page_imgs[
                                current_page_info.element.page_number
                            ],
                            current_section_heirarchy_incremental_id,
                        )
                    )
                    # Remove all spans that end before the current page does.
                    current_page_priority_spans = [
                        span
                        for span in current_page_priority_spans
                        if document_span_to_span_bounds(span).offset
                        > current_page_info.full_span_bounds.end
                    ]
                ### Process new elements. The order of element types in this if/else loop matches
                ### the ordering by `ordered_element_span_info_list` and should not be changed.
                if isinstance(element_info.element, DocumentSection):
                    current_section_heirarchy_incremental_id = (
                        element_info.section_heirarchy_incremental_id
                    )
                    full_output_list.extend(
                        self._section_processor.convert_section(
                            element_info, current_section_heirarchy_incremental_id
                        )
                    )
                    # Skip adding section span to priority spans (this would skip all contained content)
                    continue
                elif isinstance(element_info.element, DocumentPage) or isinstance(
                    element_info.element, FRDocumentPage
                ):
                    # Export page image for use by this and other processors (e.g. page and figure processors)
                    transformed_page_imgs[element_info.element.page_number] = (
                        self._page_processor.export_page_img(
                            pdf_page_img=doc_page_imgs[
                                element_info.element.page_number
                            ],
                            di_page=element_info.element,
                        )
                    )
                    current_page_info = element_info
                    full_output_list.extend(
                        self._page_processor.convert_page_start(
                            element_info,
                            transformed_page_imgs[
                                current_page_info.element.page_number
                            ],
                            current_section_heirarchy_incremental_id,
                        )
                    )
                    continue  # Skip adding span to priority spans
                # Process high priority elements with text content
                elif isinstance(element_info.element, DocumentTable) or isinstance(
                    element_info.element, FRDocumentTable
                ):
                    full_output_list.extend(
                        self._table_processor.convert_table(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentFigure):
                    full_output_list.extend(
                        self._figure_processor.convert_figure(
                            element_info,
                            transformed_page_imgs[element_info.start_page_number],
                            analyze_result,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(
                    element_info.element, DocumentSelectionMark
                ) or isinstance(element_info.element, FRDocumentSelectionMark):
                    # Skip selection marks as these are processed by each individual processor
                    continue
                elif isinstance(element_info.element, DocumentParagraph) or isinstance(
                    element_info.element, FRDocumentParagraph
                ):
                    full_output_list.extend(
                        self._paragraph_processor.convert_paragraph(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentLine) or isinstance(
                    element_info.element, FRDocumentLine
                ):
                    full_output_list.extend(
                        self._line_processor.convert_line(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentWord) or isinstance(
                    element_info.element, FRDocumentWord
                ):
                    full_output_list.extend(
                        self._word_processor.convert_word(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(
                    element_info.element, DocumentKeyValuePair
                ) or isinstance(element_info.element, FRDocumentKeyValuePair):
                    full_output_list.extend(
                        self._key_value_pair_processor.convert_kv_pair(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                # elif isinstance(element.element, Document):
                #     # TODO: Implement processor
                else:
                    unprocessed_element_counter[
                        element_info.element.__class__.__name__
                    ] += 1
                    raise NotImplementedError(
                        f"Processor for {element_info.element.__class__.__name__} is not supported."
                    )
                # Save span start and end for the current element so we can skip lower-priority elements
                current_page_priority_spans.extend(element_info.spans)
                if (
                    break_after_element_idx is not None
                    and element_idx > break_after_element_idx
                ):
                    logging.info(
                        "{} elements processed (break_after_element_idx={}), breaking loop and returning content.".format(
                            element_idx + 1, break_after_element_idx
                        )
                    )
                    break
            except Exception as _e:
                print(
                    f"Error processing element {element_info.element_id} (start_page_number: {element_info.start_page_number}).\nException: {_e}\nElement info: {element_info}"
                )
                if on_error == "raise":
                    raise
        # All content processed, add the final page output and the last chunk
        if current_page_info is not None:
            full_output_list.extend(
                self._page_processor.convert_page_end(
                    current_page_info,
                    transformed_page_imgs[current_page_info.element.page_number],
                    current_section_heirarchy_incremental_id,
                )
            )
        if unprocessed_element_counter:
            print(
                f"Warning: Some elements were not processed due to a lack of support: {dict(unprocessed_element_counter)}"
            )
        return full_output_list

    def merge_adjacent_text_content_docs(
        self,
        chunk_content_list: Union[List[List[HaystackDocument]], List[HaystackDocument]],
        default_text_merge_separator: str = "\n\n",
    ) -> Union[List[HaystackDocument], List[List[HaystackDocument]]]:
        """
        Merges a list of content chunks together so that adjacent text content
        is combined into a single document, with images and other non-text
        content separated into their own documents. This is useful for
        minimizing the number of separate messages that are required when
        sending the content to an LLM.

        Example result:
        Input documents: [text, text, image, text, table, text, image]
        Output documents: [text, image, text, image]

        :param chunk_content_list: A single list of Haystack Documents, or a
            list of lists of Haystack Documents, where
            each sublist contains the content for a single chunk.
        :type chunk_content_list: Union[List[List[HaystackDocument]], List[HaystackDocument]]
        :param default_text_merge_separator: The default separator to use when
            merging text content, defaults to "\n"
        :type default_text_merge_separator: str, optional
        :return: Returns the content in the same format as the input, with
            adjacent text content merged together.
        :rtype: Union[List[HaystackDocument], List[List[HaystackDocument]]]
        """
        is_input_single_list = all(
            isinstance(list_item, HaystackDocument) for list_item in chunk_content_list
        )
        # If a single list is provided, convert it to a list of lists for processing.
        if is_input_single_list:
            temp_chunk_content_list = [chunk_content_list]
        else:
            temp_chunk_content_list = chunk_content_list
        chunk_outputs = list()

        # Join chunks together
        for chunk in temp_chunk_content_list:
            current_text_snippets: List[str] = list()
            current_chunk_content_list = list()
            for content_doc in chunk:
                doc_type = get_processed_di_doc_type(content_doc)
                if doc_type in [
                    ProcessedDocIntelElementDocumentType.TEXT,
                    ProcessedDocIntelElementDocumentType.TABLE,
                ]:
                    # Content is text-only, so add it to the current chunk
                    current_text_snippets.append(content_doc.content)
                    current_text_snippets.append(
                        content_doc.meta.get(
                            "following_separator", default_text_merge_separator
                        )
                    )
                elif doc_type is ProcessedDocIntelElementDocumentType.IMAGE:
                    # We have hit a non-text document.
                    # Join all text in the current chunk into a single str, then add the image bytestream.
                    current_chunk_content_list.append(
                        HaystackDocument(content="".join(current_text_snippets))
                    )
                    current_text_snippets = list()
                    current_chunk_content_list.append(content_doc)

                else:
                    raise ValueError("Unknown processed DI document type.")
            # Add the last chunk
            if current_text_snippets:
                current_chunk_content_list.append(
                    HaystackDocument(content="".join(current_text_snippets))
                )
            chunk_outputs.append(current_chunk_content_list)
        # Return result in same format as input (either list[Document] or list[list[Document]])
        if is_input_single_list:
            return chunk_outputs[0]
        return chunk_outputs


class ProcessedDocIntelElementDocumentType(Enum):
    """
    Contains the simplified list of element types that result from processing
    Document Intelligence result.
    """

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


def get_processed_di_doc_type(
    processed_di_doc: HaystackDocument,
) -> ProcessedDocIntelElementDocumentType:
    """
    Classifies a processing Document Intelligence document into a simplified
    element type.

    :param processed_di_doc: Document containing processed Document Intelligence
        element content.
    :type processed_di_doc: Document
    :raises ValueError: In case the DI document type is unknown.
    :return: ProcessedDocIntelElementDocumentType Enum.
    :rtype: ProcessedDocIntelElementDocumentType
    """
    if processed_di_doc.dataframe is not None:
        return ProcessedDocIntelElementDocumentType.TABLE
    elif (
        processed_di_doc.blob is not None
        and processed_di_doc.blob.mime_type.startswith("image")
    ):
        return ProcessedDocIntelElementDocumentType.IMAGE
    elif processed_di_doc.content is not None:
        return ProcessedDocIntelElementDocumentType.TEXT
    else:
        raise ValueError(f"Unknown processed DI document type: {processed_di_doc}")


def convert_processed_di_docs_to_markdown(
    processed_content_docs: List[HaystackDocument],
    default_text_merge_separator: str = "\n",
) -> str:
    """
    Converts a list of processed Document Intelligence documents into a single
    Markdown string. This is useful for rendering the content in a human-readable
    format.

    :param processed_content_docs: List of Documents containing processed
        Document Intelligence element content.
    :type processed_content_docs: List[Document]
    :param default_text_merge_separator: Default text separator to be used when
        joining content, defaults to "\n"
    :type default_text_merge_separator: str
    :raises ValueError: In cases where the processed DI document type is
        unknown.
    :return: A single Markdown string containing the processed content.
    :rtype: str
    """
    output_texts = []
    for content_doc in processed_content_docs:
        # Check if content is italicized or bolded, and add newlines if necessary
        text_content_requires_newlines = content_doc.content is not None and any(
            [
                content_doc.content.startswith("*"),
                content_doc.content.endswith("*"),
                content_doc.content.startswith("_"),
            ]
        )
        doc_type = get_processed_di_doc_type(content_doc)
        if doc_type in [
            ProcessedDocIntelElementDocumentType.TEXT,
            ProcessedDocIntelElementDocumentType.TABLE,
        ]:
            # Text and table elements can be directly added to the output
            if text_content_requires_newlines:
                output_texts.append("\n\n")
            output_texts.append(content_doc.content)
            if text_content_requires_newlines:
                output_texts.append("\n")
            output_texts.append(
                content_doc.meta.get(
                    "following_separator", default_text_merge_separator
                )
            )
        elif doc_type is ProcessedDocIntelElementDocumentType.IMAGE:
            # We have hit an image element.
            # Hardcode the image into the markdown output so it can be rendered inline.
            if content_doc.content is not None:
                if text_content_requires_newlines:
                    output_texts.append("\n\n")
                output_texts.append(content_doc.content)
                output_texts.append("\n")
            # Pad image with extra newline before and after image
            output_texts.append("\n")
            output_texts.append(
                f"![]({base64_img_str_to_url(content_doc.blob.data.decode(), content_doc.blob.mime_type)})"
            )
            output_texts.append("\n")
            output_texts.append(
                content_doc.meta.get(
                    "following_separator", default_text_merge_separator
                )
            )
        else:
            raise ValueError("Unknown processed DI document type.")
    return "".join(output_texts).strip()


def convert_processed_di_doc_chunks_to_markdown(
    content_doc_chunks: List[List[HaystackDocument]],
    chunk_prefix: str = "**###### Start of New Chunk ######**",
    default_text_merge_separator: str = "\n",
) -> str:
    """
    Converts a list of lists of processed Document Intelligence documents into
    a single Markdown string, with a prefix shown at the start of each chunk.
    This is useful for rendering the content in a human-readable format.

    :param processed_content_docs: List of lists of Documents containing
        processed  Document Intelligence element content.
    :type processed_content_docs: List[List[Document]]
    :param chunk_prefix: Text prefix to be inserted at the beginning of each
        chunk
    :type chunk_prefix: str
    :param default_text_merge_separator: Default text separator to be used when
        joining content, defaults to "\n"
    :type default_text_merge_separator: str
    :raises ValueError: In cases where the processed DI document type is
        unknown.
    :return: A single Markdown string containing the processed content.
    :rtype: str
    """
    chunked_outputs = list()
    for chunk_docs in content_doc_chunks:
        chunked_outputs.append(chunk_prefix)
        chunked_outputs.append("\n")
        chunked_outputs.append(
            convert_processed_di_docs_to_markdown(
                chunk_docs, default_text_merge_separator=default_text_merge_separator
            )
        )
    return "\n\n".join(chunked_outputs)


class DocumentListSplitter(ABC):
    @abstractmethod
    def split_document_list(
        self, documents: List[HaystackDocument]
    ) -> List[List[HaystackDocument]]:
        """
        Splits a list of Haystack Documents into separate chunks.

        :param documents: A list of Haystack Documents.
        :type documents: List[HaystackDocument]
        :return: A list of lists, where each sublist contains the content for a
            single chunk of content.
        :rtype: List[List[HaystackDocument]]
        """
        pass


class PageDocumentListSplitter(DocumentListSplitter):
    def __init__(self, pages_per_chunk: int = 1):
        self.pages_per_chunk = pages_per_chunk

    def split_document_list(
        self, documents: List[HaystackDocument]
    ) -> List[List[HaystackDocument]]:
        """
        Splits a list of Haystack Documents into separate chunks, where each
        chunk based on the
        specified splitting method.

        :param documents: A list of Haystack Documents.
        :type documents: List[HaystackDocument]
        :param split_on: The method used to split the content, defaults to "page_number"
        :type split_on: Literal[None, "page_number", "section_heirarchy"], optional
        :return: A list of lists, where each sublist contains the content for a
            single chunk of content.
        :rtype: List[List[HaystackDocument]]
        """
        split_outputs = list()
        current_chunk = list()
        current_chunk_page_numbers = set()
        for content_doc in documents:
            current_page_number = content_doc.meta.get("page_number", None)
            num_pages_in_current_chunk = len(current_chunk_page_numbers)
            if (
                current_page_number is not None
                and num_pages_in_current_chunk == self.pages_per_chunk
                and current_page_number not in current_chunk_page_numbers
            ):
                if current_chunk:
                    split_outputs.append(current_chunk)
                    current_chunk = list()
                    current_chunk_page_numbers = set()
            current_chunk_page_numbers.add(current_page_number)
            # Add the current content to the current chunk
            current_chunk.append(content_doc)
        # All content is completed. Add the last chunk
        if current_chunk:
            split_outputs.append(current_chunk)
        return split_outputs


def base64_img_str_to_url(base64_img_str: str, mime_type: str) -> str:
    """
    Converts a base64 image string into a data URL.

    :param base64_img_str: A base64 image string.
    :type base64_img_str: str
    :param mime_type: The MIME type of the image.
    :type mime_type: str
    :return: A data URL for the image.
    :rtype: str
    """
    return f"data:{mime_type};base64,{base64_img_str}"


def convert_processed_di_docs_to_openai_message(
    content_docs: List[HaystackDocument],
    role: Literal["user", "assistant"] = "user",
    img_detail: Literal["auto", "low", "high"] = "auto",
) -> Union[ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]:
    """
    Converts a list of Haystack Document objects into an OpenAI message dict
    object, ready for sending to an OpenAI LLM API endpoint.

    :param chunked_content_docs: List of HaystackDocument objects.
    :type chunked_content_docs: List[HaystackDocument]
    :param role: Role to be used for the messages.
    :type role: str
    :param img_detail: Details to be used for image messages, defaults to "auto"
    :type img_detail: Literal["auto", "low", "high"], optional
    :return: A list of messages dictionaries
    :rtype: List[List[dict]]
    """
    msg_content_list = list()
    for content_doc in content_docs:
        doc_type = get_processed_di_doc_type(content_doc)
        if doc_type is ProcessedDocIntelElementDocumentType.IMAGE:
            # Image content
            if content_doc.content:
                msg_content_list.append(
                    ChatCompletionContentPartTextParam(
                        type="text", text=content_doc.content.strip()
                    )
                )
            msg_content_list.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=base64_img_str_to_url(
                            base64_img_str=content_doc.blob.data.decode(),
                            mime_type=content_doc.blob.mime_type,
                        ),
                        detail=img_detail,
                    ),
                )
            )
        elif doc_type in [
            ProcessedDocIntelElementDocumentType.TEXT,
            ProcessedDocIntelElementDocumentType.TABLE,
        ]:
            # All other types with text-only output (the content field should be
            # populated with the text content to be exported). Ignore it if
            # the content field is None or ""
            if content_doc.content:
                msg_content_list.append(
                    ChatCompletionContentPartTextParam(
                        type="text", text=content_doc.content.strip()
                    )
                )
        else:
            raise ValueError("Unknown processed DI document type.")
    # Combine into a single message and transform into an OpenAI message to ensure the format is correct
    message_dict = {"role": role, "content": msg_content_list}
    role_to_output_type_mapper = {
        "user": ChatCompletionUserMessageParam,
        "assistant": ChatCompletionAssistantMessageParam,
    }
    return role_to_output_type_mapper[role](message_dict)
