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
import PIL
import PIL.Image as Image
import requests
from azure.ai.documentintelligence._model_base import \
    Model as DocumentModelBase
from azure.ai.documentintelligence._model_base import rest_field
from azure.ai.documentintelligence.models import (AnalyzeResult, Document,
                                                  DocumentBarcode,
                                                  DocumentFigure,
                                                  DocumentFootnote,
                                                  DocumentFormula,
                                                  DocumentKeyValueElement,
                                                  DocumentLanguage,
                                                  DocumentLine, DocumentList,
                                                  DocumentPage,
                                                  DocumentParagraph,
                                                  DocumentSection,
                                                  DocumentSelectionMark,
                                                  DocumentSpan, DocumentTable,
                                                  DocumentWord, ParagraphRole)
from fitz import Page as PyMuPDFPage
from haystack.dataclasses import ByteStream as HaystackByteStream
from haystack.dataclasses import Document as HaystackDocument
from pydantic import BaseModel, Field
from src.components.pymupdf import load_fitz_pdf, pymupdf_pdf_page_to_img_pil
from src.helpers.image import base64_to_pil_img, pil_img_to_base64

logger = logging.getLogger(__name__)

VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/heif",
}


def is_span_in_span(span: DocumentSpan, parent_span: DocumentSpan) -> bool:
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


def get_all_formulas(result: AnalyzeResult) -> List[DocumentFormula]:
    """
    Returns all formulas from the Document Intelligence result.

    :param result: AnalyzeResult object returned by the `begin_analyze_document`
        method.
    :type result: AnalyzeResult
    :return: A list of all formulas in the result.
    :rtype: List[DocumentFormula]
    """
    return list(
        itertools.chain.from_iterable(
            page.formulas for page in result.pages if page.formulas
        )
    )


def get_formula(
    all_formulas: List[DocumentFormula], span: DocumentSpan
) -> DocumentFormula:
    """
    Get the formula that matches the given span.

    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param span: The span to match.
    :type span: DocumentSpan
    :raises ValueError: If no formula is found for the given span.
    :raises NotImplementedError: If multiple formulas are found for the given
        span.
    :return: The formula that matches the given span.
    :rtype: DocumentFormula
    """
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


def get_formulas_in_spans(
    all_formulas: List[DocumentFormula], spans: List[DocumentSpan]
) -> List[DocumentFormula]:
    """
    Get the formula that matches the given span.

    TODO: update docs

    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param span: The span to match.
    :type span: DocumentSpan
    :raises ValueError: If no formula is found for the given span.
    :raises NotImplementedError: If multiple formulas are found for the given
        span.
    :return: The formula that matches the given span.
    :rtype: DocumentFormula
    """
    matching_formulas = list()
    for span in spans:
        matching_formulas.extend(
            [formula for formula in all_formulas if is_span_in_span(formula.span, span)]
        )
    return matching_formulas


def crop_img(img: PIL.Image.Image, crop_poly: list[float]) -> PIL.Image.Image:
    """
    Crops an image based on the coordinates of an [x0, y0, x1, y1, ...] polygon.

    :param img: Image to crop.
    :type img: PIL.Image.Image
    :param crop_poly: List of coordinates of the polygon (x0, y0, x1, y1, etc.)
    :type crop_poly: list[float]
    :return: The cropped image.
    :rtype: PIL.Image.Image
    """
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


def get_document_element_spans(
    document_element: DocumentModelBase,
) -> list[DocumentSpan]:
    """
    Get the spans of a document element.

    :param document_element: The document element to get the spans of.
    :type document_element: DocumentModelBase
    :raises NotImplementedError: Raised when the document element does not
        contain a span field.
    :return: The spans of the document element.
    :rtype: list[DocumentSpan]
    """
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


def document_span_to_span_bounds(span: DocumentSpan) -> SpanBounds:
    """Converts a DocumentSpan object to a SpanBounds object."""
    return SpanBounds(span.offset, span.offset + span.length)


def get_span_bounds_to_ordered_idx_mapper(
    result: "AnalyzeResult",
) -> Dict[SpanBounds, int]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type result: AnalyzeResult
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
        elements = getattr(result, attr) or list()
        all_spans.extend([get_min_and_max_span_bounds(elem.spans) for elem in elements])
    for page in result.pages:
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

    :param di_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    """

    def __init__(self, di_result: AnalyzeResult):
        self.page_span_bounds: Dict[int, SpanBounds] = self._get_page_span_bounds(
            di_result
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

    def _get_page_span_bounds(self, di_result: AnalyzeResult) -> Dict[int, SpanBounds]:
        """
        Gets the span bounds for each page.

        :param di_result: AnalyzeResult object.
        :type di_result: AnalyzeResult
        :raises ValueError: Raised when a gap exists between the span bounds of
            two pages.
        :return: Dictionary with page number as key and tuple of start and end.
        :rtype: Dict[int, SpanBounds]
        """
        page_span_bounds: Dict[int, SpanBounds] = dict()
        page_start_span = 0  # Set first page start to 0
        for page in di_result.pages:
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


def get_min_and_max_span_bounds(spans: List[DocumentSpan]) -> SpanBounds:
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

# For Example usage: section_heirarchy_mapper = get_section_heirarchy_mapper(di_result.sections)


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
        DocumentKeyValueElement,
        Document,
        DocumentSelectionMark,
    ]
    full_span_bounds: SpanBounds
    spans: List[DocumentSpan]
    start_page_number: int
    section_heirarchy_incremental_id: tuple[int]


def get_element_page_number(
    element: Union[
        DocumentPage,
        DocumentSection,
        DocumentParagraph,
        DocumentTable,
        DocumentFigure,
        DocumentList,
        DocumentKeyValueElement,
        Document,
    ]
) -> int:
    """
    Get the page number of a document element.

    :param element: The document element to get the page number of.
    :type element: Union[DocumentPage, DocumentSection, DocumentParagraph, DocumentTable, DocumentFigure, DocumentList, DocumentKeyValueElement, Document]
    :return: The page number of the document element.
    :rtype: int
    """
    if isinstance(element, DocumentPage):
        return element.page_number
    return element.bounding_regions[0].page_number


def get_element_span_info_list(
    result: "AnalyzeResult",
    page_span_calculator: PageSpanCalculator,
    section_to_incremental_id_mapper: Dict[int, tuple[int]],
) -> Dict[DocumentSpan, ElementInfo]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
    :type result: AnalyzeResult
    :param page_span_calculator: The PageSpanCalculator object to determine the
        page location of a span.
    :type page_span_calculator: PageSpanCalculator
    :param section_to_incremental_id_mapper: A dict containing a mapping of span
        to section heirarchical ID for each object type.
    :type section_to_incremental_id_mapper: Dict[int, tuple[int]]
    :returns: A dictionary with the element ID as key and the order as value.
    """
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
            DocumentSection,
            DocumentParagraph,
            DocumentTable,
            DocumentFigure,
            DocumentList,
            DocumentKeyValueElement,
            Document,
        ] = (
            getattr(result, attr) or list()
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
    for page in result.pages:
        for attr in ["barcodes", "lines", "words", "selection_marks"]:
            elements: List[
                DocumentBarcode, DocumentLine, DocumentWord, DocumentSelectionMark
            ] = (getattr(page, attr) or list())
            for element in elements:
                element_idx = page_sub_element_counter[attr]
                page_sub_element_counter[attr] += 1
                spans = get_document_element_spans(element)
                full_span_bounds = get_min_and_max_span_bounds(spans)
                element_span_info_list.append(
                    ElementInfo(
                        element_id=f"/{attr}/{element_idx}",
                        element=element,
                        full_span_bounds=full_span_bounds,
                        spans=spans,
                        start_page_number=page.page_number,
                        section_heirarchy_incremental_id=None,  # page elements are not referred to by sections
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
        DocumentSection: 1,
        DocumentTable: 2,
        DocumentFigure: 2,
        DocumentFormula: 2,
        DocumentBarcode: 2,
        Document: 2,
        DocumentKeyValueElement: 2,
        DocumentParagraph: 3,
        DocumentLine: 4,
        DocumentSelectionMark: 4,
        DocumentWord: 5,
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


class DocumentElementExporterBase(ABC):

    expected_element = None

    def _validate_element_type(self, element_info: ElementInfo):
        if not self.expected_element:
            raise ValueError("expected_element has not been set for this exporter.")

        if not isinstance(element_info.element, self.expected_element):
            raise ValueError(
                f"Element is incorrect type - {type(element_info.element)}. It should be of `{type(self.expected_element)}` type"
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


class DocumentPageExporterBase(DocumentElementExporterBase):

    expected_element = DocumentPage

    @abstractmethod
    def convert_page_start(
        self, element_info: ElementInfo, page_img: Image.Image
    ) -> List[HaystackDocument]:
        """
        Method for exporting element for the beginning of the page.
        """
        pass

    @abstractmethod
    def convert_page_end(
        self, element_info: ElementInfo, page_img: Image.Image
    ) -> List[HaystackDocument]:
        """
        Method for exporting element for the beginning of the page.
        """
        pass


class DefaultDocumentPageExporter(DocumentPageExporterBase):

    expected_format_placeholders = ["{page_number}"]

    def __init__(
        self,
        page_start_text_formats: Optional[List[str]] = [
            "Page: {page_number} content\n"
        ],
        page_end_text_formats: Optional[str] = ["Footer: {page_number}"],
        page_img_order: Optional[Literal["before", "after"]] = "after",
        page_img_text_intro: Optional[str] = "\nPage {page_number} Image:",
        img_dpi: int = 100,
    ):
        self.page_start_text_formats = page_start_text_formats
        self.page_end_text_formats = page_end_text_formats
        self.page_img_order = page_img_order
        self.page_img_text_intro = page_img_text_intro
        self.img_dpi = img_dpi

    def convert_page_start(
        self, element_info: ElementInfo, page_img: Image.Image
    ) -> List[HaystackDocument]:
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "page_number": element_info.start_page_number,
            "page_location": "start",
        }
        if self.page_img_order == "before":
            outputs.extend(self._export_page_img_docs(element_info, page_img, meta))
        if self.page_start_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_start_text_formats, meta
                )
            )
        return outputs

    def convert_page_end(
        self, element_info: ElementInfo, page_img: Image.Image
    ) -> List[Document]:
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "page_number": element_info.start_page_number,
            "page_location": "end",
        }
        if self.page_end_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_end_text_formats, meta
                )
            )
        if self.page_img_order == "after":
            outputs.extend(self._export_page_img_docs(element_info, page_img, meta))
        return outputs

    def export_page_img(self, pdf_page: PyMuPDFPage) -> Image.Image:
        return pymupdf_pdf_page_to_img_pil(
            pdf_page, img_dpi=self.img_dpi, rotation=False
        )

    def _export_page_img_docs(
        self, element_info: ElementInfo, page_img: Image.Image, meta: Dict[str, Any]
    ) -> List[HaystackDocument]:
        img_bytestream = HaystackByteStream(
            data=pil_img_to_base64(page_img),
            mime_type="image/jpeg",
        )
        img_outputs = list()
        if self.page_img_text_intro:
            img_outputs.append(
                HaystackDocument(
                    id=f"{element_info.element_id}_page_img_intro_text",
                    content=self.page_img_text_intro.format(
                        page_number=element_info.start_page_number
                    ),
                    meta=meta,
                )
            )
        img_outputs.append(
            HaystackDocument(
                id=f"{element_info.element_id}_img",
                blob=img_bytestream,
                meta=meta,
            )
        )
        return img_outputs

    def _export_page_text_docs(
        self, element_info: ElementInfo, text_formats: List[str], meta: Dict[str, Any]
    ) -> List[HaystackDocument]:
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


class DocumentParagraphExporterBase(DocumentElementExporterBase):

    expected_element = DocumentParagraph

    @abstractmethod
    def convert_paragraph(self, element_info: ElementInfo) -> List[HaystackDocument]:
        """
        Method for exporting element for the beginning of the page.
        """
        pass


class DefaultDocumentParagraphExporter(DocumentParagraphExporterBase):
    def __init__(
        self,
        general_text_format: Optional[str] = "{content}",
        page_header_format: Optional[str] = "Page Title: {content}",
        page_footer_format: Optional[str] = "\nPage Footer: {content}",
        title_format: Optional[str] = "\nTitle: {content}\n",
        section_heading_format: Optional[str] = "\nSection heading: {content}\n",
        footnote_format: Optional[str] = "Footnote: {content}",
        page_number_format: Optional[str] = None,
    ):
        self.paragraph_format_mapper = {
            None: general_text_format,
            ParagraphRole.PAGE_HEADER: page_header_format,
            ParagraphRole.PAGE_FOOTER: page_footer_format,
            ParagraphRole.TITLE: title_format,
            ParagraphRole.SECTION_HEADING: section_heading_format,
            ParagraphRole.FOOTNOTE: footnote_format,
            ParagraphRole.PAGE_NUMBER: page_number_format,
        }

    def convert_paragraph(self, element_info: ElementInfo) -> List[HaystackDocument]:
        self._validate_element_type(element_info)
        format_mapper = self.paragraph_format_mapper[element_info.element.role]
        if format_mapper:
            return [
                HaystackDocument(
                    id=f"{element_info.element_id}",
                    content=format_mapper.format(content=element_info.element.content),
                    meta={
                        "element_id": element_info.element_id,
                        "page_number": element_info.start_page_number,
                    },
                )
            ]
        return list()


class DocumentLineExporterBase(DocumentElementExporterBase):

    expected_element = DocumentLine

    @abstractmethod
    def convert_line(self, element_info: ElementInfo) -> List[HaystackDocument]:
        """
        Method for exporting element for the beginning of the page.
        """
        pass


class DefaultDocumentLineExporter(DocumentLineExporterBase):

    def __init__(
        self,
        text_format: Optional[str] = "{content}",
    ):
        self.text_format = text_format

    def convert_line(self, element_info: ElementInfo) -> List[HaystackDocument]:
        self._validate_element_type(element_info)
        if self.text_format:
            formatted_text = self.text_format.format(
                content=element_info.element.content
            )
            formatted_if_null = self.text_format.format(content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=self.text_format.format(
                            content=element_info.element.content
                        ),
                        meta={
                            "element_id": element_info.element_id,
                            "page_number": element_info.start_page_number,
                        },
                    )
                ]
        return list()


class DocumentWordExporterBase(DocumentElementExporterBase):

    expected_element = DocumentWord

    @abstractmethod
    def convert_word(self, element_info: ElementInfo) -> List[HaystackDocument]:
        """
        Method for exporting element for the beginning of the page.
        """
        pass


class DefaultDocumentWordExporter(DocumentWordExporterBase):

    def __init__(
        self,
        text_format: Optional[str] = "{content}",
    ):
        self.text_format = text_format

    def convert_word(self, element_info: ElementInfo) -> List[HaystackDocument]:
        self._validate_element_type(element_info)
        if self.text_format:
            formatted_text = self.text_format.format(
                content=element_info.element.content
            )
            formatted_if_null = self.text_format.format(content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=self.text_format.format(
                            content=element_info.element.content
                        ),
                        meta={
                            "element_id": element_info.element_id,
                            "page_number": element_info.start_page_number,
                        },
                    )
                ]
        return list()


class DocumentTableExporterBase(DocumentElementExporterBase):

    expected_element = DocumentTable

    @abstractmethod
    def convert_table(
        self, element_info: ElementInfo, all_formulas: List[DocumentFormula]
    ) -> List[HaystackDocument]:
        """
        Method for exporting table elements.
        """
        pass


class DefaultDocumentTableExporter(DocumentTableExporterBase):

    expected_format_placeholders = ["{table_number}", "{caption}", "{footnotes}"]

    def __init__(
        self,
        before_table_text_formats: Optional[List[str]] = ["Table Caption: {caption}"],
        after_table_text_formats: Optional[List[str]] = ["Footnotes: {footnotes}"],
    ):
        self.before_table_text_formats = before_table_text_formats
        self.after_table_text_formats = after_table_text_formats

    def convert_table(
        self, element_info: ElementInfo, all_formulas: List[DocumentFormula]
    ) -> List[HaystackDocument]:
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "page_number": element_info.start_page_number,
        }
        if self.before_table_text_formats:
            outputs.extend(
                self._export_table_text(
                    element_info,
                    f"{element_info.element_id}_before_table_text",
                    self.before_table_text_formats,
                    meta,
                )
            )
        # Convert table content
        table_md, table_df = self._convert_table_content(
            element_info.element, all_formulas
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
                    f"{element_info.element_id}_after_table_text",
                    self.after_table_text_formats,
                    meta,
                )
            )

        return outputs

    def _export_table_text(
        self,
        element_info: ElementInfo,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        table_number_text = get_element_number(element_info)
        caption_text = (
            element_info.element.caption.content if element_info.element.caption else ""
        )
        if element_info.element.footnotes:
            footnotes_text = "\n".join(
                [footnote.content for footnote in element_info.element.footnotes]
            )
        else:
            footnotes_text = ""
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
        self, table: DocumentTable, all_formulas: List[DocumentFormula]
    ) -> tuple[str, pd.DataFrame]:
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
        return table_md, table_df


class DocumentFigureExporterBase(DocumentElementExporterBase):

    expected_element = DocumentFigure

    @abstractmethod
    def convert_figure(
        self, element_info: ElementInfo, page_img: Image.Image
    ) -> List[HaystackDocument]:
        """
        Method for exporting figure elements.
        """
        pass


def get_element_number(element_info: ElementInfo) -> str:
    return str(int(element_info.element_id.split("/")[-1]) + 1)


class DefaultDocumentFigureExporter(DocumentFigureExporterBase):

    expected_format_placeholders = [
        "{figure_number}",
        "{caption}",
        "{footnotes}",
        "{content}",
    ]

    def __init__(
        self,
        before_figure_text_formats: Optional[List[str]] = [
            "Figure Caption: {caption}",
            "Figure Content:\n{content}",
        ],
        output_figure_img: bool = True,
        after_figure_text_formats: Optional[List[str]] = ["Footnotes: {footnotes}"],
    ):
        self.before_figure_text_formats = before_figure_text_formats
        self.output_figure_img = output_figure_img
        self.after_figure_text_formats = after_figure_text_formats

    def convert_figure(
        self, element_info: ElementInfo, page_img: Image.Image, di_result: AnalyzeResult
    ) -> List[HaystackDocument]:
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
            "page_number": element_info.start_page_number,
        }
        if self.before_figure_text_formats:
            outputs.extend(
                self._export_figure_text(
                    element_info,
                    di_result,
                    f"{element_info.element_id}_before_figure_text",
                    self.before_figure_text_formats,
                    meta,
                )
            )
        if self.output_figure_img:
            page_element = di_result.pages[page_numbers[0]]
            figure_img = self.convert_figure_to_img(
                element_info.element, page_img, page_element
            )
            outputs.append(
                HaystackDocument(
                    id=f"{element_info.element_id}_img",
                    content=f"\nFigure {get_element_number(element_info)} Image:",
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
                    di_result,
                    f"{element_info.element_id}_after_figure_text",
                    self.after_figure_text_formats,
                    meta,
                )
            )

        return outputs

    def _export_figure_text(
        self,
        element_info: ElementInfo,
        di_result: AnalyzeResult,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        figure_number_text = get_element_number(element_info)
        caption_text = (
            element_info.element.caption if element_info.element.caption else ""
        )
        if element_info.element.footnotes:
            footnotes_text = "\n".join(
                [footnote.content for footnote in element_info.element.footnotes]
            )
        else:
            footnotes_text = ""
        # Get text content only if it is necessary (it is a slow operation)
        content_text = (
            self._get_figure_text_content(element_info.element, di_result)
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
        self, figure_element: DocumentFigure, di_result: AnalyzeResult
    ) -> str:
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
            # print(content_span)
            span_perfect_match = False
            # print("PARAGRAPHS")
            for para in di_result.paragraphs or []:
                if para.spans[0].offset > content_min_max_span_bounds.end:
                    break
                # print(para)
                span_perfect_match = para.spans[0] == content_span
                span_perfect_match = False
                if span_perfect_match or all(
                    is_span_in_span(para_span, content_span) for para_span in para.spans
                ):
                    # print("para match")
                    matched_spans.extend(para.spans)
                    if current_line_strings:
                        output_line_strings.append(" ".join(current_line_strings))
                        current_line_strings = list()
                    output_line_strings.append(para.content)
            if span_perfect_match:
                continue
            for page_number in figure_page_numbers:
                # print("LINES - page {}".format(page_number))
                for line in di_result.pages[page_number - 1].lines or []:
                    # print(line)
                    # If we have already matched the span, we can skip the rest of the lines
                    if line.spans[0].offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    if any(
                        is_span_in_span(line_span, matched_span)
                        for matched_span in matched_spans
                        for line_span in line.spans
                    ):
                        # print("Line already contained")
                        continue
                    span_perfect_match = line.spans[0] == content_span
                    if span_perfect_match or all(
                        is_span_in_span(line_span, content_span)
                        for line_span in line.spans
                    ):
                        # print("line match")
                        matched_spans.extend(line.spans)
                        if current_line_strings:
                            output_line_strings.append(" ".join(current_line_strings))
                            current_line_strings = list()
                        output_line_strings.append(line.content)
                if span_perfect_match:
                    continue
                # print("WORDS - page {}".format(page_number))
                for word in di_result.pages[page_number - 1].words or []:
                    # print(word)
                    # If we have already matched the span, we can skip the rest of the lines
                    if word.span.offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    if any(
                        is_span_in_span(word.span, matched_span)
                        for matched_span in matched_spans
                    ):
                        # print("Word already contained")
                        continue
                    span_perfect_match = word.span == content_span
                    if span_perfect_match or is_span_in_span(word.span, content_span):
                        # print("word match")
                        current_line_strings.append(word.content)
            if current_line_strings:
                output_line_strings.append(" ".join(current_line_strings))
                current_line_strings = list()
        if output_line_strings:
            # print("Figure text content for img:", "\n".join(output_line_strings))
            return "\n".join(output_line_strings)
        # print("No text content for img")
        return ""

    def convert_figure_to_img(
        self, figure: DocumentFigure, page_img: Image.Image, page_element: DocumentPage
    ) -> Image.Image:
        di_page_dimensions = (page_element.width, page_element.height)
        return crop_img(
            page_img,
            normalize_xy_coords(
                figure.bounding_regions[0].polygon,
                di_page_dimensions,
                page_img.size,
            ),
        )
