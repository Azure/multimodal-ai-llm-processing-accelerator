from typing import Any, Callable, List, Optional, overload

from azure.ai.documentintelligence import models as _models
from azure.ai.documentintelligence._model_base import rest_field
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    BoundingRegion,
    DocumentLine,
    DocumentWord,
)

from ..helpers.image import scale_flat_poly_list
from .common import is_exact_match


class EnrichedDocumentLine(DocumentLine):
    """A content line object consisting of an adjacent sequence of content elements,
    such as words and selection marks.

    All required parameters must be populated in order to send to server.

    :ivar content: Concatenated content of the contained elements in reading order. Required.
    :vartype content: str
    :ivar polygon: Bounding polygon of the line, with coordinates specified relative to the
     top-left of the page. The numbers represent the x, y values of the polygon
     vertices, clockwise from the left (-180 degrees inclusive) relative to the
     element orientation.
    :vartype polygon: List[float]
    :ivar spans: Location of the line in the reading order concatenated content. Required.
    :vartype spans: List[~azure.ai.documentintelligence.models.DocumentSpan]
    :ivar confidence: Confidence of the content. This is generated by merging the
     confidence scores of all contained words. Required.
    :vartype confidence: float
    :ivar page_number: 1-based index of the page where the line is present. Required.
    :vartype page_number: int
    :ivar contained_words: All DocumentWords contained within the line. Required.
    :vartype contained_words: List[~azure.ai.documentintelligence.models.DocumentWord]
    """

    normalized_polygon: List[BoundingRegion] = rest_field()
    """Bounding polygon of the line, with coordinates specified relative to the
     top-left of the page. The numbers represent the x, y values of the polygon
     vertices, clockwise from the left (-180 degrees inclusive) relative to the
     element orientation, and normalized to values between 0 and 1."""
    confidence: float = rest_field()
    """Confidence value between 0 and 1. Default value is 1.0."""
    page_number: int = rest_field()
    """1-based index of the page where the line is present."""
    contained_words: List[DocumentWord] = rest_field()
    """1-based index of the page where the line is present."""

    @overload
    def __init__(
        self,
        *,
        content: str,
        spans: List["_models.DocumentSpan"],
        polygon: Optional[List[float]] = None,
        normalized_polygon: Optional[List[float]] = None,
        confidence: float,
        page_number: int,
        contained_words: List[DocumentWord],
    ): ...

    def __init__(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(*args, **kwargs)


class EnrichedDocumentWord(DocumentWord):
    """A content line object consisting of an adjacent sequence of content elements,
    such as words and selection marks.

    All required parameters must be populated in order to send to server.

    :ivar content: Text content of the word. Required.
    :vartype content: str
    :ivar polygon: Bounding polygon of the word, with coordinates specified relative to the
     top-left of the page. The numbers represent the x, y values of the polygon
     vertices, clockwise from the left (-180 degrees inclusive) relative to the
     element orientation.
    :vartype polygon: List[float]
    :ivar span: Location of the word in the reading order concatenated content. Required.
    :vartype span: ~azure.ai.documentintelligence.models.DocumentSpan
    :ivar normalized_polygon: Bounding polygon of the word, with coordinates
     specified relative to the top-left of the page and normalized to values
     between 0 and 1. The numbers represent the x, y values of the polygon vertices,
     clockwise from the left (-180 degrees inclusive) relative to the element orientation.
    :vartype normalized_polygon: List[float]
    :ivar confidence: Confidence of correctly extracting the word. Required.
    :vartype confidence: float
    :ivar page_number: 1-based index of the page where the line is present. Required.
    :vartype page_number: int
    """

    normalized_polygon: List[BoundingRegion] = rest_field()
    """Bounding polygon of the word, with coordinates specified relative to the
     top-left of the page. The numbers represent the x, y values of the polygon
     vertices, clockwise from the left (-180 degrees inclusive) relative to the
     element orientation, and normalized to values between 0 and 1."""
    page_number: int = rest_field()
    """1-based index of the page where the line is present."""

    @overload
    def __init__(
        self,
        *,
        content: str,
        span: "_models.DocumentSpan",
        polygon: Optional[List[BoundingRegion]] = None,
        confidence: float,
        normalized_polygon: Optional[List[BoundingRegion]] = None,
        page_number: int,
    ): ...

    def __init__(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(*args, **kwargs)


def extract_di_words(di_result: AnalyzeResult) -> List[EnrichedDocumentWord]:
    """
    Extract all words from a Document Intelligence response, enriching them.

    :param raw_azure_doc_response:
        The raw Document Intelligence response for a single document.

    :return:
        A list of all words in the document.
    """
    di_words = list()
    for page in di_result.pages:
        for word in page.words:
            out_word = EnrichedDocumentWord(
                normalized_polygon=scale_flat_poly_list(
                    word.polygon,
                    existing_scale=(page.width, page.height),
                    new_scale=(1, 1),
                ),
                page_number=page.page_number,
                **word,
            )
            di_words.append(out_word)
    return di_words


def extract_enriched_di_lines(
    di_result: AnalyzeResult,
    word_confidence_merg_func: Callable = min,
) -> List[EnrichedDocumentLine]:
    """
    Extract all lines from a Document Intelligence response, enriching them with
    additional information (eg. confidence score, page_number, contained_words).

    :param raw_azure_doc_response:
        The raw Document Intelligence response for a single document.
    :param word_confidence_merg_func:
        The function to use to merge multiple word confidence scores. Default is
        `min`.

    :return:
        A list of all lines in the document.
    """
    di_lines = list()
    for page_number, page in enumerate(di_result.pages, start=1):
        for line in page.lines:
            # Calculate confidence score from the words in the line
            contained_words = list()
            for span in line.spans:
                span_offset_start = span.offset
                span_offset_end = span_offset_start + span.length
                words_contained = [
                    word
                    for word in page.words
                    if word.span.offset >= span_offset_start
                    and word.span.offset + word.span.length <= span_offset_end
                ]
                contained_words.extend(words_contained)
            contained_words_conf_scores = [word.confidence for word in words_contained]
            out_line = EnrichedDocumentLine(
                confidence=word_confidence_merg_func(contained_words_conf_scores),
                normalized_polygon=scale_flat_poly_list(
                    line.polygon,
                    existing_scale=(page.width, page.height),
                    new_scale=(1, 1),
                ),
                contained_words=words_contained,
                page_number=page_number,
                **line,
            )
            di_lines.append(out_line)
    return di_lines


def find_matching_di_lines(
    value: str,
    raw_azure_doc_response: dict,
    match_func: Callable = is_exact_match,
    word_confidence_merg_func: Callable = min,
) -> List[EnrichedDocumentLine]:
    """
    Find all lines in a Document Intelligence response that match a value.

    :param value:
        The value to match.
    :param raw_azure_doc_response:
        The raw Document Intelligence response for a single document.
    :param match_func:
        The function to use to match the value. Default is `is_exact_match`.
    :param word_confidence_merg_func:
        The function to use when merging the confidence scores of each sub-word
        into the confidence for the line. Default is `min`.

    :return:
        A list of all matching lines.
    """
    if not value:
        return list()
    # Get all pieces of content
    di_lines = extract_enriched_di_lines(
        raw_azure_doc_response, word_confidence_merg_func
    )
    # Find all content that matches
    matching_lines = [line for line in di_lines if match_func(value, line.content)]
    return matching_lines


def find_matching_di_words(
    value: str,
    raw_azure_doc_response: dict,
    match_func: Callable = is_exact_match,
) -> List[EnrichedDocumentWord]:
    """
    Find all lines in a Document Intelligence response that match a value.

    :param value:
        The value to match.
    :param raw_azure_doc_response:
        The raw Document Intelligence response for a single document.

    :return:
        A list of all matching words.
    """
    if not value:
        return list()
    # Get all pieces of content
    di_words = extract_di_words(raw_azure_doc_response)
    # Find all content that matches
    matching_words = [word for word in di_words if match_func(value, word.content)]
    return matching_words
