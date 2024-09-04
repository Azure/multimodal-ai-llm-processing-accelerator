import copy
import hashlib
import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Union

import networkx as nx
import pandas as pd
import PIL.Image as Image
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentFormula,
    DocumentParagraph,
    DocumentSpan,
    DocumentTable,
    ParagraphRole,
)
from haystack.dataclasses import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/heif",
}


# class TranscriptionContent(BaseModel):
#     bounding_regions: list = Field(
#         default_factory=list, description="Bounding boxes extracted from the document"
#     )
#     text: str = Field(..., description="Text transcription of the document")
#     table: Optional[pd.DataFrame] = Field(
#         None, description="Tables extracted from the document"
#     )
#     image: Optional[Image.Image] = Field(
#         None, description="Images extracted from the document"
#     )
#     handwriting: Optional[str] = Field(
#         None, description="Handwriting extracted from the document"
#     )
#     styles: list = Field(default_factory=list, description="Styles extracted")
#     meta: dict = Field(
#         default_factory=dict, description="Metadata extracted from the document"
#     )

#     class Config:
#         arbitrary_types_allowed = True


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
        page_layout: Literal["natural", "single_column"] = "natural",
        threshold_y: Optional[float] = 0.05,
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
        :param page_layout: The type reading order to follow. If "natural" is chosen then the natural reading order
            determined by Azure will be used. If "single_column" is chosen then all lines with the same height on the
            page will be grouped together based on a threshold determined by `threshold_y`.
        :param threshold_y: The threshold to determine if two recognized elements in a PDF should be grouped into a
            single line. This is especially relevant for section headers or numbers which may be spacially separated
            on the horizontal axis from the remaining text. The threshold is specified in units of inches.
            This is only relevant if "single_column" is chosen for `page_layout`.
        """
        self.preceding_context_len = preceding_context_len
        self.following_context_len = following_context_len
        self.merge_multiple_column_headers = merge_multiple_column_headers
        self.page_layout = page_layout
        self.threshold_y = threshold_y
        if self.page_layout == "single_column" and self.threshold_y is None:
            self.threshold_y = 0.05

    def process_analyze_result(self, result: AnalyzeResult):
        """
        Processes the result of an analyze operation and returns the extracted text and tables.

        :param result: The result of an analyze operation.
        :returns: A dictionary containing the extracted text and tables.
        """
        # Convert the result to a dictionary
        result_dict = result.as_dict()

        # Extract the text and tables
        paragraphs = self._convert_paragraphs(result_dict)
        tables = self._convert_tables(result_dict)
        transcribed_content = self._convert_tables_and_text(result=result)

        return transcribed_content
    
    def _convert_figures(self, result: "AnalyzeResult") -> List[Document]:

    def _convert_tables(self, result: "AnalyzeResult") -> List[EnhancedDocumentTable]:
        """
        Converts the tables extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :returns: List of Documents containing the tables extracted from the AnalyzeResult object.
        """
        all_formulas = get_all_formulas(result)
        tables = []
        for table in result.tables:
            enhanced_table = self._convert_table(table=table, all_formulas=all_formulas)
            tables.append(enhanced_table)
        return tables

    def _convert_table(
        self, table: DocumentTable, all_formulas: List[DocumentFormula]
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
            di_document_table=table,
        )

    def _convert_paragraphs(
        self, result: "AnalyzeResult"
    ) -> List[EnhancedDocumentParagraph]:
        """
        This converts the `AnalyzeResult` object ... TODO.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :returns: TODO
        """
        table_spans_by_page = self._collect_table_spans(result=result)

        if result.paragraphs:
            paragraphs_to_pages: Dict[int, list] = defaultdict(list)
            for paragraph in result.paragraphs:
                if paragraph.bounding_regions:
                    # If paragraph spans multiple pages we group it with the first page number
                    page_numbers = [b.page_number for b in paragraph.bounding_regions]
                else:
                    # If page_number is not available we put the paragraph onto an existing page
                    current_last_page_number = (
                        sorted(paragraphs_to_pages.keys())[-1]
                        if paragraphs_to_pages
                        else 1
                    )
                    page_numbers = [current_last_page_number]
                tables_on_page = table_spans_by_page[page_numbers[0]]
                # Check if paragraph is part of a table and if so skip
                if self._check_if_in_table(tables_on_page, line_or_paragraph=paragraph):
                    continue
                # Contruct the EnhancedDocumentParagraph model without an IDX or ID
                enhanced_paragraph = EnhancedDocumentParagraph.model_construct(
                    text=paragraph.content,
                    role=paragraph.role,
                    page_number=page_numbers[0],
                    span=paragraph.spans[0],
                    di_document_paragraph=paragraph,
                )
                paragraphs_to_pages[page_numbers[0]].append(enhanced_paragraph)

            # max_page_number: int = max(paragraphs_to_pages)
            # for page_idx in range(1, max_page_number + 1):
            #     # We add empty strings for missing pages so the preprocessor can still extract the correct page number
            #     # from the original PDF.
            #     page_text = paragraphs_to_pages.get(page_idx, "")
            #     paragraphs.append(page_text)
        else:
            logger.warning("No text paragraphs were detected by the OCR conversion.")
        return list(itertools.chain.from_iterable(paragraphs_to_pages.values()))

    def _collect_table_spans(self, result: "AnalyzeResult") -> Dict:
        """
        Collect the spans of all tables by page number.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
        :returns: A dictionary with the page number as key and a list of table spans as value.
        """
        table_spans_by_page = defaultdict(list)
        tables = result.tables if result.tables else []
        for table in tables:
            if not table.bounding_regions:
                continue
            table_spans_by_page[table.bounding_regions[0].page_number].append(
                table.spans[0]
            )
        return table_spans_by_page

    def _check_if_in_table(
        self,
        tables_on_page: dict,
        line_or_paragraph: Union["DocumentLine", "DocumentParagraph"],
    ) -> bool:
        """
        Check if a line or paragraph is part of a table.

        :param tables_on_page: A dictionary with the page number as key and a list of table spans as value.
        :param line_or_paragraph: The line or paragraph to check.
        :returns: True if the line or paragraph is part of a table, False otherwise.
        """
        in_table = False
        # Check if line is part of a table
        for table in tables_on_page:
            if (
                table.offset
                <= line_or_paragraph.spans[0].offset
                <= table.offset + table.length
            ):
                in_table = True
                break
        return in_table

    # def _convert_to_single_column_text(
    #     self, result: "AnalyzeResult", threshold_y: float = 0.05
    # ) -> Document:
    #     """
    #     This converts the `AnalyzeResult` object ... TODO.

    #     :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
    #         can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
    #     :param threshold_y: height threshold in inches for PDF and pixels for images
    #     :returns: TODO
    #     """
    #     table_spans_by_page = self._collect_table_spans(result=result)

    #     # Find all pairs of lines that should be grouped together based on the y-value of the upper left coordinate
    #     # of their bounding box
    #     pairs_by_page = defaultdict(list)
    #     for page_idx, page in enumerate(result.pages):
    #         lines = page.lines if page.lines else []
    #         # Only works if polygons is available
    #         if all(line.polygon is not None for line in lines):
    #             for i in range(len(lines)):  # pylint: disable=consider-using-enumerate
    #                 # left_upi, right_upi, right_lowi, left_lowi = lines[i].polygon
    #                 left_upy_i = lines[i].polygon[1]  # type: ignore
    #                 pairs_by_page[page_idx].append([i, i])
    #                 for j in range(i + 1, len(lines)):  # pylint: disable=invalid-name
    #                     left_upy_j = lines[j].polygon[1]  # type: ignore
    #                     close_on_y_axis = abs(left_upy_i - left_upy_j) < threshold_y
    #                     if close_on_y_axis:
    #                         pairs_by_page[page_idx].append([i, j])
    #         # Default if polygon is not available
    #         else:
    #             logger.info(
    #                 "Polygon information for lines on page {page_idx} is not available so it is not possible "
    #                 "to enforce a single column page layout.".format(page_idx=page_idx)
    #             )
    #             for i in range(len(lines)):
    #                 pairs_by_page[page_idx].append([i, i])

    #     # merged the line pairs that are connected by page
    #     merged_pairs_by_page = {}
    #     for page_idx in pairs_by_page:
    #         graph = nx.Graph()
    #         graph.add_edges_from(pairs_by_page[page_idx])
    #         merged_pairs_by_page[page_idx] = [
    #             list(a) for a in list(nx.connected_components(graph))
    #         ]

    #     # Convert line indices to the DocumentLine objects
    #     merged_lines_by_page = {}
    #     for page_idx, page in enumerate(result.pages):
    #         rows = []
    #         lines = page.lines if page.lines else []
    #         # We use .get(page_idx, []) since the page could be empty
    #         for row_of_lines in merged_pairs_by_page.get(page_idx, []):
    #             lines_in_row = [lines[line_idx] for line_idx in row_of_lines]
    #             rows.append(lines_in_row)
    #         merged_lines_by_page[page_idx] = rows

    #     # Sort the merged pairs in each row by the x-value of the upper left bounding box coordinate
    #     x_sorted_lines_by_page = {}
    #     for page_idx, _ in enumerate(result.pages):
    #         sorted_rows = []
    #         for row_of_lines in merged_lines_by_page[page_idx]:
    #             sorted_rows.append(sorted(row_of_lines, key=lambda x: x.polygon[0]))  # type: ignore
    #         x_sorted_lines_by_page[page_idx] = sorted_rows

    #     # Sort each row within the page by the y-value of the upper left bounding box coordinate
    #     y_sorted_lines_by_page = {}
    #     for page_idx, _ in enumerate(result.pages):
    #         sorted_rows = sorted(x_sorted_lines_by_page[page_idx], key=lambda x: x[0].polygon[1])  # type: ignore
    #         y_sorted_lines_by_page[page_idx] = sorted_rows

    #     # Construct the text to write
    #     texts = []
    #     for page_idx, page in enumerate(result.pages):
    #         tables_on_page = table_spans_by_page[page.page_number]
    #         page_text = ""
    #         for row_of_lines in y_sorted_lines_by_page[page_idx]:
    #             # Check if line is part of a table and if so skip
    #             if any(
    #                 self._check_if_in_table(tables_on_page, line_or_paragraph=line)
    #                 for line in row_of_lines
    #             ):
    #                 continue
    #             page_text += " ".join(line.content for line in row_of_lines)
    #             page_text += "\n"
    #         texts.append(page_text)
    #     all_text = "\f".join(texts)
    #     document = Document(content=all_text)
    #     return (
    #         document,
    #         pairs_by_page,
    #         merged_pairs_by_page,
    #         merged_lines_by_page,
    #         x_sorted_lines_by_page,
    #         y_sorted_lines_by_page,
    #         texts,
    #         all_text,
    #     )

    # def _convert_tables_and_text(self, result: "AnalyzeResult") -> List[Document]:
    #     """
    #     Converts the tables and text extracted by Azure's Document Intelligence service into Haystack Documents.

    #     :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
    #         can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
    #     :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
    #         Can be any custom keys and values.
    #     :returns: List of Documents containing the tables and text extracted from the AnalyzeResult object.
    #     """
    #     tables = self._convert_tables(result=result)
    #     if self.page_layout == "natural":
    #         text = self._convert_to_natural_text(result=result)
    #     else:
    #         assert isinstance(self.threshold_y, float)
    #         text = self._convert_to_single_column_text(
    #             result=result, threshold_y=self.threshold_y
    #         )
    #     docs = [*tables, text]
    #     return docs

    # def _hash_dataframe(
    #     self, df: pd.DataFrame, desired_samples=5, hash_length=4
    # ) -> str:
    #     """
    #     Returns a hash of the DataFrame content.

    #     The hash is based on the content of the DataFrame.
    #     :param df: The DataFrame to hash.
    #     :param desired_samples: The desired number of samples to hash.
    #     :param hash_length: The length of the hash for each sample.

    #     :returns: A hash of the DataFrame content.
    #     """
    #     # take adaptive sample of rows to hash because we can have very large dataframes
    #     hasher = hashlib.md5()
    #     total_rows = len(df)
    #     # sample rate based on DataFrame size and desired number of samples
    #     sample_rate = max(1, total_rows // desired_samples)

    #     hashes = pd.util.hash_pandas_object(df, index=True)
    #     sampled_hashes = hashes[::sample_rate]

    #     for hash_value in sampled_hashes:
    #         partial_hash = str(hash_value)[:hash_length].encode("utf-8")
    #         hasher.update(partial_hash)

    #     return hasher.hexdigest()
