import pandas as pd
import pytest
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    BoundingRegion,
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
    DocumentTableCell,
    DocumentWord,
    ParagraphRole,
)
from haystack.dataclasses import Document as HaystackDocument
from src.components.doc_intelligence import (
    DefaultDocumentTableLoader,
    DocumentTableLoaderBase,
    PageFormatConfig,
    TableFormatConfig,
    _convert_section_heirarchy_to_incremental_numbering,
    convert_element_heirarchy_to_incremental_numbering,
    get_element_heirarchy_mapper,
    get_section_heirarchy,
    substitute_content_formulas,
)


@pytest.mark.parametrize(
    "section_direct_children_mapper,expected",
    [
        [
            {0: [0, 1, 3], 1: [1, 2], 2: [2], 3: [3]},
            {
                0: (0,),
                1: (0, 1),
                2: (0, 1, 2),
                3: (0, 3),
            },
        ],
        [
            {
                0: [0, 1, 3, 5, 7],
                1: [1, 2],
                2: [2],
                3: [3, 4],
                4: [4, 6],
                5: [5],
                6: [6],
                7: [7, 8],
                8: [8],
            },
            {
                0: (0,),
                1: (0, 1),
                2: (0, 1, 2),
                3: (0, 3),
                4: (0, 3, 4),
                5: (0, 5),
                6: (0, 3, 4, 6),
                7: (
                    0,
                    7,
                ),
                8: (0, 7, 8),
            },
        ],
    ],
)
def test_get_section_heirarchy(section_direct_children_mapper, expected):
    result = get_section_heirarchy(section_direct_children_mapper)
    print(result)
    print(expected)
    assert result == expected


@pytest.mark.parametrize(
    "analyze_result,expected",
    [
        [
            AnalyzeResult(
                sections=[
                    DocumentSection(
                        elements=["/paragraphs/1", "/sections/1"],
                    ),
                    DocumentSection(
                        elements=["/paragraphs/2"],
                    ),
                ],
                paragraphs=[
                    DocumentParagraph(
                        content="This paragraph occurs before the first section.",
                        spans=[DocumentSpan(offset=0, length=1)],
                    ),
                    DocumentParagraph(
                        content="This is contained in section 1.",
                        spans=[DocumentSpan(offset=0, length=1)],
                    ),
                    DocumentParagraph(
                        content="This is contained in section 2, which is contained by section 1.",
                        spans=[DocumentSpan(offset=0, length=1)],
                    ),
                    DocumentParagraph(
                        content="This paragraph appears outside of all sections.",
                        spans=[DocumentSpan(offset=0, length=1)],
                    ),
                ],
            ),
            {
                "paragraphs": {
                    # 0 & 3 do not appear in result since they do not belong to a section
                    1: (0,),
                    2: (0, 1),
                },
                "sections": {
                    0: (0,),
                    1: (0, 1),
                },
            },
        ],
    ],
)
def test_get_element_heirarchy_mapper(analyze_result, expected):
    result = get_element_heirarchy_mapper(analyze_result)
    print(result)
    print(expected)
    assert result == expected


@pytest.mark.parametrize(
    "section_heirarchy_mapper,expected",
    [
        [
            {
                0: (0,),  # Should be excluded from the output
                1: (0, 1),  # 1
                2: (0, 2),  # 2
                3: (0, 2, 3),  # 2.1
                4: (0, 2, 4),  # 2.2
                5: (0, 2, 4, 5),  # 2.2.1
                6: (0, 6),  # 3
                7: (0, 6, 7),  # 3.1
                8: (0, 8),  # 4
            },
            {
                1: (1,),
                2: (2,),
                3: (2, 1),
                4: (2, 2),
                5: (2, 2, 1),
                6: (3,),
                7: (3, 1),
                8: (4,),
            },
        ],
        [{}, {}],
    ],
)
def test_convert_section_heirarchy_to_incremental_numbering(
    section_heirarchy_mapper, expected
):
    result = _convert_section_heirarchy_to_incremental_numbering(
        section_heirarchy_mapper
    )
    print(result)
    print(expected)
    assert result == expected


@pytest.mark.parametrize(
    "section_heirarchy_mapper,expected",
    [
        [
            {
                "sections": {
                    1: (0, 1),  # 1
                    2: (0, 2),  # 2
                    3: (0, 2, 3),  # 2.1
                    4: (0, 2, 4),  # 2.2
                    5: (0, 2, 4, 5),  # 2.2.1
                    6: (0, 6),  # 3
                },
                "tables": {
                    0: (0, 1),
                    1: (0, 2, 3),
                    2: (0, 6),
                },
                "paragraphs": {
                    # 0: # paragraph occurs before first section, so is not present in section elements
                    1: (
                        0,
                    ),  # Occurs within first section - should result in no section heirarchy
                    2: (0, 1),
                    3: (0, 2),
                    4: (0, 2, 4, 5),
                    # 5: # paragraph occurs before first section, so is not present in section elements
                },
            },
            {
                "sections": {
                    1: (1,),  # 1
                    2: (2,),  # 2
                    3: (2, 1),  # 2.1
                    4: (2, 2),  # 2.2
                    5: (2, 2, 1),  # 2.2.1
                    6: (3,),  # 3
                },
                "tables": {
                    0: (1,),
                    1: (2, 1),
                    2: (3,),
                },
                "paragraphs": {
                    # 0: # paragraph occurs before first section
                    1: None,
                    2: (1,),
                    3: (2,),
                    4: (2, 2, 1),
                    # 5: # paragraph occurs before first section
                },
            },
        ],
        [{}, {}],
    ],
)
def test_convert_element_heirarchy_to_incremental_numbering(
    section_heirarchy_mapper, expected
):
    result = convert_element_heirarchy_to_incremental_numbering(
        section_heirarchy_mapper
    )
    print(result)
    print(expected)
    assert result == expected


import io

from haystack.dataclasses import ByteStream as HaystackBytestream
from PIL import Image
from src.helpers.image import base64_to_pil_img, pil_img_to_base64


### PageFormatConfigs ###
class TestPageFormatConfig:
    def test_format_page_element_start_text_only(self):
        page = DocumentPage(page_number=123)
        page_format_config = PageFormatConfig(
            include=True,
            text_format="Custom format - Page {page_number}",
            page_text_order="start",
            page_img_order="none",
        )
        element_id = "/pages/123"
        expected = [
            HaystackDocument(
                id=element_id + "_text",
                content="Custom format - Page 123",
                meta={"element_id": element_id, "page_text_order": "start"},
            )
        ]
        result = page_format_config.format_page_element(
            page, element_id, current_location="start"
        )
        print(result)
        print(expected)
        assert result == expected

    def test_format_page_element_end_text_and_image(self):
        page = DocumentPage(page_number=123)
        page_format_config = PageFormatConfig(
            include=True,
            text_format="Custom format - Page {page_number}",
            page_text_order="end",
            page_img_order="end",
        )
        page_img = Image.new("RGB", (100, 100))
        page_img_data = pil_img_to_base64(page_img)
        element_id = "/pages/123"
        expected = [
            HaystackDocument(
                id=element_id + "_text",
                content="Custom format - Page 123",
                meta={"element_id": element_id, "page_text_order": "end"},
            ),
            HaystackDocument(
                id=element_id + "_img",
                blob=HaystackBytestream(data=page_img_data, mime_type="image/jpeg"),
                meta={"element_id": element_id, "page_img_order": "end"},
            ),
        ]
        result = page_format_config.format_page_element(
            page, element_id, current_location="end", page_img=page_img
        )
        print(result)
        print(expected)
        assert result == expected


# @pytest.fixture(auto_use=True, scope="module")
# def default_document_loader() -> DefaultDocumentLoader:
#     return DefaultDocumentLoader(all_formulas=list())

# @pytest.fixture(auto_use=True, scope="module")
# def di_document_table_md(di_document_table: DocumentTable, default_document_loader: DefaultDocumentLoader) -> str:
#     return default_document_loader.load_di_document_table(di_document_table)[0]

# @pytest.fixture(auto_use=True, scope="module")
# def di_document_table_df(di_document_table: DocumentTable, default_document_loader: DefaultDocumentLoader) -> pd.DataFrame:
#     return default_document_loader.load_di_document_table(di_document_table)[1]

di_document_table = DocumentTable(
    # row_count=3,
    # column_count=2,
    # cells=[
    #     DocumentTableCell(
    #         row_index=0,
    #         column_index=0,
    #         content="I=6",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     0.9323,
    #                     2.0053,
    #                     1.4066,
    #                     2.0053,
    #                     1.4066,
    #                     2.2236,
    #                     0.9323,
    #                     2.2236,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352737, length=3)],
    #         elements=["/paragraphs/5481"],
    #     ),
    #     DocumentTableCell(
    #         row_index=0,
    #         column_index=1,
    #         content="Number of measurement series for each material, performed over six days.",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     1.4066,
    #                     2.0053,
    #                     5.4187,
    #                     2.0101,
    #                     5.4235,
    #                     2.2236,
    #                     1.4066,
    #                     2.2236,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352741, length=72)],
    #         elements=["/paragraphs/5482"],
    #     ),
    #     DocumentTableCell(
    #         row_index=1,
    #         column_index=0,
    #         content="J=2",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     0.9323,
    #                     2.2236,
    #                     1.4066,
    #                     2.2236,
    #                     1.4066,
    #                     2.4134,
    #                     0.9323,
    #                     2.4086,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352814, length=3)],
    #         elements=["/paragraphs/5483"],
    #     ),
    #     DocumentTableCell(
    #         row_index=1,
    #         column_index=1,
    #         content="Number of replicates per series.",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     1.4066,
    #                     2.2236,
    #                     5.4235,
    #                     2.2236,
    #                     5.4235,
    #                     2.4181,
    #                     1.4066,
    #                     2.4134,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352818, length=32)],
    #         elements=["/paragraphs/5484"],
    #     ),
    #     DocumentTableCell(
    #         row_index=2,
    #         column_index=0,
    #         content="K=6",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     0.9323,
    #                     2.4086,
    #                     1.4066,
    #                     2.4134,
    #                     1.4066,
    #                     2.7692,
    #                     0.9323,
    #                     2.7692,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352851, length=3)],
    #         elements=["/paragraphs/5485"],
    #     ),
    #     DocumentTableCell(
    #         row_index=2,
    #         column_index=1,
    #         content="Number of validation materials with known target concentration levels, set between 0.20 and 10.0 µg/l.",
    #         bounding_regions=[
    #             BoundingRegion(
    #                 page_number=161,
    #                 polygon=[
    #                     1.4066,
    #                     2.4134,
    #                     5.4235,
    #                     2.4181,
    #                     5.4235,
    #                     2.774,
    #                     1.4066,
    #                     2.7692,
    #                 ],
    #             )
    #         ],
    #         spans=[DocumentSpan(offset=352855, length=102)],
    #         elements=["/paragraphs/5486"],
    #     ),
    # ],
    # spans=[DocumentSpan(offset=352737, length=220)],
    # bounding_regions=[
    #     BoundingRegion(page_number=161, polygon=[0.9323, 2.0053, 5.4187, 2.4181]),
    # ],
    caption="Table 1",
    footnotes=[
        DocumentFootnote(content="Footnote 1"),
        DocumentFootnote(content="Footnote 2"),
    ],
)


class MockDocumentTableLoader(DocumentTableLoaderBase):
    def load_di_document_table(self, di_document_table: DocumentTable) -> tuple:
        return ("Table MD", pd.DataFrame())


mock_document_table_loader = MockDocumentTableLoader()
di_document_table_md, di_document_table_df = (
    mock_document_table_loader.load_di_document_table(di_document_table)
)


class TestTableFormatConfig:
    def test_format_page_element_text_only(self):
        table_format_config = TableFormatConfig(
            include=True,
            table_loader=mock_document_table_loader,
            text_format="Table:\n{caption}\n87{table}\n23{footnotes}1",
            content_outputs=["text"],
        )
        caption = "Table 1"
        footnotes = "Footnote 1\nFootnote 2"
        element_id = "/paragraphs/24"
        dummy_table_md, _dummy_table_df = (
            mock_document_table_loader.load_di_document_table(di_document_table)
        )
        expected = [
            HaystackDocument(
                id="/paragraphs/24_text",
                content=f"Table:\n{caption}\n87{dummy_table_md}\n23{footnotes}1",
                meta={"element_id": element_id},
            )
        ]
        result = table_format_config.format_page_element(di_document_table, element_id)
        print(result)
        print(expected)
        assert result == expected

    def test_format_page_element_text_then_df(self):
        table_format_config = TableFormatConfig(
            include=True,
            table_loader=mock_document_table_loader,
            text_format="Table:\n{caption}\n87{table}\n23{footnotes}1",
            content_outputs=["text", "dataframe"],
        )
        caption = "Table 1"
        footnotes = "Footnote 1\nFootnote 2"
        element_id = "/paragraphs/23"
        dummy_table_md, dummy_table_df = (
            mock_document_table_loader.load_di_document_table(di_document_table)
        )
        expected = [
            HaystackDocument(
                id="/paragraphs/23_text",
                content=f"Table:\n{caption}\n87{dummy_table_md}\n23{footnotes}1",
                meta={"element_id": element_id},
            ),
            HaystackDocument(
                "/paragraphs/23_dataframe",
                dataframe=dummy_table_df,
                meta={"element_id": element_id},
            ),
        ]
        result = table_format_config.format_page_element(di_document_table, element_id)
        print(result)
        print(expected)
        assert result == expected


@pytest.mark.parametrize(
    "content,matching_formulas,expected",
    [
        ["", [], ""],
        ["rest of string", [], "rest of string"],
        [
            ":formula: rest of string",
            [
                DocumentFormula(
                    value="\\left( \\beta \\% = 8 0 \\% \\right.",
                    span=DocumentSpan(offset=0, length=8),
                )
            ],
            "\\left( \\beta \\% = 8 0 \\% \\right. rest of string",
        ],
        [
            ":formula: rest :formula: of string",
            [
                DocumentFormula(
                    value="\\left( \\beta \\% = 8 0 \\% \\right.",
                    span=DocumentSpan(offset=0, length=8),
                ),
                DocumentFormula(
                    value="second substitution", span=DocumentSpan(offset=15, length=8)
                ),
            ],
            "\\left( \\beta \\% = 8 0 \\% \\right. rest second substitution of string",
        ],
        [
            ":formula: rest :formula: of string :formula:",
            [
                DocumentFormula(
                    value="\\left( \\beta \\% = 8 0 \\% \\right.",
                    span=DocumentSpan(offset=0, length=8),
                ),
                DocumentFormula(
                    value="second substitution", span=DocumentSpan(offset=15, length=8)
                ),
                DocumentFormula(
                    value="final substitution", span=DocumentSpan(offset=30, length=8)
                ),
            ],
            "\\left( \\beta \\% = 8 0 \\% \\right. rest second substitution of string final substitution",
        ],
    ],
)
def test_substitute_content_formulas(content, matching_formulas, expected):
    result = substitute_content_formulas(content, matching_formulas)
    print(result)
    print(expected)
    assert result == expected
