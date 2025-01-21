import json
import logging
import os
from typing import Optional

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.components.doc_intelligence import (
    DefaultDocumentFigureProcessor,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
    PageDocumentListSplitter,
    convert_processed_di_doc_chunks_to_markdown,
)
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict

logger = logging.getLogger(__name__)

load_dotenv()

bp_multimodal_doc_intel_processing = func.Blueprint()

FUNCTION_ROUTE = "multimodal_doc_intel_processing"

DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("DOC_INTEL_API_KEY")

# Create the client for Document Intelligence and Azure OpenAI
DOC_INTEL_MODEL_ID = "prebuilt-layout"  # Set Document Intelligence model ID

# Set up the Document Intelligence v4.0 preview client. This will allow us to
# use the latest features of the Document Intelligence service. Check out the
# Document Intelligence Processor Walkthrough Notebook for more information
# (within the `notebooks` folder).
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_API_KEY),
    api_version="2024-07-31-preview",
)


class FunctionRequestModel(BaseModel):
    """
    Defines the schema that will be expected in the request body. We'll use this to
    ensure that the request contains the correct values and structure, and to allow
    a partially filled request to be processed in case of an error.
    """

    file: bytes = Field(description="The text to be summarized")
    include_page_images_after_content: bool = Field(
        description="Whether to include page images after the text content of each page"
    )
    extract_and_crop_inline_figures: bool = Field(
        description="Whether to extract and crop inline figures"
    )


class FunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    processed_di_result_markdown: str = Field(
        description="Markdown text of the processed Document Intelligence Result."
    )
    raw_di_response: dict = Field(description="Raw document intelligence response.")
    di_time_taken_secs: Optional[float] = Field(
        description="The time taken to process the document with Document Intelligence.",
    )


@bp_multimodal_doc_intel_processing.route(route=FUNCTION_ROUTE)
def multimodal_doc_intel_processing(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    try:
        # Load and validate input data
        error_text = "Error while loading and validating the input data."
        error_code = 422

        # Check the request body
        request_json_content = json.loads(req.files["json"].read().decode("utf-8"))
        include_page_images_after_content = request_json_content.get(
            "include_page_images_after_content", False
        )
        extract_and_crop_inline_figures = request_json_content.get(
            "extract_and_crop_inline_figures", False
        )

        # Now construct the a splitter class which can separate the outputs into different chunks
        pages_per_chunk = request_json_content.get("pages_per_chunk", 3)
        page_chunk_splitter = PageDocumentListSplitter(pages_per_chunk=pages_per_chunk)

        file_bytes = req.files["doc_for_extraction"].read()
        file_mime_type = req.files["doc_for_extraction"].content_type

        # Create the Doc Intelligence result processor. This can be configured to
        # process the raw Doc Intelligence result into a format that is easier
        # to work with downstream.
        doc_intel_result_processor = DocumentIntelligenceProcessor(
            page_processor=DefaultDocumentPageProcessor(
                page_img_order="after" if include_page_images_after_content else None,
            ),
            figure_processor=DefaultDocumentFigureProcessor(
                output_figure_img=extract_and_crop_inline_figures
            ),
        )

        # Process the document with Document Intelligence
        error_text = "An error occurred while processing the document."
        error_code = 422

        # Load content as images
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            file_bytes, file_mime_type, starting_idx=1, pdf_img_dpi=100
        )
        # Get Doc Intelligence resul;t
        poller = di_client.begin_analyze_document(
            model_id=DOC_INTEL_MODEL_ID,
            analyze_request=AnalyzeDocumentRequest(bytes_source=file_bytes),
        )
        di_result = poller.result()
        # Process the result into Documents containing the content of every element
        processed_content_docs = doc_intel_result_processor.process_analyze_result(
            analyze_result=di_result,
            doc_page_imgs=doc_page_imgs,
            on_error="raise",
        )
        # Chunk the content by page
        page_chunked_content_docs = page_chunk_splitter.split_document_list(
            processed_content_docs
        )
        # Merge adjacent text content together (reducing the number of objects)
        merged_page_chunked_content_docs = (
            doc_intel_result_processor.merge_adjacent_text_content_docs(
                page_chunked_content_docs
            )
        )
        # Convert the chunks into a single Markdown string
        di_processed_md = convert_processed_di_doc_chunks_to_markdown(
            merged_page_chunked_content_docs
        )
        return func.HttpResponse(
            body=di_processed_md,
            mimetype="text/plain",
            status_code=200,
        )
    except Exception as e:
        logging.exception(e)
        return func.HttpResponse(error_text, status_code=error_code)
