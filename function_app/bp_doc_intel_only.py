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
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
)
from src.helpers.common import MeasureRunTime
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict

logger = logging.getLogger(__name__)

load_dotenv()

bp_doc_intel_only = func.Blueprint()

FUNCTION_ROUTE = "doc_intel_only"

DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("DOC_INTEL_API_KEY")

# Create the client for Document Intelligence and Azure OpenAI
DOC_INTEL_MODEL_ID = "prebuilt-layout"  # Set Document Intelligence model ID
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_API_KEY),
    api_version="2024-07-31-preview",
)

# Create the Doc Intelligence result processor. This can be configured to
# process the raw Doc Intelligence result into a format that is easier
# to work with downstream.
doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(
        page_img_order=None,  # Exclude each page image after the page's text content"
    )
)


class FunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    di_extracted_text: str = Field(
        description="Raw text xtracted by Document Intelligence."
    )
    raw_di_response: dict = Field(description="Raw document intelligence response.")
    di_time_taken_secs: Optional[float] = Field(
        description="The time taken to process the document with Document Intelligence.",
    )


@bp_doc_intel_only.route(route=FUNCTION_ROUTE)
def doc_intel_only(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info(
            f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request."
        )
        # Load and validate input data
        error_text = "Error while loading and validating the input data."
        error_code = 422

        mime_type = req.headers.get("Content-Type")
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                    ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
                ),
                status_code=422,
            )

        req_body = req.get_body()

        if len(req_body) == 0:
            return func.HttpResponse(
                "No file was provided in the request body.", status_code=422
            )
        # Process the document with Document Intelligence
        error_text = "An error occurred while processing the document."
        error_code = 422

        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )
        with MeasureRunTime() as di_timer:
            poller = di_client.begin_analyze_document(
                model_id=DOC_INTEL_MODEL_ID,
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
            processed_content_docs = doc_intel_result_processor.process_analyze_result(
                analyze_result=di_result,
                doc_page_imgs=doc_page_imgs,
                on_error="raise",
            )
            merged_subchunk_content_docs = (
                doc_intel_result_processor.merge_subchunk_text_content(
                    processed_content_docs
                )
            )
            merged_subchunk_content = list()
            for subchunk in merged_subchunk_content_docs:
                for document in subchunk:
                    if document.content:
                        merged_subchunk_content.append(document.content)
            merged_subchunk_content = "\n".join(merged_subchunk_content)
        # Return joined text content
        output_model = FunctionReponseModel(
            di_extracted_text=merged_subchunk_content,
            raw_di_response=di_result.as_dict(),
            di_time_taken_secs=di_timer.time_taken,
        )
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as e:
        logging.exception(e)
        return func.HttpResponse(error_text, status_code=error_code)
