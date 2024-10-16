import json
import logging
import os
from typing import Optional

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from haystack import Document
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from src.components.doc_intelligence import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DefaultDocumentFigureProcessor,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
    convert_content_chunks_to_openai_messages,
)
from src.helpers.common import MeasureRunTime
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_doc_intel_extract_city_names = func.Blueprint()

FUNCTION_ROUTE = "doc_intel_extract_city_names"

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("DOC_INTEL_API_KEY")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")

# Create the clients for Document Intelligence and Azure OpenAI
DOC_INTEL_MODEL_ID = "prebuilt-read"  # Set Document Intelligence model ID

di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_API_KEY),
    api_version="2024-07-31-preview",
)
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    api_key=AOAI_API_KEY,
    api_version="2024-06-01",
    timeout=30,
    max_retries=0,
)

# Create the Doc Intelligence result processor. This can be configured to
# process the raw Doc Intelligence result into a format that is easier
# to work with downstream.
doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(
        page_img_order="after",  # Include each page image after the page's text content"
    ),
    figure_processor=DefaultDocumentFigureProcessor(
        output_figure_img=False,  # Exclude cropped figure images from the output
    ),
)


# Setup Pydantic models for validation of LLM calls, and the Function response itself
class LLMCityNamesModel(LLMResponseBaseModel):
    """
    Defines the required JSON schema for the LLM to adhere to. This can be used
    to validate that the LLM's raw text response can be parsed into the format
    that is expected by downstream processes (e.g. when we need to save the data
    into a database).

    This class inherits from LLMResponseBaseModel and sets a description and
    example for each field, allowing us to run `model.get_prompt_json_example()`
    to generate a prompt-friendly string representation of the expected JSON
    that we can provide to the LLM.
    """

    city_names: list[str] = Field(
        description="A list of city names, that were extracted from the text.",
        examples=[["London", "Paris", "New York"]],
    )


class FunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    success: bool = Field(
        False, description="Indicates whether the pipeline was successful."
    )
    result: Optional[LLMCityNamesModel] = Field(
        None, description="The final result of the pipeline."
    )
    func_time_taken_secs: Optional[float] = Field(
        None, description="The total time taken to process the request."
    )
    error_text: Optional[str] = Field(
        None,
        description="If an error occurred, this field will contain the error message.",
    )
    di_extracted_text: Optional[str] = Field(
        None, description="The raw text content extracted by Document Intelligence."
    )
    di_raw_response: Optional[dict] = Field(
        None, description="The raw API response from Document Intelligence."
    )
    di_time_taken_secs: Optional[float] = Field(
        None,
        description="The time taken to extract the text using Document Intelligence.",
    )
    llm_input_messages: Optional[list[dict]] = Field(
        None, description="The messages that were sent to the LLM."
    )
    llm_reply_message: Optional[dict] = Field(
        default=None, description="The message that was received from the LLM."
    )
    llm_raw_response: Optional[str] = Field(
        None, description="The raw text response from the LLM."
    )
    llm_time_taken_secs: Optional[float] = Field(
        None, description="The time taken to receive a response from the LLM."
    )


LLM_SYSTEM_PROMPT = (
    "Your task is to review the following information and extract all city names that appear in the text.\n"
    f"{LLMCityNamesModel.get_prompt_json_example(include_preceding_json_instructions=True)}"
)


@bp_doc_intel_extract_city_names.route(route=FUNCTION_ROUTE)
def doc_intel_extract_city_names(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info(
            f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request."
        )
        # Create an error_text variable. This will be updated as we move through
        # the pipeline so that if a step fails, the error_text var reflects what
        # has failed. If all steps complete successfully, the var is never used.
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()
        # Create the object to hold all intermediate and final values. We will progressively update
        # values as each stage of the pipeline is completed, allowing us to return a partial
        # response in case of an error at any stage.
        output_model = FunctionReponseModel(success=False)
        # Check mime_type of the request data
        mime_type = req.headers.get("Content-Type")
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                    ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
                ),
                status_code=422,
            )

        ### Check the request body
        req_body = req.get_body()
        if len(req_body) == 0:
            return func.HttpResponse(
                "Please provide a base64 encoded PDF in the request body.",
                status_code=422,
            )
        ### 1. Load the images from the PDF/image input
        error_text = "An error occurred during image extraction."
        error_code = 500
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )
        ### Extract the text using Document Intelligence
        error_text = "An error occurred during Document Intelligence extraction."
        with MeasureRunTime() as di_timer:
            poller = di_client.begin_analyze_document(
                model_id=DOC_INTEL_MODEL_ID,
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
            output_model.di_raw_response = di_result.as_dict()
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
        di_result_docs: list[Document] = processed_content_docs
        output_model.di_extracted_text = "\n".join(
            doc.content for doc in di_result_docs if doc.content is not None
        )
        output_model.di_time_taken_secs = di_timer.time_taken
        ### 3. Create the messages to send to the LLM in the following order:
        #      i. System prompt
        #      ii. Extracted text and images from Document Intelligence
        error_text = "An error occurred while creating the LLM input messages."
        # Convert chunk content to OpenAI messages
        content_openai_messages = convert_content_chunks_to_openai_messages(
            merged_subchunk_content_docs, role="user"
        )
        input_messages = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            *content_openai_messages,
        ]
        output_model.llm_input_messages = input_messages

        ### 4. Send request to LLM
        error_text = "An error occurred when sending the LLM request."
        with MeasureRunTime() as llm_timer:
            llm_result = aoai_client.chat.completions.create(
                messages=input_messages,
                model=AOAI_LLM_DEPLOYMENT,
                response_format={"type": "json_object"},  # Ensure we get JSON responses
            )
        output_model.llm_time_taken_secs = llm_timer.time_taken
        ### 5. Validate that the LLM response matches the expected schema
        error_text = "An error occurred when validating the LLM's returned response into the expected schema."
        output_model.llm_reply_message = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content
        llm_structured_response = LLMCityNamesModel(
            **json.loads(llm_result.choices[0].message.content)
        )
        output_model.result = llm_structured_response
        ### 8. All steps completed successfully, set success=True and return the final result
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as _e:
        # If an error occurred at any stage, return the partial response. Update the error_text
        # field to contain the error message, and ensure success=False.
        output_model.success = False
        output_model.error_text = error_text
        output_model.func_time_taken_secs = func_timer.stop()
        logging.exception(output_model.error_text)
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=error_code,
        )
