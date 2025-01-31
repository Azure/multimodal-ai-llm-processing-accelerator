import datetime
import json
import logging
import mimetypes
import os
from typing import Any, Dict, Optional
from uuid import uuid4

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import Field
from src.components.doc_intelligence import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
    convert_processed_di_docs_to_openai_message,
)
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict
from src.schema import LLMResponseBaseModel

load_dotenv()

aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")

# Create the clients for Document Intelligence and Azure OpenAI
DOC_INTEL_MODEL_ID = "prebuilt-layout"  # Set Document Intelligence model ID

# Set up the Document Intelligence v4.0 preview client. This will allow us to
# use the latest features of the Document Intelligence service. Check out the
# Document Intelligence Processor Walkthrough Notebook for more information
# (within the `notebooks` folder).
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=DefaultAzureCredential(),
    api_version="2024-07-31-preview",
)
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
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
    )
)


# Setup Pydantic models for validation of LLM calls, and the Function response itself
class LLMExtractedFieldsModel(LLMResponseBaseModel):
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

    account_no: str = Field(
        description="The account number to be opened.",
        examples=["1189234623462"],
    )
    branch_ifsc: str = Field(
        description="The branch IFSC.",
        examples=["SWTK892374"],
    )
    title: str = Field(
        description="The Title of the account holder.",
        examples=["Mrs"],
    )
    first_name: str = Field(
        description="The first name of the account holder.",
        examples=["John"],
    )
    last_name: str = Field(
        description="The last name of the account holder.",
        examples=["Smith"],
    )
    day_of_birth: str = Field(
        description="The day of birth of the account holder.",
        examples=["31"],
    )
    month_of_birth: str = Field(
        description="The month of birth of the account holder.",
        examples=["12"],
    )
    year_of_birth: str = Field(
        description="The year of birth of the account holder.",
        examples=["1985"],
    )
    pan: str = Field(
        description="The PAN of the account holder.",
        examples=["SKIFP1234K"],
    )
    customer_id: str = Field(
        description="The Customer ID of the account holder.",
        examples=["128740928"],
    )


class FunctionReponseModel(LLMExtractedFieldsModel):
    """
    Defines the schema that will be returned by the function.
    Inherit from LLMExtractedFieldsModel and add a couple additional fields.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID of the pipeline result.",
    )
    success: bool = Field(
        False, description="Indicates whether the pipeline was successful."
    )
    filename: str = Field(
        description="The filename of the blob that was processed.",
    )
    blobname: str = Field(
        description="The name of the blob that was processed.",
    )
    ttl: Optional[int] = Field(
        description="The time-to-live of the item in seconds.",
        default=None,
    )
    date_processed: str = Field(
        description="The date and time that the file was processed.",
        default_factory=lambda: datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(),
    )


# Create the system prompt for the LLM, dynamically including the JSON schema
# of the expected response so that any changes to the schema are automatically
# reflected in the prompt, and in a JSON format that is similar in structure
# to the training data on which the LLM was trained (increasing reliability of
# the result).
LLM_SYSTEM_PROMPT = (
    "You are a data extraction expert. "
    "Your task is to review the following information and extract all of the information that appears in the form.\n"
    f"{LLMExtractedFieldsModel.get_prompt_json_example(include_preceding_json_instructions=True)}"
)


def get_structured_extraction_func_outputs(
    input_blob: func.InputStream,
) -> Dict[str, Any]:
    logging.info(f"Function triggered by blob name: {input_blob.name}")

    try:
        # Check mime_type of the request data
        mime_type = mimetypes.guess_type(input_blob.name)[0]
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                    ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
                ),
                status_code=422,
            )
        req_body = input_blob.read()
        # Load the PDF into separate images
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )
        # Extract the text using Document Intelligence
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
        merged_processed_content_docs = (
            doc_intel_result_processor.merge_adjacent_text_content_docs(
                processed_content_docs
            )
        )
        ### 3. Create the messages to send to the LLM in the following order:
        #      i. System prompt
        #      ii. Extracted text and images from Document Intelligence
        content_openai_message = convert_processed_di_docs_to_openai_message(
            merged_processed_content_docs, role="user"
        )
        input_messages = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            content_openai_message,
        ]
        ### 4. Send request to LLM
        llm_result = aoai_client.chat.completions.create(
            messages=input_messages,
            model=AOAI_LLM_DEPLOYMENT,
            response_format={"type": "json_object"},  # Ensure we get JSON responses
        )
        ### 5. Validate that the LLM response matches the expected schema
        llm_structured_response = LLMExtractedFieldsModel(
            **json.loads(llm_result.choices[0].message.content)
        )
        ### 8. All steps completed successfully, set success=True and return the final result
        # All steps completed successfully, set success=True and return the final result
        output_model = FunctionReponseModel(
            id=str(uuid4()),
            success=True,
            filename=os.path.basename(input_blob.name),
            blobname=input_blob.name,
            ttl=300,  # Set time-to-live - item will expire in if container is configured to use TTL
            **llm_structured_response.dict(),
        )
        return output_model.model_dump(mode="python")
    except Exception as e:
        raise RuntimeError("An error occurred during processing.") from e
