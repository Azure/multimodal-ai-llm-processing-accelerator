import datetime
import json
import logging
import mimetypes
import os
from typing import Any, Dict
from uuid import uuid4

import azure.functions as func
from dotenv import load_dotenv
from haystack import Document
from haystack.components.converters import AzureOCRDocumentConverter
from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator
from haystack.dataclasses import ByteStream, ChatMessage
from haystack.utils import Secret
from pydantic import Field
from src.components.doc_intelligence import VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES
from src.components.pymupdf import PyMuPDFConverter
from src.helpers.common import haystack_doc_to_string
from src.schema import LLMResponseBaseModel

load_dotenv()

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
# Load the API key as a Secret, so that it is not logged in any traces or saved if the component is exported.
DOC_INTEL_API_KEY_SECRET = Secret.from_env_var("DOC_INTEL_API_KEY")
AOAI_API_KEY_SECRET = Secret.from_env_var("AOAI_API_KEY")

# Setup Haystack components
pymupdf_converter = PyMuPDFConverter(
    to_img_dpi=100,
    correct_img_rotation=False,
)
di_converter = AzureOCRDocumentConverter(
    endpoint=DOC_INTEL_ENDPOINT,
    api_key=DOC_INTEL_API_KEY_SECRET,
    model_id="prebuilt-read",
)
azure_generator = AzureOpenAIChatGenerator(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    api_key=AOAI_API_KEY_SECRET,
    api_version="2024-06-01",
    generation_kwargs={
        "response_format": {"type": "json_object"}
    },  # Ensure we get JSON responses
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
    logging.info(f"Blob name: {input_blob.name}")

    try:
        # Check mime_type of the request data
        mime_type = mimetypes.guess_type(input_blob.name)[0]
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                    ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
                ),
                status_code=400,
            )
        # Load the data into a ByteStream object, as expected by Haystack components
        try:
            bytestream = ByteStream(data=input_blob.read(), mime_type=mime_type)
        except Exception as e:
            raise RuntimeError("An error occurred while loading the document.") from e
        # Extract the images from the PDF using PyMuPDF
        try:
            pymupdf_result = pymupdf_converter.run(sources=[bytestream])
            pdf_images = pymupdf_result["images"]
        except Exception as e:
            raise RuntimeError(
                "An error occurred during PyMuPDF image extraction."
            ) from e
        # Extract the text using Document Intelligence
        try:
            di_result = di_converter.run(sources=[bytestream])
            di_result_docs: list[Document] = di_result["documents"]
        except Exception as e:
            raise RuntimeError(
                "An error occurred during Document Intelligence extraction."
            ) from e
        # Create the messages to send to the LLM in the following order:
        # 1. System prompt
        # 2. Extracted text from Document Intelligence
        # 3. Extracted images from PyMuPDF
        try:
            input_messages = [
                ChatMessage.from_system(LLM_SYSTEM_PROMPT),
                *[
                    ChatMessage.from_user(haystack_doc_to_string(doc))
                    for doc in di_result_docs
                ],
                *[ChatMessage.from_user(bytestream) for bytestream in pdf_images],
            ]
        except Exception as e:
            raise RuntimeError(
                "An error occurred while creating the LLM input messages."
            ) from e
        # Send request to LLM
        try:
            llm_result = azure_generator.run(messages=input_messages)
        except Exception as e:
            raise RuntimeError("An error occurred when sending the LLM request.") from e
        # Validate that the LLM response matches the expected schema
        try:
            llm_structured_response = LLMExtractedFieldsModel(
                **json.loads(llm_result["replies"][0].content)
            )
        except Exception as e:
            raise RuntimeError(
                "An error occurred when validating the LLM's returned response into the expected schema."
            ) from e
        # All steps completed successfully, set success=True and return the final result
        output_model = FunctionReponseModel(
            id=str(uuid4()),
            success=True,
            filename=os.path.basename(input_blob.name),
            blobname=input_blob.name,
            **llm_structured_response.dict(),
        )
        result = output_model.model_dump(mode="python")
        return result
    except Exception as e:
        raise RuntimeError("An error occurred during processing.") from e
