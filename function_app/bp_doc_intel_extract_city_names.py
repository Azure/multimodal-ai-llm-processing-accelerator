import json
import logging
import os
from typing import Optional

import azure.functions as func
from dotenv import load_dotenv
from haystack import Document
from haystack.components.converters import AzureOCRDocumentConverter
from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator
from haystack.dataclasses import ByteStream, ChatMessage
from haystack.utils import Secret
from pydantic import BaseModel, Field
from src.helpers.common import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    MeasureRunTime,
    haystack_doc_to_string,
)
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_doc_intel_extract_city_names = func.Blueprint()

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_DEPLOYMENT = os.getenv("AOAI_DEPLOYMENT")
# Load the API key as a Secret, so that it is not logged in any traces or saved if the component is exported.
DOC_INTEL_API_KEY_SECRET = Secret.from_env_var("DOC_INTEL_API_KEY")
AOAI_API_KEY_SECRET = Secret.from_env_var("AOAI_API_KEY")

# Setup Haystack components
di_converter = AzureOCRDocumentConverter(
    endpoint=DOC_INTEL_ENDPOINT,
    api_key=DOC_INTEL_API_KEY_SECRET,
    model_id="prebuilt-layout",
)
azure_generator = AzureOpenAIChatGenerator(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_DEPLOYMENT,
    api_key=AOAI_API_KEY_SECRET,
    api_version="2024-06-01",
    generation_kwargs={
        "response_format": {"type": "json_object"}
    },  # Ensure we get JSON responses
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
    di_extracted_text: Optional[list[dict]] = Field(
        None, description="The raw text content extracted by Document Intelligence."
    )
    di_raw_response: Optional[list[dict]] = Field(
        None, description="The raw API response from Document Intelligence."
    )
    di_time_taken_secs: Optional[float] = Field(
        None,
        description="The time taken to extract the text using Document Intelligence.",
    )
    llm_input_messages: Optional[list[dict]] = Field(
        None, description="The messages that were sent to the LLM."
    )
    llm_reply_messages: Optional[list[dict]] = Field(
        None, description="The messages that were received from the LLM."
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


@bp_doc_intel_extract_city_names.route(route="doc_intel_extract_city_names")
def doc_intel_extract_city_names(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    try:
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
                status_code=400,
            )

        # Check the request body
        req_body = req.get_body()
        if len(req_body) == 0:
            return func.HttpResponse(
                "Please provide a base64 encoded PDF in the request body.",
                status_code=400,
            )
        # Load the data into a ByteStream object, as expected by Haystack components
        try:
            bytestream = ByteStream(data=req_body, mime_type=mime_type)
        except Exception as _e:
            output_model.error_text = "An error occurred while loading the document."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Extract the text using Document Intelligence
        try:
            with MeasureRunTime() as di_timer:
                di_result = di_converter.run(sources=[bytestream])
            di_result_docs: list[Document] = di_result["documents"]
            output_model.di_extracted_text = [doc.to_dict() for doc in di_result_docs]
            output_model.di_raw_response = di_result["raw_azure_response"]
            output_model.di_time_taken_secs = di_timer.time_taken
        except Exception as _e:
            output_model.error_text = (
                "An error occurred during Document Intelligence extraction."
            )
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Create the messages to send to the LLM
        try:
            input_messages = [
                ChatMessage.from_system(LLM_SYSTEM_PROMPT),
                *[
                    ChatMessage.from_user(haystack_doc_to_string(doc))
                    for doc in di_result_docs
                ],
            ]
            output_model.llm_input_messages = [
                msg.to_openai_format() for msg in input_messages
            ]
        except Exception as _e:
            output_model.error_text = (
                "An error occurred while creating the LLM input messages."
            )
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Send request to LLM
        try:
            with MeasureRunTime() as llm_timer:
                llm_result = azure_generator.run(messages=input_messages)
            output_model.llm_time_taken_secs = llm_timer.time_taken
        except Exception as _e:
            output_model.error_text = "An error occurred when sending the LLM request."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Validate that the LLM response matches the expected schema
        try:
            output_model.llm_reply_messages = [
                msg.to_openai_format() for msg in llm_result["replies"]
            ]
            output_model.llm_raw_response = llm_result["replies"][0].content
            llm_structured_response = LLMCityNamesModel(
                **json.loads(llm_result["replies"][0].content)
            )
            output_model.result = llm_structured_response
        except Exception as _e:
            output_model.error_text = "An error occurred when validating the LLM's returned response into the expected schema."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # All steps completed successfully, set success=True and return the final result
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as _e:
        logging.exception("An error occurred during processing.")
        return func.HttpResponse(
            "An error occurred during processing.",
            status_code=500,
        )
