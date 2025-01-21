import json
import logging
import os
from typing import Optional

import azure.functions as func
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.components.content_understanding_client import (
    AzureContentUnderstandingClient,
    get_existing_analyzer_ids,
)
from src.helpers.common import MeasureRunTime
from src.helpers.content_understanding import (
    condense_webvtt_transcription,
    cu_fields_dict_to_markdown,
)

load_dotenv()

bp_content_understanding_audio = func.Blueprint()
FUNCTION_ROUTE = "content_understanding_audio"

# Load environment variables
CONTENT_UNDERSTANDING_ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")
CONTENT_UNDERSTANDING_KEY = os.getenv("CONTENT_UNDERSTANDING_KEY")

# Load existing analyzer schemas
with open("config/content_understanding_schemas.json", "r") as f:
    CONTENT_UNDERSTANDING_SCHEMAS = json.load(f)

cu_client = AzureContentUnderstandingClient(
    endpoint=CONTENT_UNDERSTANDING_ENDPOINT,
    subscription_key=CONTENT_UNDERSTANDING_KEY,
    api_version="2024-12-01-preview",
    enable_face_identification=False,
)

# # Get list of existing CU analyzers
existing_cu_analyzer_ids = get_existing_analyzer_ids(cu_client)


class FunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    success: bool = Field(
        default=False, description="Indicates whether the pipeline was successful."
    )
    error_text: Optional[str] = Field(
        default=None,
        description="If an error occurred, this field will contain the error message.",
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )
    condensed_transcription: Optional[str] = Field(
        default=None,
        description="The condensed transcription extracted from the audio file.",
    )
    formatted_fields_md: Optional[str] = Field(
        default=None,
        description="The extracted fields in markdown format.",
    )
    cu_raw_response: Optional[dict] = Field(
        default=None, description="The raw API response from Content Understanding."
    )
    cu_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to extract the text using Content Understanding.",
    )


@bp_content_understanding_audio.route(route=FUNCTION_ROUTE)
def content_understanding_audio(
    req: func.HttpRequest,
) -> func.HttpResponse:
    """
    This function processes a request to extract a summary and key fields from
    an audio recording using Azure Content Understanding. If an error occurs at
    any stage, the function will return a partial response with the error
    message and the fields that have been populated up to that point.
    """
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    cu_client = AzureContentUnderstandingClient(
        endpoint=CONTENT_UNDERSTANDING_ENDPOINT,
        subscription_key=CONTENT_UNDERSTANDING_KEY,
        api_version="2024-12-01-preview",
        enable_face_identification=False,
    )
    # Create the object to hold all intermediate and final values. We will progressively update
    # values as each stage of the pipeline is completed, allowing us to return a partial
    # response in case of an error at any stage.
    output_model = FunctionReponseModel(success=False)
    try:
        # Create error_text and error_code variables. These will be updated as
        # we move through the pipeline so that if a step fails, the vars reflect
        # what has failed. If all steps complete successfully, the vars are
        # never used.
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        # Check the request body
        request_json_content = json.loads(req.files["json"].read().decode("utf-8"))
        analyzer_id = request_json_content.get("analyzer_id", None)
        file_bytes = req.files["file"].read()

        ### 1. Ensure the analyzer exists
        error_text = "Invalid Analyzer ID."
        error_code = 500

        # Check if the analyzer already exists when the resource was last checked.
        global existing_cu_analyzer_ids
        if analyzer_id not in existing_cu_analyzer_ids:
            # Refresh the list of existing analyzers
            existing_cu_analyzer_ids = get_existing_analyzer_ids(cu_client)
            if analyzer_id not in existing_cu_analyzer_ids:
                # Analyzer is not available or deployed in the resource.
                raise KeyError(
                    (
                        f"Analyzer ID '{analyzer_id}' is not available. "
                        "Ensure that the Analyzer has already been created within the AI services resource."
                    )
                )

        ### 2. Extract the content using Content Understanding
        error_text = "An error occurred during Content Understanding extraction."
        with MeasureRunTime() as cu_timer:
            response = cu_client.begin_analyze(
                analyzer_id=analyzer_id,
                file_bytes=file_bytes,
            )
            cu_result = cu_client.poll_result(response)
        output_model.cu_raw_response = cu_result
        output_model.cu_time_taken_secs = cu_timer.time_taken

        ### 3. Process the markdown WEBVTT transcript to a slightly more readable format
        error_text = "An error occurred during post-processing."

        webvtt_str = cu_result["result"]["contents"][0]["markdown"][3:-3]
        condensed_transcription = condense_webvtt_transcription(webvtt_str)
        output_model.condensed_transcription = condensed_transcription

        ### 4. Format the fields in markdown for easy reading
        output_model.formatted_fields_md = cu_fields_dict_to_markdown(
            cu_result["result"]["contents"][0]["fields"]
        )

        ### 5. All steps completed successfully, set success=True and return the final result
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
