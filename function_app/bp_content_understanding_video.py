import json
import logging
import os
import re
from itertools import chain
from typing import Optional

import azure.functions as func
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.components.content_understanding_client import (
    AzureContentUnderstandingClient,
    get_existing_analyzer_ids,
)
from src.components.doc_intelligence import base64_img_str_to_url
from src.helpers.common import MeasureRunTime
from src.helpers.content_understanding import (
    cu_fields_dict_to_markdown,
    extract_frames_from_video_bytes,
)
from src.helpers.image import pil_img_to_base64_bytes, resize_img_by_max

load_dotenv()

bp_content_understanding_video = func.Blueprint()
FUNCTION_ROUTE = "content_understanding_video"

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

# Get list of existing CU analyzers
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
    cu_raw_response: Optional[dict] = Field(
        default=None, description="The raw API response from Content Understanding."
    )
    cu_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to extract the text using Content Understanding.",
    )
    segment_formatted_field_mds: Optional[list[str]] = Field(
        default=None,
        description="A list of markdown summaries of the extracted fields for each segment of the video.",
    )
    rich_markdown_output: Optional[str] = Field(
        default=None,
        description=(
            "A copy of the markdown field provided by the Content Understanding API response, but with key frame images"
            " embedded directly into the string (as base64 image data) instead of references to video timestamps."
        ),
    )


@bp_content_understanding_video.route(route=FUNCTION_ROUTE)
def content_understanding_video(
    req: func.HttpRequest,
) -> func.HttpResponse:
    """
    This function processes a request to extract information from a video file
    using Azure Content Understanding. If an error occurs at any stage, the
    function will return a partial response with the error message and the
    fields that have been populated up to that point.
    """
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
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
            cu_result = cu_client.poll_result(response, timeout_seconds=180)
        output_model.cu_raw_response = cu_result
        output_model.cu_time_taken_secs = cu_timer.time_taken

        ### 3. Extract key frames from the video input
        error_text = "An error occurred during post-processing."
        all_key_frame_times_ms = chain.from_iterable(
            [
                segment_result.get("KeyFrameTimesMs", [])
                for segment_result in cu_result["result"]["contents"]
            ]
        )
        key_frame_mapper = extract_frames_from_video_bytes(
            file_bytes, all_key_frame_times_ms, img_type="pil"
        )
        # Reduce the size of the frames
        key_frame_mapper = {
            frame_time_ms: resize_img_by_max(img, 480)
            for frame_time_ms, img in key_frame_mapper.items()
        }

        segment_formatted_field_mds = []

        rich_markdown_output = ""

        for segment_contents in cu_result["result"]["contents"]:
            # Get key frames and segment summary
            segment_md = segment_contents.get("markdown", "")

            # Replace image '![](KeyFrame.1234.jpg)' references with base64 strings

            pattern = r"!\[\]\(keyFrame\.\d+\.jpg\)"
            matches = re.findall(pattern, segment_md)

            for match in matches:
                key_frame_time_ms = int(match.split(".")[1].split(".")[0])
                base64_img_str = pil_img_to_base64_bytes(
                    key_frame_mapper[key_frame_time_ms]
                )
                segment_md = segment_md.replace(
                    match,
                    "![]({})".format(
                        base64_img_str_to_url(
                            base64_img_str.decode(), mime_type="image/jpeg"
                        )
                    ),
                )

            rich_markdown_output += f"{segment_md}\n\n"

            segment_formatted_field_md = cu_fields_dict_to_markdown(
                segment_contents.get("fields", {})
            )
            segment_formatted_field_mds.append(segment_formatted_field_md)

        # Convert the key frames to base64
        output_model.rich_markdown_output = rich_markdown_output
        output_model.segment_formatted_field_mds = segment_formatted_field_mds

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
