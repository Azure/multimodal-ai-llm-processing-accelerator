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
    cu_fields_dict_to_markdown,
    draw_fields_on_imgs,
    enrich_extracted_cu_fields,
)
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict
from src.helpers.image import pil_img_to_base64_bytes, resize_img_by_max, rotate_img_pil

load_dotenv()

bp_content_understanding_document = func.Blueprint()
FUNCTION_ROUTE = "content_understanding_document"

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
    enriched_cu_fields: Optional[dict] = Field(
        default=None,
        description="The enriched fields result from Content Understanding.",
    )
    formatted_fields_md: Optional[str] = Field(
        default=None,
        description="The field outputs in a pre-formatted markdown format.",
    )
    cu_raw_response: Optional[dict] = Field(
        default=None, description="The raw API response from Content Understanding."
    )
    cu_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to extract the text using Content Understanding.",
    )
    result_imgs_with_bboxes: Optional[dict[int, bytes]] = Field(
        default=None,
        description="Dictionary of page images with bounding boxes drawn around the extracted fields.",
    )


@bp_content_understanding_document.route(route=FUNCTION_ROUTE)
def content_understanding_document(
    req: func.HttpRequest,
) -> func.HttpResponse:
    """
    This function processes a request to extract fields from a PDF document
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
        file_mime_type = req.files["file"].content_type

        ### 1. Load the images from the PDF/image input
        error_text = "An error occurred during image extraction."
        error_code = 500
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            file_bytes, file_mime_type, starting_idx=1, pdf_img_dpi=100
        )

        ### 2. Ensure the analyzer exists
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

        ### 3. Enrich the raw CU API result with additional metadata (normalized polygons, page number etc)
        error_text = "An error occurred during post-processing."
        enriched_cu_fields = enrich_extracted_cu_fields(
            cu_result["result"]["contents"][0]
        )
        output_model.enriched_cu_fields = enriched_cu_fields

        ### 4. Format the fields in markdown for easy reading
        output_model.formatted_fields_md = cu_fields_dict_to_markdown(
            enriched_cu_fields
        )

        ### 5. Draw bounding boxes of the extracted fields on images of the input document
        # With the locations of the extracted fields now known, we can draw
        # bounding boxes on the image and to make it easier to digest the output.
        # Draw the bounding boxes on the image
        pil_imgs = draw_fields_on_imgs(
            enriched_fields=enriched_cu_fields, pil_imgs=doc_page_imgs
        )
        # Resize the images to reduce transfer size
        pil_imgs = {
            page_num: resize_img_by_max(pil_img, max_height=1000, max_width=1000)
            for page_num, pil_img in pil_imgs.items()
        }
        # Rotate the images to be the correct orientation
        for page_list_idx, (page_num, pil_img) in enumerate(pil_imgs.items()):
            pil_imgs[page_num] = rotate_img_pil(
                pil_img,
                angle=cu_result["result"]["contents"][0]["pages"][page_list_idx][
                    "angle"
                ],
            )
        output_model.result_imgs_with_bboxes = {
            page_num: pil_img_to_base64_bytes(pil_img)
            for page_num, pil_img in pil_imgs.items()
        }

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
