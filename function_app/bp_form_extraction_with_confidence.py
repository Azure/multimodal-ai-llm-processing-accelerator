import json
import logging
import os
from typing import Optional

import azure.functions as func
import jellyfish
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
from src.helpers.image import (
    draw_polygon_on_pil_img,
    flat_poly_list_to_poly_dict_list,
    pil_img_to_base64,
    resize_img_by_max,
    scale_flat_poly_list,
)
from src.result_enrichment.common import merge_confidence_scores
from src.result_enrichment.doc_intelligence import (
    find_matching_di_lines,
    find_matching_di_words,
)
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_form_extraction_with_confidence = func.Blueprint()

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("DOC_INTEL_API_KEY")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")

# Set the minimum confidence score required for the result to be accepted without requiring human review
MIN_CONFIDENCE_SCORE = 0.8
FUNCTION_ROUTE = "form_extraction_with_confidence"

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


class FieldWithConfidenceModel(BaseModel):
    """
    Defines the enriched schema for extracted fields, including useful metadata.
    """

    value: str = Field(description="The extracted value.")
    doc_intel_content_matches_count: Optional[int] = Field(
        description=(
            "The number of Document Intelligence content objects that matched the extracted value."
        ),
    )
    confidence: Optional[float] = Field(
        description="The confidence score associated with the extracted value.",
    )
    normalized_polygons: Optional[list[list[float]]] = Field(
        description=(
            "The polygons that represent the bounding region of the extracted "
            "value, normalized to between 0-1."
        ),
    )


class ExtractedFieldsWithConfidenceModel(BaseModel):
    """
    Defines the schema for all extracted fields, including useful metadata.
    """

    account_no: FieldWithConfidenceModel = Field(
        description="The account number to be opened."
    )
    branch_ifsc: FieldWithConfidenceModel = Field(description="The branch IFSC.")
    title: FieldWithConfidenceModel = Field(
        description="The Title of the account holder."
    )
    first_name: FieldWithConfidenceModel = Field(
        description="The first name of the account holder."
    )
    last_name: FieldWithConfidenceModel = Field(
        description="The last name of the account holder."
    )
    day_of_birth: FieldWithConfidenceModel = Field(
        description="The day of birth of the account holder."
    )
    month_of_birth: FieldWithConfidenceModel = Field(
        description="The month of birth of the account holder."
    )
    year_of_birth: FieldWithConfidenceModel = Field(
        description="The year of birth of the account holder."
    )
    pan: FieldWithConfidenceModel = Field(description="The PAN of the account holder.")
    customer_id: FieldWithConfidenceModel = Field(
        description="The Customer ID of the account holder."
    )


class FunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    success: bool = Field(
        default=False, description="Indicates whether the pipeline was successful."
    )
    requires_human_review: bool = Field(
        default=False, description="Indicates whether the result requires human review."
    )
    min_extracted_field_confidence_score: Optional[float] = Field(
        default=None,
        description="The minimum confidence score across all extracted fields.",
    )
    required_confidence_score: float = Field(
        description="The minimum confidence score required for the result to be accepted."
    )
    result: Optional[ExtractedFieldsWithConfidenceModel] = Field(
        default=None, description="The final result of the pipeline."
    )
    error_text: Optional[str] = Field(
        default=None,
        description="If an error occurred, this field will contain the error message.",
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )
    di_extracted_text: Optional[str] = Field(
        default=None,
        description="The raw text content extracted by Document Intelligence.",
    )
    di_raw_response: Optional[list[dict]] = Field(
        default=None, description="The raw API response from Document Intelligence."
    )
    di_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to extract the text using Document Intelligence.",
    )
    llm_input_messages: Optional[list[dict]] = Field(
        default=None, description="The messages that were sent to the LLM."
    )
    llm_reply_messages: Optional[dict] = Field(
        default=None, description="The messages that were received from the LLM."
    )
    llm_raw_response: Optional[str] = Field(
        default=None, description="The raw text response from the LLM."
    )
    llm_time_taken_secs: Optional[float] = Field(
        default=None, description="The time taken to receive a response from the LLM."
    )
    result_img_with_bboxes: Optional[bytes] = Field(
        default=None,
        description="The image with bounding boxes drawn around the extracted fields.",
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


@bp_form_extraction_with_confidence.route(route=FUNCTION_ROUTE)
def form_extraction_with_confidence(
    req: func.HttpRequest,
) -> func.HttpResponse:
    """
    This function processes a request to extract fields from a PDF document
    using Document Intelligence and a Language Model. The function runs a series
    of steps to process the document and progressively populate the fields in
    the output model. If an error occurs at any stage, the function will return
    a partial response with the error message and the fields that have been
    populated up to that point.
    """
    try:
        logging.info(
            f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request."
        )
        # Create error_text and error_code variables. These will be updated as
        # we move through the pipeline so that if a step fails, the vars reflect
        # what has failed. If all steps complete successfully, the vars are
        # never used.
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        # Check mime_type of the request data
        mime_type = req.headers.get("Content-Type")
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                    ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
                ),
                status_code=error_code,
            )

        # Check the request body
        req_body = req.get_body()
        if len(req_body) == 0:
            return func.HttpResponse(
                "Please provide a base64 encoded PDF in the request body.",
                status_code=error_code,
            )
        # Create the object to hold all intermediate and final values. We will progressively update
        # values as each stage of the pipeline is completed, allowing us to return a partial
        # response in case of an error at any stage.
        output_model = FunctionReponseModel(
            success=False, required_confidence_score=MIN_CONFIDENCE_SCORE
        )

        ### 1. Load the images from the PDF/image input
        error_text = "An error occurred during image extraction."
        error_code = 500
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )

        ### 2. Extract the text and images using Document Intelligence
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
        output_model.llm_reply_messages = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content
        llm_structured_response = LLMExtractedFieldsModel(
            **json.loads(llm_result.choices[0].message.content)
        )

        ### 6. Add confidence scores from Doc Intelligence to each the extracted fields
        ###    and determine whether we need to escalate to human review
        error_text = (
            "An error occurred when adding confidence scores to the extracted content."
        )
        result = dict()
        # Record whether any fields are missing and the min confidence score across every field
        is_any_field_missing = False
        min_field_confidence_score = 1
        # Get the raw text content from Document Intelligence for the document that was processed
        for field, value in llm_structured_response.__dict__.items():
            # We will check for matches where the LLM output is part of the Doc Intelligence content.
            # This is useful for cases where Doc Intelligence extracts the date of birth as a single word
            # (e.g. 30/12/1985), but where the LLM separates the day, month and year into separate fields.
            def is_lower_exact_match(value: str, content: str) -> bool:
                """Returns True if the lower-case value is equal to the lower-case content."""
                return value.lower() == content.lower()

            def is_lower_value_in_lower_content_without_whitespace(
                value: str, content: str
            ) -> bool:
                """
                Returns True if the lower-case value without whitespace is
                equal to the lower-case content without whitespace.
                """
                return (
                    value.replace(" ", "").lower() in content.replace(" ", "").lower()
                )

            def is_lower_value_in_split_lower_content(value: str, content: str) -> bool:
                """
                Returns True if the lower-case value without whitespace is
                equal to the lower-case content without whitespace.
                """
                return value.lower() in content.lower().split()

            def is_lower_value_within_levenstein_distance_1(
                value: str, content: str
            ) -> bool:
                """
                Returns True if the lower-cased value is within a
                Levenshtein distance of `max_distance` from the lower-cased
                content.
                """
                return jellyfish.levenshtein_distance(value, content) <= 1

            # 1. First try for exact matches by word
            matches = find_matching_di_words(
                value, di_result, match_func=is_lower_exact_match
            )
            if not matches:
                # 2. Try for levenshtein distance of 1 (this allows for 1 character difference)
                matches = find_matching_di_words(
                    value,
                    di_result,
                    match_func=is_lower_value_within_levenstein_distance_1,
                )
            if not matches:
                # 3. If no exact word match was found, look for an exact line match
                matches = find_matching_di_lines(
                    value, di_result, match_func=is_lower_exact_match
                )
            if not matches:
                # 4. If not exact match was found, we will check for matches where the LLM output is part of the
                # doc intel content. This is useful for cases where Doc Intelligence extracts the date of birth as
                # a single word (e.g. 30/12/1985), but where the LLM separates the day, month and year into
                # separate fields.
                matches = find_matching_di_lines(
                    value,
                    di_result,
                    match_func=is_lower_value_in_lower_content_without_whitespace,
                )
            if not matches:
                # 5. If not exact match was found, we will check for matches where the LLM output is part of the
                # doc intel content. This is useful for cases where Doc Intelligence extracts the date of birth as
                # a single word (e.g. 30/12/1985), but where the LLM separates the day, month and year into
                # separate fields.
                matches = find_matching_di_lines(
                    value,
                    di_result,
                    match_func=is_lower_value_in_split_lower_content,
                )
            # Merge confidence score values in case there isn't a single value
            field_confidence_score = merge_confidence_scores(
                scores=[match.confidence for match in matches],
                no_values_replacement=0.0,  # If no matches, give confidence score of 0
                multiple_values_replacement_func=min,  # If multiple matches, take the minimum of all values
            )
            normalized_polygons = [
                content_obj.normalized_polygon for content_obj in matches
            ]
            result[field] = FieldWithConfidenceModel(
                value=value,
                doc_intel_content_matches_count=len(matches),
                confidence=field_confidence_score,
                normalized_polygons=normalized_polygons,
            )
            # Check the extracted value and its confidence score to determine if we need to escalate to human review.
            if value is None or value == "":
                is_any_field_missing = True
            if field_confidence_score is None or (
                min_field_confidence_score is not None
                and field_confidence_score < min_field_confidence_score
            ):
                min_field_confidence_score = field_confidence_score
        # Convert to structured Pydantic object and save to output
        output_model.result = ExtractedFieldsWithConfidenceModel(**result)
        output_model.min_extracted_field_confidence_score = min_field_confidence_score
        # Determine whether the result requires human review. This is determined by the confidence scores
        # of the extracted fields. If any field has a confidence score below a certain threshold, we will
        # require human review.
        output_model.requires_human_review = (
            is_any_field_missing
            or min_field_confidence_score is None
            or min_field_confidence_score < MIN_CONFIDENCE_SCORE
        )

        ### 7. Draw bounding boxes on the extracted fields
        # With the locations of the extracted fields now known, we can draw
        # bounding boxes on the image and to make it easier to digest the output.
        error_text = (
            "An error occurred when drawing bounding boxes on the extracted fields."
        )
        # Get the base64 image from the first page
        pil_img = doc_page_imgs[1]
        for _field_name, field_value in output_model.result.__dict__.items():
            for polygon in field_value.normalized_polygons:
                # Change from normalized scale to pixel-based scale
                pixel_based_polygon = scale_flat_poly_list(
                    polygon,
                    existing_scale=(1, 1),
                    new_scale=(pil_img.width, pil_img.height),
                )
                pixel_based_polygon_dict = flat_poly_list_to_poly_dict_list(
                    pixel_based_polygon
                )
                pil_img = draw_polygon_on_pil_img(
                    pil_img=pil_img,
                    polygon=pixel_based_polygon_dict,
                    outline_color="blue",
                    outline_width=3,
                )
        # Resize the image to reduce transfer size
        pil_img = resize_img_by_max(pil_img, max_height=1000, max_width=1000)
        output_model.result_img_with_bboxes = pil_img_to_base64(pil_img)

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
