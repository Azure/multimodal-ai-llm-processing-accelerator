import logging
import os
from base64 import b64encode
from copy import deepcopy
from enum import Enum
from typing import Optional

import azure.functions as func
import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.ai.textanalytics import PiiEntity, TextAnalyticsClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from fitz import Document, get_text_length
from pydantic import BaseModel, Field
from src.helpers.common import MeasureRunTime
from src.helpers.data_loading import load_pymupdf_pdf, pymupdf_pdf_page_to_img_pil
from src.helpers.image import pil_img_to_base64_bytes

logger = logging.getLogger(__name__)

load_dotenv()

aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

bp_pii_redaction = func.Blueprint()
TEXT_FUNCTION_ROUTE = "pii_redaction_text"
PDF_FUNCTION_ROUTE = "pii_redaction_pdf"


class PDFTextExtractionMethod(Enum):
    """Sets the option for extracting raw text from a PDF."""

    DOCUMENT_INTELLIGENCE = "DOCUMENT_INTELLIGENCE"
    PYMUPDF = "PYMUPDF"


# Set the text extraction method to be used for PDFs
# - PYMUPDF is runs locally and is free, but will only extract text that is embedded into the PDF in the data layer.
#   Any text embedded within images will not be extracted, and no text will be extracted when the PDF is a scanned image.
# - DOCUMENT_INTELLIGENCE uses Azure Document Intelligence to extract all text, including embedded and scanned text.
TEXT_EXTRACTION_METHOD = PDFTextExtractionMethod.DOCUMENT_INTELLIGENCE

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
LANGUAGE_ENDPOINT = os.getenv("LANGUAGE_ENDPOINT")

credential = DefaultAzureCredential()

# Create the clients for Document Intelligence and Azure OpenAI
text_analytics_client = TextAnalyticsClient(
    endpoint=LANGUAGE_ENDPOINT, credential=credential
)

# Set up the Document Intelligence v4.0 preview client. This will allow us to
# use the latest features of the Document Intelligence service. Check out the
# Document Intelligence Processor Walkthrough Notebook for more information
# (within the `notebooks` folder).
DOC_INTEL_MODEL_ID = "prebuilt-read"  # Set Document Intelligence model ID
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=credential,
    api_version="2024-07-31-preview",
)


# Setup Pydantic models for validation of the request and response
class TextFunctionRequestModel(BaseModel):
    """
    Defines the schema that will be expected in the request body. We'll use this to
    ensure that the request contains the correct values and structure, and to allow
    a partially filled request to be processed in case of an error.
    """

    text: Optional[str] = Field(description="The text to be summarized")


class TextFunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    success: bool = Field(
        False, description="Indicates whether the pipeline was successful."
    )
    error_text: Optional[str] = Field(
        None,
        description="If an error occurred, this field will contain the error message.",
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )
    redacted_text: Optional[str] = Field(None, description="The raw redacted text.")
    pii_raw_response: Optional[dict] = Field(
        None, description="The raw API response from PII recognition."
    )


def replace_text(text: str, replacements: dict) -> str:
    """
    Replace all occurrences of text in a text with replacement text.

    :param text:
        The text to be redacted.
    :param replacements:
        A dictionary of text to replace and the text to replace it with.
    :returns:
        The redacted text.
    """
    for find_text, replace_text in replacements.items():
        text = text.replace(find_text, replace_text)
    return text


@bp_pii_redaction.route(route=TEXT_FUNCTION_ROUTE)
def redact_pii_text(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(
        f"Python HTTP trigger function `{TEXT_FUNCTION_ROUTE}` received a request."
    )
    # Create the object to hold all intermediate and final values. We will progressively update
    # values as each stage of the pipeline is completed, allowing us to return a partial
    # response in case of an error at any stage.
    output_model = TextFunctionReponseModel(success=False)
    try:
        # Create error_text and error_code variables. These will be updated as
        # we move through the pipeline so that if a step fails, the vars reflect
        # what has failed. If all steps complete successfully, the vars are
        # never used.
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        ### 1. Check the request body
        req_body = req.get_json()
        try:
            request_obj = TextFunctionRequestModel(**req_body)
        except Exception as _e:
            raise ValueError(
                (
                    "The request body was not in the expected format. Please ensure that it is "
                    "a valid JSON object with the following fields: {}"
                ).format(list(TextFunctionRequestModel.model_fields.keys()))
            )

        ### 2. Create the messages to send to the LLM
        error_text = "An error occurred during PII recognition"
        error_code = 500

        documents = [request_obj.text]
        ### 3. Redact PII from the text using Azure AI Language service
        pii_result = text_analytics_client.recognize_pii_entities(
            documents=documents,
        )
        output_model.pii_raw_response = [str(doc_result) for doc_result in pii_result]
        pii_result_doc = pii_result[0]
        if pii_result_doc.is_error:
            raise RuntimeError(
                f"An error occurred during PII recognition: {pii_result_doc.error.message}",
            )

        ### 4. Replace the PII entities with '<<CATEGORY>>' text.
        # This gives us a more readable output than the redacted text from the raw API response (which simply replaces PII with asterixes).
        replacements = {
            entity.text: f"<<{entity.category}>>" for entity in pii_result_doc.entities
        }
        output_model.redacted_text = replace_text(request_obj.text, replacements)

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


class PDFFunctionReponseModel(BaseModel):
    """
    Defines the schema that will be returned by the function. We'll use this to
    ensure that the response contains the correct values and structure, and
    to allow a partially filled response to be returned in case of an error.
    """

    success: bool = Field(
        False, description="Indicates whether the pipeline was successful."
    )
    error_text: Optional[str] = Field(
        None,
        description="If an error occurred, this field will contain the error message.",
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )
    redacted_text: Optional[str] = Field(None, description="The raw redacted text.")
    redacted_pdf: Optional[str] = Field(
        None,
        description="The base64 encoded redacted PDF.",
    )
    input_doc_pdf_page_imgs: Optional[list[bytes]] = Field(
        None,
        description="The base64 encoded input PDF pages.",
    )
    redacted_pdf_page_imgs: Optional[list[bytes]] = Field(
        None,
        description="The base64 encoded redacted PDF pages.",
    )
    di_raw_response: Optional[dict] = Field(
        None, description="The raw API response from Document Intelligence."
    )
    pii_raw_response: Optional[dict] = Field(
        None, description="The raw API response from PII recognition."
    )
    merged_pii_entities: Optional[list[str]] = Field(
        None, description="The merged PII entities from the split documents."
    )


def replace_text_in_pdf(
    doc: Document, replacements: dict, inplace: bool = True
) -> Document:
    """
    Replace all occurrences of text in a PDF with replacement text. This function
    will only work for text that is embedded in the PDF in the data layer. Text
    embedded in images is not supported at this stage and would require the
    Azure Document Intelligence response (with page numbers and bounding boxes)
    to be used to redact the text based on it's location in the PDF.

    :param doc:
        The PDF to be redacted.
    :param replacements:
        A dictionary of text to replace and the text to replace it with.
    :param inplace:
        If True, the PDF will be redacted in-place and returned. If False, a new
            PDF will be returned.
    :returns:
        The redacted PDF.
    """

    # The code in this function has been copied and modified from
    # https://dev.to/abbazs/replace-text-in-pdfs-using-python-42k6

    if not inplace:
        doc = deepcopy(doc)

    for find_text, replace_text in replacements.items():
        occurence_found = False
        for page_num, page in enumerate(doc, start=1):
            # Search for occurrences of find_text
            instances = page.search_for(find_text)

            if not instances:
                continue

            occurence_found = True
            for rect in instances:
                # First, redact (remove) the original text
                page.add_redact_annot(rect)
                page.apply_redactions()

                # Default values for text properties
                font = "helv"  # Default to Helvetica
                font_size = 10.0  # Default size
                color = (255, 0, 0)  # Default to black

                # Calculate the max width of the text to be replaced
                max_width = rect.x1 - rect.x0

                # Calculate the baseline position for text insertion
                baseline = fitz.Point(
                    rect.x0, rect.y1 - 2.2
                )  # Adjust the -2 offset as needed

                candidate_text_length = get_text_length(
                    replace_text, fontname=font, fontsize=font_size
                )
                if candidate_text_length > max_width:
                    # If the replacement text is too wide to fit in the original location, reduce the font size
                    font_size = font_size * (max_width / candidate_text_length)
                elif candidate_text_length < max_width:
                    # if the text is too short, change the baseline position to center the text
                    offset = (max_width - candidate_text_length) / 2
                    baseline = fitz.Point(
                        rect.x0 + offset, rect.y1 - 2.2
                    )  # Adjust the -2 offset as needed

                # Normalize the color values to range 0 to 1
                normalized_color = (
                    tuple(c / 255 for c in color)
                    if isinstance(color, tuple)
                    else (0, 0, 0)
                )

                # Insert the new text at the adjusted position
                page.insert_text(
                    baseline,
                    replace_text,
                    fontsize=font_size,
                    fontname=font,
                    color=normalized_color,
                )

        if not occurence_found:
            logging.warning(
                (
                    f"No occurrences of '{find_text}' found in PyMuPDF document. "
                    "This text could be embedded in an image which is not yet supported by this function."
                )
            )

    return doc


def match_pii_entities_to_di_bounding_boxes(
    pii_entities: list[PiiEntity],
    di_response: AnalyzeResult,
) -> list[PiiEntity]:
    """
    Match the identified PII entities to their associated bounding boxes by
    comparing the text of the PII entity to the words and lines in the Document
    Intelligence response.
    """
    # Match PII entities to Document Intelligence words and lines
    for entity in pii_entities:
        for page in di_response.pages:
            for line in page.lines:
                if entity.text in line.content:
                    entity.bounding_boxes = line.bounding_regions
    return pii_entities


def replace_text_in_pdf_with_bounding_boxes(
    doc: Document, replacements: dict, inplace: bool = True
) -> Document:
    """
    Replace all occurrences of text in a PDF with replacement text. This function
    will only work for text that is embedded in the PDF in the data layer. Text
    embedded in images is not supported at this stage and would require the
    Azure Document Intelligence response (with page numbers and bounding boxes)
    to be used to redact the text based on it's location in the PDF.

    :param doc:
        The PDF to be redacted.
    :param replacements:
        A dictionary of text to replace and the text to replace it with.
    :param inplace:
        If True, the PDF will be redacted in-place and returned. If False, a new
            PDF will be returned.
    :returns:
        The redacted PDF.
    """

    # The code in this function has been copied and modified from
    # https://dev.to/abbazs/replace-text-in-pdfs-using-python-42k6

    if not inplace:
        doc = deepcopy(doc)

    for find_text, replace_text in replacements.items():
        occurence_found = False
        for page_num, page in enumerate(doc, start=1):
            # Search for occurrences of find_text
            instances = page.search_for(find_text)

            if not instances:
                continue

            occurence_found = True
            for rect in instances:
                # First, redact (remove) the original text
                page.add_redact_annot(rect)
                page.apply_redactions()

                # Default values for text properties
                font = "helv"  # Default to Helvetica
                font_size = 10.0  # Default size
                color = (255, 0, 0)  # Default to black

                # Calculate the max width of the text to be replaced
                max_width = rect.x1 - rect.x0

                # Calculate the baseline position for text insertion
                baseline = fitz.Point(
                    rect.x0, rect.y1 - 2.2
                )  # Adjust the -2 offset as needed

                candidate_text_length = get_text_length(
                    replace_text, fontname=font, fontsize=font_size
                )
                if candidate_text_length > max_width:
                    # If the replacement text is too wide to fit in the original location, reduce the font size
                    font_size = font_size * (max_width / candidate_text_length)
                elif candidate_text_length < max_width:
                    # if the text is too short, change the baseline position to center the text
                    offset = (max_width - candidate_text_length) / 2
                    baseline = fitz.Point(
                        rect.x0 + offset, rect.y1 - 2.2
                    )  # Adjust the -2 offset as needed

                # Normalize the color values to range 0 to 1
                normalized_color = (
                    tuple(c / 255 for c in color)
                    if isinstance(color, tuple)
                    else (0, 0, 0)
                )

                # Insert the new text at the adjusted position
                page.insert_text(
                    baseline,
                    replace_text,
                    fontsize=font_size,
                    fontname=font,
                    color=normalized_color,
                )

        if not occurence_found:
            logging.warning(
                (
                    f"No occurrences of '{find_text}' found in PyMuPDF document. "
                    "This text could be embedded in an image which is not yet supported by this function."
                )
            )

    return doc


@bp_pii_redaction.route(route=PDF_FUNCTION_ROUTE)
def redact_pii_pdf(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(
        f"Python HTTP trigger function `{PDF_FUNCTION_ROUTE}` received a request."
    )
    output_model = PDFFunctionReponseModel(success=False)
    try:
        # Create error_text and error_code variables. These will be updated as
        # we move through the pipeline so that if a step fails, the vars reflect
        # what has failed. If all steps complete successfully, the vars are
        # never used.
        error_text = "An error occurred while reading the PDF file"
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        ### 1. Check the request body
        # Check mime_type of the request data
        mime_type = req.headers.get("Content-Type")
        if mime_type != "application/pdf":
            return func.HttpResponse(
                "This function only supports a Content-Type of 'application/pdf'. Supplied file is of type {}".format(
                    mime_type
                ),
                status_code=error_code,
            )

        req_body = req.get_body()
        if len(req_body) == 0:
            return func.HttpResponse(
                "Please provide a base64 encoded PDF in the request body.",
                status_code=error_code,
            )
        pdf = load_pymupdf_pdf(pdf_bytes=req_body)

        ### 2. Extract the text and images using Document Intelligence
        error_text = "An error occurred during text & image extraction."
        error_code = 500

        input_doc_pdf_page_imgs = [
            pil_img_to_base64_bytes(pymupdf_pdf_page_to_img_pil(page, 80, 0))
            for page in pdf
        ]
        output_model.input_doc_pdf_page_imgs = input_doc_pdf_page_imgs
        # Extract the text from the PDF
        if TEXT_EXTRACTION_METHOD is PDFTextExtractionMethod.PYMUPDF:
            raw_text = "\f".join(
                [page.get_text(page_num) for page_num, page in enumerate(pdf, start=1)]
            )
        elif TEXT_EXTRACTION_METHOD is PDFTextExtractionMethod.DOCUMENT_INTELLIGENCE:
            poller = di_client.begin_analyze_document(
                model_id=DOC_INTEL_MODEL_ID,
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
            output_model.di_raw_response = di_result.as_dict()
            raw_text = di_result.content
        else:
            raise ValueError(
                f"Invalid text extraction method: {TEXT_EXTRACTION_METHOD}"
            )

        ### 3. Redact PII from the text using Azure AI Language service
        error_text = "An error occurred during PII recognition."

        # Split the document into chunks of 5120 characters or less (the max length for the PII recognition API)
        split_delimiter = "\n"
        paragraphs = raw_text.split(split_delimiter)
        split_documents = [""]
        for paragraph in paragraphs:
            para_length = len(paragraph)
            if para_length > 5120:
                raise ValueError("A paragraph is longer than 5120 characters")
            elif para_length + len(split_delimiter) + len(split_documents[-1]) < 5120:
                split_documents[-1] = (
                    f"{split_documents[-1]}{split_delimiter}{paragraph}"
                )
            else:
                split_documents.append(paragraph)
        # Record the correct starting offset (based on Document Intelligence) for each document chunk
        split_documents_offsets = [0]
        for doc in split_documents[:-1]:
            split_documents_offsets.append(
                len(doc) + len(split_delimiter) + sum(split_documents_offsets)
            )

        # Process the documents
        pii_result = text_analytics_client.recognize_pii_entities(
            documents=split_documents,
        )
        output_model.pii_raw_response = [str(doc_result) for doc_result in pii_result]
        if any(doc_result.is_error for doc_result in pii_result):
            raise Exception("An error occurred during PII recognition")
        # Combine the separated documents back together, adjusting the index of the PII recognition results to match the original document
        merged_pii_entities: list[PiiEntity] = []
        for doc_idx, doc_result in enumerate(pii_result):
            for entity in doc_result.entities:
                entity.offset = entity.offset + split_documents_offsets[doc_idx]
                merged_pii_entities.append(entity)

        output_model.merged_pii_entities = [
            str(entity) for entity in merged_pii_entities
        ]

        ### 3. Replace the PII entities with '<<CATEGORY>>' text.
        # This gives us a more readable output than the redacted text from the raw API response (which simply replaces PII with asterixes).
        replacements = {
            entity.text: f"<<{entity.category}>>" for entity in merged_pii_entities
        }
        output_model.redacted_text = replace_text(raw_text, replacements)
        redacted_pdf = replace_text_in_pdf(pdf, replacements)
        output_model.redacted_pdf = b64encode(redacted_pdf.tobytes()).decode("utf-8")
        redacted_pdf_page_imgs = [
            pil_img_to_base64_bytes(pymupdf_pdf_page_to_img_pil(page, 80, 0))
            for page in redacted_pdf
        ]
        output_model.redacted_pdf_page_imgs = redacted_pdf_page_imgs

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
