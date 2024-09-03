import json
import logging
import os

import azure.functions as func
from dotenv import load_dotenv
from haystack.components.converters import AzureOCRDocumentConverter
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from src.components.doc_intelligence import VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES
from src.helpers.common import MeasureRunTime, haystack_doc_to_string

logger = logging.getLogger(__name__)

load_dotenv()

bp_doc_intel_only = func.Blueprint()


DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_API_KEY = Secret.from_env_var("DOC_INTEL_API_KEY")

di_converter = AzureOCRDocumentConverter(
    endpoint=DOC_INTEL_ENDPOINT,
    api_key=DOC_INTEL_API_KEY,
    model_id="prebuilt-read",
)


@bp_doc_intel_only.route(route="doc_intel_only")
def doc_intel_only(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    mime_type = req.headers.get("Content-Type")
    if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
        return func.HttpResponse(
            "This function only supports a Content-Type of {}. Supplied file is of type {}".format(
                ", ".join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES), mime_type
            ),
            status_code=400,
        )

    req_body = req.get_body()

    if len(req_body) == 0:
        return func.HttpResponse(
            "Please provide a base64 encoded PDF in the request body.", status_code=400
        )
    try:
        bytestream = ByteStream(data=req_body, mime_type=mime_type)
        with MeasureRunTime() as di_timer:
            di_result = di_converter.run(sources=[bytestream])
        di_documents = di_result["documents"]
        # Return joined text content
        output = {
            "di_extracted_text": (
                "\n".join([haystack_doc_to_string(doc) for doc in di_documents])
                if di_documents
                else "No text was extracted from the document."
            ),
            "raw_di_response": di_result["raw_azure_response"],
            "di_time_taken_secs": di_timer.time_taken,
        }
    except Exception as e:
        logging.exception(e)
        return func.HttpResponse(
            "An error occurred while processing the document.", status_code=500
        )
    return func.HttpResponse(body=json.dumps(output), mimetype="application/json")
