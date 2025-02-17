import logging
import os

import azure.functions as func
from bp_call_center_audio_analysis import bp_call_center_audio_analysis
from bp_content_understanding_audio import bp_content_understanding_audio
from bp_content_understanding_document import bp_content_understanding_document
from bp_content_understanding_image import bp_content_understanding_image
from bp_content_understanding_video import bp_content_understanding_video
from bp_doc_intel_extract_city_names import bp_doc_intel_extract_city_names
from bp_form_extraction_with_confidence import bp_form_extraction_with_confidence
from bp_multimodal_doc_intel_processing import bp_multimodal_doc_intel_processing
from bp_pii_redaction import bp_pii_redaction
from bp_summarize_text import bp_summarize_text
from dotenv import load_dotenv
from extract_blob_field_info_to_cosmosdb import get_structured_extraction_func_outputs

load_dotenv()

COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")

# Reduce Azure SDK logging level
_logger = logging.getLogger("azure")
_logger.setLevel(logging.WARNING)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


### Register blueprints for HTTP functions
app.register_blueprint(bp_form_extraction_with_confidence)
app.register_blueprint(bp_call_center_audio_analysis)
app.register_blueprint(bp_summarize_text)
app.register_blueprint(bp_doc_intel_extract_city_names)
app.register_blueprint(bp_multimodal_doc_intel_processing)
app.register_blueprint(bp_content_understanding_document)
app.register_blueprint(bp_content_understanding_video)
app.register_blueprint(bp_content_understanding_audio)
app.register_blueprint(bp_content_understanding_image)
app.register_blueprint(bp_pii_redaction)


### Define functions with input/output binding decorators (these do not work when defined in blueprint files).
@app.function_name("blob_form_extraction_to_cosmosdb")
@app.blob_trigger(
    arg_name="inputblob",
    path="blob-form-to-cosmosdb-blobs/{name}",  # Triggered by any blobs created in this container
    connection="AzureWebJobsStorage",
)
@app.cosmos_db_output(
    arg_name="outputdocument",
    connection="CosmosDbConnectionSetting",
    database_name=COSMOSDB_DATABASE_NAME,
    container_name="blob-form-to-cosmosdb-container",
)
def extract_blob_pdf_fields_to_cosmosdb(
    inputblob: func.InputStream, outputdocument: func.Out[func.Document]
):
    """
    Extracts field information from a PDF and writes the extracted information
    to CosmosDB.

    :param inputblob: The input blob to process.
    :type inputblob: func.InputStream
    :param outputdocument: The output document to write to CosmosDB.
    :type outputdocument: func.Out[func.Document]
    """
    output_result = get_structured_extraction_func_outputs(inputblob)
    outputdocument.set(func.Document.from_dict(output_result))
