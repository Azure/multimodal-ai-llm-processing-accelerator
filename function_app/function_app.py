import logging
import os

import azure.functions as func
from dotenv import load_dotenv
from src.helpers.azure_function import (
    check_if_azurite_storage_emulator_is_running,
    check_if_env_var_is_set,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Reduce Azure SDK logging level
_logger = logging.getLogger("azure")
_logger.setLevel(logging.WARNING)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

### Read environment variables to determine which backend resources/services are deployed
IS_CONTENT_UNDERSTANDING_DEPLOYED = check_if_env_var_is_set(
    "CONTENT_UNDERSTANDING_ENDPOINT"
)
IS_AOAI_DEPLOYED = check_if_env_var_is_set("AOAI_ENDPOINT")
IS_DOC_INTEL_DEPLOYED = check_if_env_var_is_set("DOC_INTEL_ENDPOINT")
IS_SPEECH_DEPLOYED = check_if_env_var_is_set("SPEECH_ENDPOINT")
IS_LANGUAGE_DEPLOYED = check_if_env_var_is_set("LANGUAGE_ENDPOINT")
IS_STORAGE_ACCOUNT_AVAILABLE = (
    # If running the function app locally, check if the Azurite storage emulator is running
    os.getenv("AzureWebJobsStorage") == "UseDevelopmentStorage=true"
    and check_if_azurite_storage_emulator_is_running()
) or all(
    # If running on Azure, check if the storage account env vars were set correctly
    [
        check_if_env_var_is_set("AzureWebJobsStorage__accountName"),
        check_if_env_var_is_set("AzureWebJobsStorage__blobServiceUri"),
        check_if_env_var_is_set("AzureWebJobsStorage__queueServiceUri"),
        check_if_env_var_is_set("AzureWebJobsStorage__tableServiceUri"),
    ]
)
IS_COSMOSDB_AVAILABLE = check_if_env_var_is_set(
    "COSMOSDB_DATABASE_NAME"
) and check_if_env_var_is_set("CosmosDbConnectionSetting__accountEndpoint")

### Register blueprints for HTTP functions, provided the relevant backend AI services are deployed
### and the relevant environment variables are set
if IS_AOAI_DEPLOYED:
    from bp_summarize_text import bp_summarize_text

    app.register_blueprint(bp_summarize_text)
if IS_DOC_INTEL_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_doc_intel_extract_city_names import bp_doc_intel_extract_city_names
    from bp_form_extraction_with_confidence import bp_form_extraction_with_confidence

    app.register_blueprint(bp_doc_intel_extract_city_names)
    app.register_blueprint(bp_form_extraction_with_confidence)
if IS_SPEECH_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_call_center_audio_analysis import bp_call_center_audio_analysis

    app.register_blueprint(bp_call_center_audio_analysis)
if IS_DOC_INTEL_DEPLOYED:
    from bp_multimodal_doc_intel_processing import bp_multimodal_doc_intel_processing

    app.register_blueprint(bp_multimodal_doc_intel_processing)
if IS_CONTENT_UNDERSTANDING_DEPLOYED:
    from bp_content_understanding_audio import bp_content_understanding_audio
    from bp_content_understanding_document import bp_content_understanding_document
    from bp_content_understanding_image import bp_content_understanding_image
    from bp_content_understanding_video import bp_content_understanding_video

    app.register_blueprint(bp_content_understanding_document)
    app.register_blueprint(bp_content_understanding_video)
    app.register_blueprint(bp_content_understanding_audio)
    app.register_blueprint(bp_content_understanding_image)
if IS_LANGUAGE_DEPLOYED:
    from bp_pii_redaction import bp_pii_redaction

    app.register_blueprint(bp_pii_redaction)


### Define functions with input/output binding decorators (these do not work when defined in blueprint files).

## Blob storage -> CosmosDB Document Processing Pipeline
# Only register the function if Azure Storage and CosmosDB information is available and can be connected to
if (
    IS_STORAGE_ACCOUNT_AVAILABLE
    and IS_COSMOSDB_AVAILABLE
    and IS_AOAI_DEPLOYED
    and IS_DOC_INTEL_DEPLOYED
):
    from extract_blob_field_info_to_cosmosdb import (
        get_structured_extraction_func_outputs,
    )

    COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")

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
