# This script creates the analyzers for the Azure Content Understanding service.

import json
import logging
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from src.components.content_understanding_client import (
    AzureContentUnderstandingClient,
    create_analyzers,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

# Load environment variables
CONTENT_UNDERSTANDING_ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")

if not CONTENT_UNDERSTANDING_ENDPOINT:
    raise ValueError(
        (
            "CONTENT_UNDERSTANDING_ENDPOINT must be set in in the environment variables. "
            "To set CONTENT_UNDERSTANDING_ENDPOINT based on the currently selected and deployed bicep environment, "
            "copy and run the code from the postprovision hook in the azure.yaml file."
        )
    )

# If existing analyzers have had their schemas changed and they need to be
# recreated, set this to True.
FORCE_ANALYZER_RECREATION = True


def create_config_analyzer_schemas(force_recreation: bool = False):
    """
    Create the schemas for the Content Understanding analyzers.

    :param force_recreation: If True, the existing analyzers will be recreated
        even if it already exists. This is useful when a schema needs to be
        updated.
    :type force_recreation: bool
    """
    # Load existing analyzer schemas
    config_path = os.path.join(
        os.path.dirname(__file__), "config/content_understanding_schemas.json"
    )
    with open(config_path, "r") as f:
        CONTENT_UNDERSTANDING_SCHEMAS: dict[str, dict[str, dict]] = json.load(f)

    # Create analyzers for any missing schemas
    analyzer_to_schema_mapper = list()
    for _modality, analyzer_schemas in CONTENT_UNDERSTANDING_SCHEMAS.items():
        for analyzer_id, schema in analyzer_schemas.items():
            analyzer_to_schema_mapper.append((analyzer_id, schema))

    cu_client = AzureContentUnderstandingClient(
        endpoint=CONTENT_UNDERSTANDING_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version="2024-12-01-preview",
        enable_face_identification=False,
    )

    _cu_analyzer_ids = create_analyzers(
        cu_client, analyzer_to_schema_mapper, force_recreation=force_recreation
    )


if __name__ == "__main__":
    create_config_analyzer_schemas(FORCE_ANALYZER_RECREATION)
