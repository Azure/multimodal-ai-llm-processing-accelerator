# This code is copied directly from a private Azure Content Understanding Examples repository: https://github.com/Azure-Samples/azure-ai-content-understanding-samples

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from requests.models import Response


class AzureContentUnderstandingClient:
    def __init__(
        self,
        endpoint: str,
        api_version: str,
        azure_ad_token_provider: Optional[Callable] = None,
        subscription_key: Optional[str] = None,
        api_token: Optional[str] = None,
        enable_face_identification: bool = False,
    ):
        if not any([subscription_key, api_token, azure_ad_token_provider]):
            raise ValueError(
                "One of azure_ad_token_provider, subscription_key or api_token must be provided."
            )
        if not api_version:
            raise ValueError("API version must be provided.")
        if not endpoint:
            raise ValueError("Endpoint must be provided.")

        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._logger = logging.getLogger(__name__)
        self._subscription_key = subscription_key
        self._api_token = api_token
        self._azure_ad_token_provider = azure_ad_token_provider
        self._enable_face_identification = enable_face_identification

    def _get_analyzer_url(self, endpoint, api_version, analyzer_id):
        return f"{endpoint}/contentunderstanding/analyzers/{analyzer_id}?api-version={api_version}"  # noqa

    def _get_analyzer_list_url(self, endpoint, api_version):
        return f"{endpoint}/contentunderstanding/analyzers?api-version={api_version}"

    def _get_analyze_url(self, endpoint, api_version, analyzer_id):
        return f"{endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyze?api-version={api_version}"  # noqa

    def _get_training_data_config(
        self, storage_container_sas_url, storage_container_path_prefix
    ):
        return {
            "containerUrl": storage_container_sas_url,
            "kind": "blob",
            "prefix": storage_container_path_prefix,
        }

    def _get_headers(self):
        """Returns the headers for the HTTP requests.

        Returns:
            dict: A dictionary containing the headers for the HTTP requests.
        """
        headers = (
            {"Authorization": f"Bearer {self._azure_ad_token_provider()}"}
            if self._azure_ad_token_provider
            else (
                {"Ocp-Apim-Subscription-Key": self._subscription_key}
                if self._subscription_key
                else {"Authorization": f"Bearer {self._api_token}"}
            )
        )
        headers["x-ms-useragent"] = "cu-sample-code"
        if self._enable_face_identification:
            headers["cogsvc-videoanalysis-face-identification-enable"] = "true"
        return headers

    def get_all_analyzers(self):
        """
        Retrieves a list of all available analyzers from the content understanding service.

        This method sends a GET request to the service endpoint to fetch the list of analyzers.
        It raises an HTTPError if the request fails.

        Returns:
            dict: A dictionary containing the JSON response from the service, which includes
                  the list of available analyzers.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
        """
        response = requests.get(
            url=self._get_analyzer_list_url(self._endpoint, self._api_version),
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    def get_analyzer_detail_by_id(self, analyzer_id):
        """
        Retrieves a specific analyzer detail through analyzerid from the content understanding service.
        This method sends a GET request to the service endpoint to get the analyzer detail.

        Args:
            analyzer_id (str): The unique identifier for the analyzer.

        Returns:
            dict: A dictionary containing the JSON response from the service, which includes the target analyzer detail.

        Raises:
            HTTPError: If the request fails.
        """
        response = requests.get(
            url=self._get_analyzer_url(self._endpoint, self._api_version, analyzer_id),
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    def begin_create_analyzer(
        self,
        analyzer_id: str,
        analyzer_schema: dict = None,
        analyzer_schema_path: str = "",
        training_storage_container_sas_url: str = "",
        training_storage_container_path_prefix: str = "",
    ):
        """
        Initiates the creation of an analyzer with the given ID and schema.

        Args:
            analyzer_id (str): The unique identifier for the analyzer.
            analyzer_schema (dict, optional): The schema definition for the analyzer. Defaults to None.
            analyzer_schema_path (str, optional): The file path to the analyzer schema JSON file. Defaults to "".
            training_storage_container_sas_url (str, optional): The SAS URL for the training storage container. Defaults to "".
            training_storage_container_path_prefix (str, optional): The path prefix within the training storage container. Defaults to "".

        Raises:
            ValueError: If neither `analyzer_schema` nor `analyzer_schema_path` is provided.
            requests.exceptions.HTTPError: If the HTTP request to create the analyzer fails.

        Returns:
            requests.Response: The response object from the HTTP request.
        """
        if analyzer_schema_path and Path(analyzer_schema_path).exists():
            with open(analyzer_schema_path, "r") as file:
                analyzer_schema = json.load(file)

        if not analyzer_schema:
            raise ValueError("Analyzer schema must be provided.")

        if (
            training_storage_container_sas_url
            and training_storage_container_path_prefix
        ):  # noqa
            analyzer_schema["trainingData"] = self._get_training_data_config(
                training_storage_container_sas_url,
                training_storage_container_path_prefix,
            )

        headers = {"Content-Type": "application/json"}
        headers.update(self._get_headers())

        response = requests.put(
            url=self._get_analyzer_url(self._endpoint, self._api_version, analyzer_id),
            headers=headers,
            json=analyzer_schema,
        )
        response.raise_for_status()
        self._logger.info(f"Analyzer {analyzer_id} create request accepted.")
        return response

    def delete_analyzer(self, analyzer_id: str):
        """
        Deletes an analyzer with the specified analyzer ID.

        Args:
            analyzer_id (str): The ID of the analyzer to be deleted.

        Returns:
            response: The response object from the delete request.

        Raises:
            HTTPError: If the delete request fails.
        """
        response = requests.delete(
            url=self._get_analyzer_url(self._endpoint, self._api_version, analyzer_id),
            headers=self._get_headers(),
        )
        response.raise_for_status()
        self._logger.info(f"Analyzer {analyzer_id} deleted.")
        return response

    def begin_analyze(
        self,
        analyzer_id: str,
        file_location: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
    ):
        """
        Begins the analysis of a file or URL using the specified analyzer.

        Args:
            analyzer_id (str): The ID of the analyzer to use.
            file_location (str, optional: The path to the file or the URL to analyze. Defaults to None.
            file_bytes (bytes, optional): The bytes of the file to analyze. Defaults to None.

        Returns:
            Response: The response from the analysis request.

        Raises:
            ValueError: If the file location is not a valid path or URL.
            HTTPError: If the HTTP request returned an unsuccessful status code.
        """
        data = None
        if file_location and file_bytes:
            raise ValueError(
                "Only one of file_location or file_bytes should be provided."
            )
        if not file_location and not file_bytes:
            raise ValueError("Either file_location or file_bytes must be provided.")
        if file_bytes:
            data = file_bytes
            headers = {"Content-Type": "application/octet-stream"}
        elif Path(file_location).exists():
            with open(file_location, "rb") as file:
                data = file.read()
            headers = {"Content-Type": "application/octet-stream"}
        elif "https://" in file_location or "http://" in file_location:
            data = {"url": file_location}
            headers = {"Content-Type": "application/json"}
        else:
            raise ValueError("file_location must be a valid path or URL.")

        headers.update(self._get_headers())
        if isinstance(data, dict):
            response = requests.post(
                url=self._get_analyze_url(
                    self._endpoint, self._api_version, analyzer_id
                ),
                headers=headers,
                json=data,
            )
        else:
            response = requests.post(
                url=self._get_analyze_url(
                    self._endpoint, self._api_version, analyzer_id
                ),
                headers=headers,
                data=data,
            )

        response.raise_for_status()
        self._logger.info(
            f"Analyzing file {file_location or ''} with analyzer: {analyzer_id}"
        )
        return response

    def get_image_from_analyze_operation(
        self, analyze_response: Response, image_id: str
    ):
        """Retrieves an image from the analyze operation using the image ID.
        Args:
            analyze_response (Response): The response object from the analyze operation.
            image_id (str): The ID of the image to retrieve.
        Returns:
            bytes: The image content as a byte string.
        """
        operation_location = analyze_response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError(
                "Operation location not found in the analyzer response header."
            )
        operation_location = operation_location.split("?api-version")[0]
        image_retrieval_url = (
            f"{operation_location}/images/{image_id}?api-version={self._api_version}"
        )
        try:
            response = requests.get(
                url=image_retrieval_url, headers=self._get_headers()
            )
            response.raise_for_status()

            assert response.headers.get("Content-Type") == "image/jpeg"

            return response.content
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None

    def poll_result(
        self,
        response: Response,
        timeout_seconds: int = 120,
        polling_interval_seconds: int = 2,
    ) -> dict:
        """
        Polls the result of an asynchronous operation until it completes or times out.

        Args:
            response (Response): The initial response object containing the operation location.
            timeout_seconds (int, optional): The maximum number of seconds to wait for the operation to complete. Defaults to 120.
            polling_interval_seconds (int, optional): The number of seconds to wait between polling attempts. Defaults to 2.

        Raises:
            ValueError: If the operation location is not found in the response headers.
            TimeoutError: If the operation does not complete within the specified timeout.
            RuntimeError: If the operation fails.

        Returns:
            dict: The JSON response of the completed operation if it succeeds.
        """
        operation_location = response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError("Operation location not found in response headers.")

        headers = {"Content-Type": "application/json"}
        headers.update(self._get_headers())

        start_time = time.time()
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(
                    f"Operation timed out after {timeout_seconds:.2f} seconds."
                )

            response = requests.get(operation_location, headers=self._get_headers())
            response.raise_for_status()
            status = response.json().get("status").lower()
            if status == "succeeded":
                self._logger.info(
                    f"Request result is ready after {elapsed_time:.2f} seconds."
                )
                return response.json()
            elif status == "failed":
                self._logger.error(f"Request failed. Reason: {response.json()}")
                raise RuntimeError("Request failed.")
            else:
                self._logger.info(f"Request {operation_location} in progress ...")
            time.sleep(polling_interval_seconds)


def get_existing_analyzer_ids(cu_client: AzureContentUnderstandingClient) -> list[str]:
    """
    Gets a list of existing Content Understanding analyzer IDs.

    :param cu_client: A Content Understanding client.
    :type cu_client: AzureContentUnderstandingClient
    :return: A list of existing Content Understanding analyzer IDs.
    :rtype: list[str]
    """
    existing_cu_analyzer_ids = [
        analyzer["analyzerId"]
        for analyzer in cu_client.get_all_analyzers().get("value", [])
    ]
    return existing_cu_analyzer_ids


def create_analyzers(
    cu_client: AzureContentUnderstandingClient,
    analyzer_id_to_schema_mapper: dict,
    force_recreation: bool = False,
) -> list[str]:
    """
    Creates Content Understanding analyzers for any missing schemas.

    :param cu_client: Content Understanding client.
    :type cu_client: AzureContentUnderstandingClient
    :param analyzer_id_to_schema_mapper: A dictionary mapping analyzer IDs to
        their schemas.
    :type analyzer_id_to_schema_mapper: dict
    :param force_recreation: If True, will recreate all analyzers even if they
        already exist, defaults to False
    :type force_recreation: bool, optional
    :return: A list of existing Content Understanding analyzer IDs within the
        AI services resource.
    :rtype: list[str]
    """
    # Get list of existing CU analyzers
    existing_cu_analyzer_ids = get_existing_analyzer_ids(cu_client)
    created_analyzer_ids = list()
    for analyzer_id, schema in analyzer_id_to_schema_mapper:
        # Optionally delete the analyzer to force recreation of it (in case
        # the schema has changed)
        if force_recreation and analyzer_id in existing_cu_analyzer_ids:
            _delete_analyzer_response = cu_client.delete_analyzer(
                analyzer_id=analyzer_id,
            )
            if _delete_analyzer_response.status_code == 204:
                existing_cu_analyzer_ids.remove(analyzer_id)
            time.sleep(0.5)

        # Create the analyzer if it doesn't exist
        if analyzer_id not in existing_cu_analyzer_ids:
            create_analyzer_response = cu_client.begin_create_analyzer(
                analyzer_id=analyzer_id,
                analyzer_schema=schema,
            )
            _result = cu_client.poll_result(create_analyzer_response)
            if _result.get("status") == "Succeeded":
                logging.info(f"Analyzer {analyzer_id} created successfully.")
                created_analyzer_ids.append(analyzer_id)
    if created_analyzer_ids:
        logging.info(
            f"Newly Created Content Understanding Analyzers: {created_analyzer_ids}"
        )
    else:
        logging.info("No new Content Understanding Analyzers created.")
    return get_existing_analyzer_ids(cu_client)
