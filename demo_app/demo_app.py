import base64
import json
import logging
import mimetypes
import os
import time
from io import BytesIO
from typing import Optional, Union

import fitz
import gradio as gr
import requests
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Reduce Azure SDK logging level
_logger = logging.getLogger("azure.core")
_logger.setLevel(logging.WARNING)

load_dotenv()

# Get function endpoint from env vars. If not set, default to local host.
# Note: This assumes the function is being run prior to launching the web app.
FUNCTION_HOSTNAME = os.getenv("FUNCTION_HOSTNAME", "http://localhost:7071/")
FUNCTION_KEY = os.getenv("FUNCTION_KEY", "")
FUNCTION_ENDPOINT = os.path.join(FUNCTION_HOSTNAME, "api")

# Create clients for Azure services. If connecting to Azure resources, ensure
# the current Azure identity has the necessary permissions to access the
# storage & cosmosDB accounts.
# You can set the `additionalRoleAssignmentIdentityIds` property when deploying
# the solution to grant the necessary permissions to the logged-in user. See
# the README for more info.
USE_LOCAL_STORAGE_EMULATOR = os.getenv("USE_LOCAL_STORAGE_EMULATOR", "false") == "true"
LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING = os.getenv(
    "LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING"
)
STORAGE_ACCOUNT_ENDPOINT = os.getenv("STORAGE_ACCOUNT_ENDPOINT")
COSMOSDB_ACCOUNT_ENDPOINT = os.getenv("COSMOSDB_ACCOUNT_ENDPOINT")
COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")

credential = DefaultAzureCredential()

if USE_LOCAL_STORAGE_EMULATOR:
    # Connect to the local development storage account emulator using a connection string
    logging.info("Connecting to the local development storage account emulator.")
    blob_service_client = BlobServiceClient.from_connection_string(
        LOCAL_STORAGE_ACCOUNT_CONNECTION_STRING
    )
else:
    # Connect to the Azure storage account using identity auth
    logging.info("Connecting to the Azure storage account.")
    blob_service_client = BlobServiceClient(
        account_url=STORAGE_ACCOUNT_ENDPOINT, credential=credential
    )
# Connect to CosmosDB using identity auth (this requires a role assignment for
# the current identity)
cosmos_client = CosmosClient(url=COSMOSDB_ACCOUNT_ENDPOINT, credential=credential)

# Set authentication values based on the environment variables, defaulting to auth enabled
WEB_APP_USE_PASSWORD_AUTH = (
    os.getenv("WEB_APP_USE_PASSWORD_AUTH", "true").lower() == "true"
)
if WEB_APP_USE_PASSWORD_AUTH:
    try:
        WEB_APP_USERNAME = os.getenv("WEB_APP_USERNAME")
        WEB_APP_PASSWORD = os.getenv("WEB_APP_PASSWORD")
    except KeyError:
        raise RuntimeError(
            "`WEB_APP_USERNAME` and `WEB_APP_PASSWORD` must be set unless `WEB_APP_USE_PASSWORD_AUTH` is set to false."
        )
    auth_message = "Contact the site owner for the password"
    auth = (WEB_APP_USERNAME, WEB_APP_PASSWORD)
else:
    auth_message = None
    auth = None

# Get list of various demo files
DEMO_CALL_CENTER_AUDIO_FILES = [
    os.path.join("demo_files/call_center_audio", fn)
    for fn in os.listdir("demo_files/call_center_audio")
    if fn.endswith((".wav"))
]
DEMO_PDF_FILES = [
    os.path.join("demo_files", fn)
    for fn in os.listdir("demo_files")
    if fn.endswith((".pdf"))
]
DEMO_IMG_FILES = [
    os.path.join("demo_files", fn)
    for fn in os.listdir("demo_files")
    if fn.endswith((".png", ".jpg", ".jpeg"))
]
DEMO_ACCOUNT_OPENING_FORM_FILES = [
    os.path.join("demo_files/account_opening_forms", fn)
    for fn in os.listdir("demo_files/account_opening_forms")
    if fn.endswith((".pdf"))
]
DEMO_VISION_FILES = DEMO_PDF_FILES + DEMO_IMG_FILES
with open("demo_files/text_samples.json", "r") as json_file:
    TEXT_SAMPLES: dict[str, str] = json.load(json_file)

FILE_EXAMPLES_LABEL = "You can upload your own file, or select one of the following examples and then click 'Process File'."
AUDIO_EXAMPLES_LABEL = FILE_EXAMPLES_LABEL.replace("File", "Audio").replace(
    "upload", "record or upload"
)
TEXT_EXAMPLES_LABEL = FILE_EXAMPLES_LABEL.replace("File", "Text").replace(
    "upload your own file", "insert your own text"
)


def send_request(
    route: str,
    json_data: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
    headers: Optional[dict] = None,
    force_json_content_type: bool = True,
) -> tuple[int, float, Union[str, dict]]:
    """Sends a POST request to the specified route on the function host."""
    if FUNCTION_KEY:
        headers = headers or {}
        headers["x-functions-key"] = FUNCTION_KEY
    start_time = time.time()
    response = requests.post(
        os.path.join(FUNCTION_ENDPOINT, route),
        json=json_data,
        data=data,
        files=files,
        headers=headers,
    )
    client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
    # Check if the response is valid JSON
    if response.headers.get("Content-Type") == "application/json":
        response_content = response.json()
    else:
        response_content = response.content.decode("utf-8")
    # If the output component expects JSON, ensure the output is JSON.
    if response.status_code != 200:
        if force_json_content_type:
            response_content = json.dumps(
                {
                    "Error code": response.status_code,
                    "Error Reason": response.reason,
                    "Content": response_content,
                }
            )
        else:
            response_content = f"Error: [{response.status_code}] {response.reason}, Content: {response_content}"
    return (
        response.status_code,
        client_side_time_taken,
        response_content,
    )


def fitz_pdf_to_images(file: str) -> list[Image.Image]:
    """Converts a PDF to images using PyMuPDF."""
    fitz_pdf = fitz.open(file)
    imgs = []
    for page in fitz_pdf.pages():
        pix = page.get_pixmap(dpi=75)
        imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return imgs


def render_visual_media_input(file: str):
    """Renders visual media input into a Gradio component."""
    mime_type = mimetypes.guess_type(file)[0]
    if mime_type.startswith("image"):
        return gr.Gallery([file], columns=1, visible=True)
    elif mime_type == "application/pdf":
        fitz_pdf = fitz.open(file)
        page_thumbnails = fitz_pdf_to_images(fitz_pdf)
        return gr.Gallery(
            page_thumbnails,
            columns=min(len(page_thumbnails), 2),
            type="pil",
            visible=True,
        )
    else:
        raise NotImplementedError(f"Unsupported mime_type: {mime_type}")


### Form Extraction with confidence scores - HTTP example ###
with gr.Blocks(analytics_enabled=False) as form_extraction_with_confidence_block:
    # Define requesting function, which reshapes the input into the correct schema
    def form_ext_w_conf_upload(file: str):
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            (
                status_code,
                client_side_time_taken,
                response_content,
            ) = send_request(
                route="form_extraction_with_confidence",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        # Extract the image with bounding boxes from the result
        if isinstance(response_content, dict) and response_content.get(
            "result_img_with_bboxes"
        ):
            result_img = gr.Image(
                value=Image.open(
                    BytesIO(
                        base64.b64decode(response_content.pop("result_img_with_bboxes"))
                    )
                ),
                height=600,
                width=750,
                visible=True,
            )

        else:
            result_img = gr.Image(value=None, visible=False)
        return (
            status_code,
            client_side_time_taken,
            result_img,
            response_content,
        )

    # Input components
    form_ext_w_conf_instructions = gr.Markdown(
        (
            "This example extracts key information from Bank Account application forms "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_form_extraction_with_confidence.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. PyMuPDF is used to convert the PDFs into images.\n"
            "2. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores for each "
            "word/line.\n"
            "3. The image and raw extracted text are sent together in a multimodal request to GPT-4o. "
            "GPT-4o is instructed to extract all of the key information from the form in a structured format.\n"
            "4. GPT-4o's response is then cross-checked against the Document Intelligence response, combining GPT-4o's "
            "extracted values with the confidence scores from Document Intelligence.\n"
            "5. Finally, we look at all of the extracted values and their confidence scores to determine whether the "
            "result requires human review. If any of the values are missing or their confidence scores are below the "
            "minimum required threshold of 0.8 (80% confidence), the result is flagged for human review.\n\n"
            "This approach allows us to extract all of the required information in a structured format, while also "
            "having confidence that the extracted values are accurate. If the extracted values are below our minimum "
            "required confidence threshold, we can automatically route the form to our human review team. "
            "The final response also includes many of the intermediate outputs from the processing pipeline.\n"
            "To see how this process works, compare the outputs between the handwritten and machine-typed forms.\n"
            "* The machine-typed version has a higher confidence score for each extracted value, resulting in a "
            "`min_extracted_field_confidence_score` above the required threshold, meaning no human review is required.\n"
            "* The handwritten form on the other hand has a lower confidence score for each extracted value, resulting "
            "in a `min_extracted_field_confidence_score` below the required threshold, meaning the result is flagged "
            "for human review."
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        form_ext_w_conf_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        form_ext_w_conf_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    form_ext_w_conf_examples = gr.Examples(
        examples=DEMO_ACCOUNT_OPENING_FORM_FILES,
        inputs=[form_ext_w_conf_file_upload],
        label=FILE_EXAMPLES_LABEL,
        outputs=[
            form_ext_w_conf_input_thumbs,
        ],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    form_ext_w_conf_process_btn = gr.Button("Process File")
    # Output components
    with gr.Column(render=False) as form_ext_w_conf_output_row:
        form_ext_w_conf_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            form_ext_w_conf_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            form_ext_w_conf_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        form_ext_w_conf_img_output = gr.Image(
            label="Extracted Field Locations", visible=False
        )
        form_ext_w_conf_output_json = gr.JSON(label="API Response")

    form_ext_w_conf_output_row.render()
    # Actions
    form_ext_w_conf_process_btn.click(
        fn=form_ext_w_conf_upload,
        inputs=[form_ext_w_conf_file_upload],
        outputs=[
            form_ext_w_conf_status_code,
            form_ext_w_conf_time_taken,
            form_ext_w_conf_img_output,
            form_ext_w_conf_output_json,
        ],
    )


### Form Extraction - Blob to CosmosDB example ###
with gr.Blocks(analytics_enabled=False) as blob_form_extraction_to_cosmosdb_block:
    # Setup the CosmosDB container client for this use pipeline.
    # It relies on the CosmosDB database + container that were deployed with bicep
    blob_form_extraction_to_cosmosdb_container_client = (
        cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME).get_container_client(
            "blob-form-to-cosmosdb-container"
        )
    )

    # Define the processing function, which reshapes the input into the correct schema
    def blob_form_to_cosmosdb_process(file: str):
        start_time = time.time()
        try:
            # Append a random prefix to the filename to avoid conflicts
            random_prefix = str(int(time.time()))
            name, ext = os.path.splitext(os.path.basename(file))
            blob_name = f"{name}_{random_prefix}{ext}"
            # Upload data to blob storage
            blob_client = blob_service_client.get_blob_client(
                container="blob-form-to-cosmosdb-blobs", blob=blob_name
            )
            with open(file, "rb") as f:
                data = f.read()
                blob_client.upload_blob(data)
            # Query the CosmosDB container until a result is found
            gr.Info(
                "File uploaded to Blob Storage. Polling CosmosDB until the result appears..."
            )
            cosmos_timeout_secs = 30
            query = "SELECT * FROM c WHERE c.filename = @filename"
            parameters = [{"name": "@filename", "value": blob_name}]
            items = None
            cosmos_query_timer = time.time()
            while not items:
                time.sleep(2)
                if time.time() - cosmos_query_timer > cosmos_timeout_secs:
                    # If no record is found, return a placeholder result
                    timeout_error_text = f"Timeout: No result found in CosmosDB after {cosmos_timeout_secs} seconds."
                    client_side_time_taken = (
                        f"{round(time.time() - start_time, 1)} seconds"
                    )
                    return (404, client_side_time_taken, {"Error": timeout_error_text})

                items = list(
                    blob_form_extraction_to_cosmosdb_container_client.query_items(
                        query=query,
                        parameters=parameters,
                        enable_cross_partition_query=True,
                    )
                )
            client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
            # Delete the blob from storage
            blob_client.delete_blob()
            gr.Info("Task complete. Deleting the file from blob storage...")
            # Return status code, time taken and the first item from the list (it should only have one item)
            return (200, client_side_time_taken, items[0])
        except Exception as e:
            logging.exception("Error occurred during blob upload and CosmosDB query.")
            # Ensure the blob is deleted from storage if an error occurs
            try:
                blob_client.delete_blob()
                gr.Error("Error occurred. Deleting the file from blob storage...")
            except Exception as _e:
                pass
            # Return the result including the error.
            client_side_time_taken = f"{round(time.time() - start_time, 1)} seconds"
            return (500, client_side_time_taken, {"Error": str(e)})

    # Input components
    blob_form_to_cosmosdb_instructions = gr.Markdown(
        (
            "This example extracts key information from Bank Account application forms, triggered by uploading the document to blob storage and with the results written to a CosmosDB NoSQL database."
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_form_extraction_with_confidence.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. The PDF is uploaded to a blob storage container, triggering the pipeline.\n"
            "2. PyMuPDF is used to convert the PDFs into images.\n"
            "3. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores for each "
            "word/line.\n"
            "4. The image and raw extracted text are sent together in a multimodal request to GPT-4o. "
            "GPT-4o is instructed to extract all of the key information from the form in a structured format.\n"
            "5. The result is saved directly to a CosmosDB NoSQL table, ready for querying by downstream applications.\n"
            "Note: Any files uploaded as part of this pipeline are deleted immediately after processing.\n"
        ),
        show_label=False,
        line_breaks=True,
    )
    with gr.Row():
        blob_form_to_cosmosdb_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        blob_form_to_cosmosdb_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    blob_form_to_cosmosdb_examples = gr.Examples(
        examples=DEMO_ACCOUNT_OPENING_FORM_FILES,
        inputs=[blob_form_to_cosmosdb_file_upload],
        label=FILE_EXAMPLES_LABEL,
        outputs=[
            blob_form_to_cosmosdb_input_thumbs,
        ],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    blob_form_to_cosmosdb_process_btn = gr.Button("Process File")
    # Output components
    with gr.Column() as blob_form_to_cosmosdb_output_row:
        blob_form_to_cosmosdb_output_label = gr.Label(
            value="Pipeline Result", show_label=False
        )
        with gr.Row():
            blob_form_to_cosmosdb_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            blob_form_to_cosmosdb_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        blob_form_to_cosmosdb_output_json = gr.JSON(label="CosmosDB Record")
    # Actions
    blob_form_to_cosmosdb_process_btn.click(
        fn=blob_form_to_cosmosdb_process,
        inputs=[blob_form_to_cosmosdb_file_upload],
        outputs=[
            blob_form_to_cosmosdb_status_code,
            blob_form_to_cosmosdb_time_taken,
            blob_form_to_cosmosdb_output_json,
        ],
    )


### Call Center Audio Processing Example ###
with gr.Blocks(analytics_enabled=False) as call_center_audio_processing_block:
    # Define requesting function, which reshapes the input into the correct schema
    def cc_audio_upload(file: str, transcription_method: str):
        if file is None:
            gr.Warning(
                "Please select or upload an audio file, then click 'Process File'."
            )
            return ("", "", {})
        # Get response from the API
        with open(file, "rb") as f:
            payload = {"method": transcription_method}
            files = {
                "audio": (file, f, "audio/wav"),
                "json": ("payload.json", json.dumps(payload), "application/json"),
            }

            request_outputs = send_request(
                route="call_center_audio_analysis",
                files=files,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        return request_outputs

    # Input components
    cc_audio_proc_instructions = gr.Markdown(
        (
            "This example transcribes call center audio records and extract key information "
            "([Code Link](https://github.com/azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_call_center_audio_analytics.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. Azure AI Speech or Azure OpenAI Whisper is used to transcribe a call center recording.\n"
            "2. The result is formatted into a more readable format (with timestamps, speaker IDs and the text).\n"
            "3. GPT-4o is instructed to:\n* Classify the customer's sentiment & satisfaction\n"
            "* Summarize the call into 20 words or less.\n"
            "* Determine the next action that needs to be completed and the timestamp that the action was mentioned.\n"
            "* Extract a list of keywords from the conversation. Each of these keywords is cross-referenced with the "
            "raw transcription to retrieve the entire sentence that contains the keyword.\n"
            "4. The result is returned in a structured JSON format.\n\n"
            "The response includes the final result, as well as many of the intermediate outputs from the processing pipeline."
        ),
        show_label=False,
    )
    with gr.Column():
        cc_audio_proc_audio_input = gr.Audio(
            label="Upload Audio File",
            sources=["upload", "microphone"],
            type="filepath",
        )
        # Examples
        cc_audio_proc_examples = gr.Examples(
            label=AUDIO_EXAMPLES_LABEL,
            examples=DEMO_CALL_CENTER_AUDIO_FILES,
            inputs=[
                cc_audio_proc_audio_input,
            ],
        )
        cc_audio_proc_transcription_method = gr.Dropdown(
            label="Select Transcription Method",
            value="fast",
            choices={
                ("Fast Transcription API", "fast"),
                ("Azure OpenAI Whisper", "aoai_whisper"),
            },
        )
        cc_audio_proc_start_btn = gr.Button("Process File")
    # Output components
    with gr.Column() as cc_audio_proc_output_row:
        cc_audio_proc_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            cc_audio_proc_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            cc_audio_proc_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        cc_audio_proc_output_json = gr.JSON(label="API Response")
    # Actions
    cc_audio_proc_start_btn.click(
        fn=cc_audio_upload,
        inputs=[cc_audio_proc_audio_input, cc_audio_proc_transcription_method],
        outputs=[
            cc_audio_proc_status_code,
            cc_audio_proc_time_taken,
            cc_audio_proc_output_json,
        ],
    )


### Text summarization Example ###
def echo_input(input):
    """A simple function to echo the input. Useful for updating gradio components."""
    return input


with gr.Blocks(analytics_enabled=False) as sum_text_block:
    # Define requesting function, which reshapes the input into the correct schema
    def sum_text_send_request(input_text: str, summary_style: str, num_sentences: int):
        return send_request(
            route="summarize_text",
            json_data={
                "text": input_text,
                "summary_style": summary_style,
                "num_sentences": num_sentences,
            },
            force_json_content_type=False,
        )

    # Input components
    sum_text_instructions = gr.Markdown(
        (
            "This simple demo uses a single LLM request to summarize a body of text "
            "into a desired style and number of sentences "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_summarize_text.py))."
        ),
        show_label=False,
    )
    with gr.Row():
        sum_text_style_dropdown = gr.Dropdown(
            label="Select or enter the style of the generated summary",
            value="Australian Slang",
            choices=["Academic Writing", "Australian Slang", "Pirate", "Poetry"],
            allow_custom_value=True,
        )
        sum_text_num_sentences = gr.Number(
            label="Enter the desired number of sentences in the summary",
            value=3,
        )
    # Examples
    sum_text_input_text = gr.Textbox(
        value="", lines=3, label="Enter the text to be summarized"
    )
    sum_text_example_dropdown = gr.Dropdown(
        label=TEXT_EXAMPLES_LABEL,
        value=None,
        choices=TEXT_SAMPLES.items(),
    )
    sum_text_example_dropdown.change(
        fn=echo_input,
        inputs=[sum_text_example_dropdown],
        outputs=[sum_text_input_text],
    )
    # Output components
    with gr.Column(render=False) as sum_text_output_row:
        sum_text_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            sum_text_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            sum_text_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        sum_text_output_text = gr.Textbox(label="API Response", interactive=False)
    sum_text_get_response_btn = gr.Button("Process Text")
    sum_text_output_row.render()
    # Actions
    sum_text_get_response_btn.click(
        fn=sum_text_send_request,
        inputs=[sum_text_input_text, sum_text_style_dropdown, sum_text_num_sentences],
        outputs=[
            sum_text_status_code,
            sum_text_time_taken,
            sum_text_output_text,
        ],
    )

### Doc Intelligence Extraction + LLM Name Extraction Example ###
with gr.Blocks(analytics_enabled=False) as di_llm_ext_names_block:
    # Define requesting function, which reshapes the input into the correct schema
    def di_llm_ext_names_upload(file: str):
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            return send_request(
                route="doc_intel_extract_city_names",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )

    # Input components
    di_llm_ext_names_instructions = gr.Markdown(
        (
            "This example uses Azure Document Intelligence to extract the raw text from a PDF or image file, then "
            "sends a the instructions and extracted text to GPT-4o "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_doc_intel_extract_city_names.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores for each "
            "word/line.\n"
            "2. The raw extracted text is sent to GPT-4o and it is instructed to extract a list of city names that "
            "appear in the file.\n\n"
            "The response includes the final result, as well as many of the intermediate outputs from the processing pipeline."
        ),
        show_label=False,
    )
    with gr.Row():
        di_llm_ext_names_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        di_llm_ext_names_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    di_llm_ext_names_examples = gr.Examples(
        examples=DEMO_VISION_FILES,
        inputs=[di_llm_ext_names_file_upload],
        label=FILE_EXAMPLES_LABEL,
        outputs=[
            di_llm_ext_names_input_thumbs,
        ],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    form_ext_w_conf_process_btn = gr.Button("Process File")
    # Output components
    with gr.Column() as di_llm_ext_names_output_row:
        di_llm_ext_names_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_llm_ext_names_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            di_llm_ext_names_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        di_llm_ext_names_output_json = gr.JSON(label="API Response")
    # Actions
    form_ext_w_conf_process_btn.click(
        fn=di_llm_ext_names_upload,
        inputs=[di_llm_ext_names_file_upload],
        outputs=[
            di_llm_ext_names_status_code,
            di_llm_ext_names_time_taken,
            di_llm_ext_names_output_json,
        ],
    )


### Local PDF processing and extraction example ###
with gr.Blocks(analytics_enabled=False) as local_pdf_prc_block:
    # Define requesting function, which reshapes the input into the correct schema
    def local_pdf_process_upload(file: str):
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            return send_request(
                route="pymupdf_extract_city_names",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )

    # Input components
    local_pdf_process_instructions = gr.Markdown(
        (
            "This example uses PyMuPDF to extract the raw text from a PDF or image file, then "
            "sends a multimodal request (images and text) to GPT-4o "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_pymupdf_extract_city_names.py))."
            "\n\nThe pipeline is as follows:\n"
            "1. PyMuPDF extracts the raw text from the PDF.\n"
            "2. The raw extracted text is sent to GPT-4o and it is instructed to extract a list of city names that "
            "appear in the file.\n\n"
            "The response includes the final result, as well as many of the intermediate outputs from the processing pipeline."
        ),
        show_label=False,
    )
    with gr.Row():
        local_pdf_process_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        local_pdf_process_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    local_pdf_process_examples = gr.Examples(
        label=FILE_EXAMPLES_LABEL,
        examples=DEMO_PDF_FILES,
        inputs=[local_pdf_process_file_upload],
        outputs=[local_pdf_process_input_thumbs],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    local_pdf_process_process_btn = gr.Button("Process File")
    # Output components
    with gr.Column() as local_pdf_process_output_row:
        local_pdf_process_output_label = gr.Label(
            value="API Response", show_label=False
        )
        with gr.Row():
            local_pdf_process_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            local_pdf_process_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        local_pdf_process_output_json = gr.JSON(label="API Response")
    # Actions
    local_pdf_process_process_btn.click(
        fn=local_pdf_process_upload,
        inputs=[local_pdf_process_file_upload],
        outputs=[
            local_pdf_process_status_code,
            local_pdf_process_time_taken,
            local_pdf_process_output_json,
        ],
    )

### Doc Intelligence: Extract Raw Text Only Example ###
with gr.Blocks(analytics_enabled=False) as di_only_block:
    # Define requesting function, which reshapes the input into the correct schema
    def di_only_process_upload(file: str):
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            return send_request(
                route="doc_intel_only",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )

    # Input components
    di_only_instructions = gr.Markdown(
        (
            "This example uses Azure Document Intelligence to extract the raw text from a PDF or image file "
            "([Code Link](https://github.com/Azure/multimodal-ai-llm-processing-accelerator/blob/main/function_app/bp_doc_intel_only.py))."
            "\n\nNo further processing is completed on the output, and this example serves as a very basic example of "
            "an decoupled endpoint which can be used as one step in a larger, multi-stage pipeline."
        ),
        show_label=False,
    )
    with gr.Row():
        di_only_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        di_only_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Examples
    di_only_examples = gr.Examples(
        label=FILE_EXAMPLES_LABEL,
        examples=DEMO_VISION_FILES,
        inputs=[di_only_file_upload],
        outputs=[di_only_input_thumbs],
        fn=render_visual_media_input,
        run_on_click=True,
    )
    di_only_process_btn = gr.Button("Process File")
    # Output components
    with gr.Column() as di_only_output_row:
        di_only_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_only_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            di_only_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        di_only_output_text = gr.JSON(label="API Response")
    # Actions
    di_only_process_btn.click(
        fn=di_only_process_upload,
        inputs=[di_only_file_upload],
        outputs=[
            di_only_status_code,
            di_only_time_taken,
            di_only_output_text,
        ],
    )


with gr.Blocks(
    title="Multimodal AI & LLM Processing Accelerator",
    css="footer {visibility: hidden}",
    analytics_enabled=False,
) as demo:
    gr.Markdown(
        (
            "## Azure AI Services + OpenAI Pipeline Demos\n"
            "\n\nThis demo app showcases a few examples of different processing pipelines that incorporate Azure AI "
            "Services and Azure OpenAI. Click through the tabs to see examples of different processing pipelines.\n\n"
            "This app is based on the [Azure Multimodal AI & LLM Processing Accelerator]"
            "(https://github.com/azure/multimodal-ai-llm-processing-accelerator)."
        ),
        show_label=False,
    )
    with gr.Tab("Form Extraction with Confidence Scores (HTTP)"):
        form_extraction_with_confidence_block.render()
    with gr.Tab("Call Center Audio Processing (HTTP)"):
        call_center_audio_processing_block.render()
    with gr.Tab("Form Extraction (Blob -> CosmosDB)"):
        blob_form_extraction_to_cosmosdb_block.render()
    with gr.Tab("Summarize Text (HTTP)"):
        sum_text_block.render()
    with gr.Tab("City Names Extraction, Doc Intel (HTTP)"):
        di_llm_ext_names_block.render()
    with gr.Tab("City Names Extraction, PyMuPDF (HTTP)"):
        local_pdf_prc_block.render()
    with gr.Tab("Doc Intelligence Only (HTTP)"):
        di_only_block.render()

if __name__ == "__main__":
    # Start server by running: `gradio demo_app.py`, then navigate to http://localhost:8000
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        auth_message=auth_message,
        auth=auth,
    )
