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
from dotenv import load_dotenv
from PIL import Image

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()

# Get function endpoint from env vars. If not set, default to local host.
# Note: This assumes the function is being run prior to launching the web app.
FUNCTION_HOSTNAME = os.getenv("FUNCTION_HOSTNAME", "http://localhost:7071/")
FUNCTION_KEY = os.getenv("FUNCTION_KEY", "")
FUNCTION_ENDPOINT = os.path.join(FUNCTION_HOSTNAME, "api")
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


def send_request(
    route: str,
    json_data: Optional[dict] = None,
    data: Optional[dict] = None,
    headers: Optional[dict] = None,
    force_json_content_type: bool = True,
) -> Union[int, str]:
    """Sends a POST request to the specified route on the function host."""
    if FUNCTION_KEY:
        headers = headers or {}
        headers["x-functions-key"] = FUNCTION_KEY
    start_time = time.time()
    response = requests.post(
        os.path.join(FUNCTION_ENDPOINT, route),
        json=json_data,
        data=data,
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
            "into a desired style and number of sentences."
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
        label="Select an example text from this Dropdown",
        value="",
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
    sum_text_get_response_btn = gr.Button("Get Response")
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

### Local PDF processing and extraction example ###
with gr.Blocks(analytics_enabled=False) as local_pdf_prc_block:
    # Define requesting function, which reshapes the input into the correct schema
    def local_pdf_process_upload(file: str):
        # Render the input media
        rendered_input = render_visual_media_input(file)
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            request_outputs = send_request(
                route="pymupdf_extract_city_names",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        return (rendered_input,) + request_outputs

    # Input components
    local_pdf_process_instructions = gr.Markdown(
        "This example uses PyMuPDF to extract the raw text and images from a PDF, then uses "
        "GPT-4o to extract all city names from the text.",
        show_label=False,
    )
    with gr.Row():
        local_pdf_process_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        local_pdf_process_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Output components
    with gr.Column(render=False) as local_pdf_process_output_row:
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
    # Examples
    local_pdf_process_examples = gr.Examples(
        examples=DEMO_PDF_FILES,
        inputs=[local_pdf_process_file_upload],
        outputs=[
            local_pdf_process_input_thumbs,
            local_pdf_process_status_code,
            local_pdf_process_time_taken,
            local_pdf_process_output_json,
        ],
        fn=local_pdf_process_upload,
        run_on_click=True,
    )
    local_pdf_process_output_row.render()
    # Actions
    local_pdf_process_file_upload.upload(
        fn=local_pdf_process_upload,
        inputs=[local_pdf_process_file_upload],
        outputs=[
            local_pdf_process_input_thumbs,
            local_pdf_process_status_code,
            local_pdf_process_time_taken,
            local_pdf_process_output_json,
        ],
    )

### Doc Intelligence: Extract Raw Text Example ###
with gr.Blocks(analytics_enabled=False) as di_ert_block:
    # Define requesting function, which reshapes the input into the correct schema
    def di_ert_process_upload(file: str):
        # Render the input media
        rendered_input = render_visual_media_input(file)
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            request_outputs = send_request(
                route="doc_intel_only",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        return (rendered_input,) + request_outputs

    # Input components
    di_ert_instructions = gr.Markdown(
        (
            "This example uses Azure Document Intelligence to extract the raw text from a PDF or image file. "
            "No further processing is completed on the output."
        ),
        show_label=False,
    )
    with gr.Row():
        di_ert_file_upload = gr.File(
            label="Upload File", file_count="single", type="filepath"
        )
        di_ert_input_thumbs = gr.Gallery(
            label="File Preview", object_fit="contain", visible=True
        )
    # Output components
    with gr.Column(render=False) as di_ert_output_row:
        di_ert_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_ert_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            di_ert_time_taken = gr.Textbox(label="Time Taken", interactive=False)
        di_ert_output_text = gr.JSON(label="API Response")
    # Examples
    di_ert_examples = gr.Examples(
        examples=DEMO_VISION_FILES,
        inputs=[di_ert_file_upload],
        outputs=[
            di_ert_input_thumbs,
            di_ert_status_code,
            di_ert_time_taken,
            di_ert_output_text,
        ],
        fn=di_ert_process_upload,
        run_on_click=True,
    )
    di_ert_output_row.render()
    # Actions
    di_ert_file_upload.upload(
        fn=di_ert_process_upload,
        inputs=[di_ert_file_upload],
        outputs=[
            di_ert_input_thumbs,
            di_ert_status_code,
            di_ert_time_taken,
            di_ert_output_text,
        ],
    )

### Doc Intelligence Extraction + LLM Name Extraction Example ###
with gr.Blocks(analytics_enabled=False) as di_llm_ext_names_block:
    # Define requesting function, which reshapes the input into the correct schema
    def di_llm_ext_names_upload(file: str):
        # Render the input media
        rendered_input = render_visual_media_input(file)
        # Get response from the API
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as f:
            data = f.read()
            headers = {"Content-Type": mime_type}
            request_outputs = send_request(
                route="doc_intel_extract_city_names",
                data=data,
                headers=headers,
                force_json_content_type=True,  # JSON gradio block requires this
            )
        return (rendered_input,) + request_outputs

    # Input components
    di_llm_ext_names_instructions = gr.Markdown(
        (
            "This example uses Azure Document Intelligence to extract the raw text from a PDF or image file, then "
            "sends a multimodal request (images and text) to GPT-4o. GPT-4o is instructed to return all names found in "
            "the input in a structured JSON format.<br>The response includes the final result, as well as many of the "
            "intermediate outputs from the processing pipeline."
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
    # Output components
    with gr.Column(render=False) as di_llm_ext_names_output_row:
        di_llm_ext_names_output_label = gr.Label(value="API Response", show_label=False)
        with gr.Row():
            di_llm_ext_names_status_code = gr.Textbox(
                label="Response Status Code", interactive=False
            )
            di_llm_ext_names_time_taken = gr.Textbox(
                label="Time Taken", interactive=False
            )
        di_llm_ext_names_output_json = gr.JSON(label="API Response")
    # Examples
    di_llm_ext_names_examples = gr.Examples(
        examples=DEMO_VISION_FILES,
        inputs=[di_llm_ext_names_file_upload],
        outputs=[
            di_llm_ext_names_input_thumbs,
            di_llm_ext_names_status_code,
            di_llm_ext_names_time_taken,
            di_llm_ext_names_output_json,
        ],
        fn=di_llm_ext_names_upload,
        run_on_click=True,
    )
    di_llm_ext_names_output_row.render()
    # Actions
    di_llm_ext_names_file_upload.upload(
        fn=di_llm_ext_names_upload,
        inputs=[di_llm_ext_names_file_upload],
        outputs=[
            di_llm_ext_names_input_thumbs,
            di_llm_ext_names_status_code,
            di_llm_ext_names_time_taken,
            di_llm_ext_names_output_json,
        ],
    )

### Form Extraction with confidence scores example ###
with gr.Blocks(analytics_enabled=False) as form_extraction_with_confidence_block:
    # Define requesting function, which reshapes the input into the correct schema
    def form_ext_w_conf_upload(file: str):
        # Render the input media
        rendered_input = render_visual_media_input(file)
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
            rendered_input,
            status_code,
            client_side_time_taken,
            result_img,
            response_content,
        )

    # Input components
    form_ext_w_conf_instructions = gr.Markdown(
        (
            "This example extracts key information from Bank Account application forms. The pipeline runs as follows:\n"
            "1. PyMuPDF is used to convert the PDFs into images.\n"
            "2. Azure Document Intelligence extracts the raw text from the PDF along with confidence scores for each "
            "word/line.\n"
            "3. The image and raw extracted text are sent together in a multimodal request to GPT-4o. "
            "GPT-4o is instructed to extract al of the key information from the form in a structured format.\n"
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
    # Examples
    form_ext_w_conf_examples = gr.Examples(
        examples=DEMO_ACCOUNT_OPENING_FORM_FILES,
        inputs=[form_ext_w_conf_file_upload],
        outputs=[
            form_ext_w_conf_input_thumbs,
            form_ext_w_conf_status_code,
            form_ext_w_conf_time_taken,
            form_ext_w_conf_img_output,
            form_ext_w_conf_output_json,
        ],
        fn=form_ext_w_conf_upload,
        run_on_click=True,
    )
    form_ext_w_conf_output_row.render()
    # Actions
    form_ext_w_conf_file_upload.upload(
        fn=form_ext_w_conf_upload,
        inputs=[form_ext_w_conf_file_upload],
        outputs=[
            form_ext_w_conf_input_thumbs,
            form_ext_w_conf_status_code,
            form_ext_w_conf_time_taken,
            form_ext_w_conf_img_output,
            form_ext_w_conf_output_json,
        ],
    )

with gr.Blocks(css="footer {visibility: hidden}", analytics_enabled=False) as demo:
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
    with gr.Tab("Form Extraction with Confidence Scores"):
        form_extraction_with_confidence_block.render()
    with gr.Tab("Summarize Text"):
        sum_text_block.render()
    with gr.Tab("City Names Extraction (Doc Intelligence)"):
        di_llm_ext_names_block.render()
    with gr.Tab("City Names Extraction (PyMuPDF)"):
        local_pdf_prc_block.render()
    with gr.Tab("Doc Intelligence Only"):
        di_ert_block.render()

if __name__ == "__main__":
    # Start server by running: `gradio app.py`, then navigate to http://localhost:8000
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        auth_message=auth_message,
        auth=auth,
    )
