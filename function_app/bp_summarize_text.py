import logging
import os

import azure.functions as func
from dotenv import load_dotenv
from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from pydantic import BaseModel, Field

load_dotenv()

bp_summarize_text = func.Blueprint()

# Load environment variables
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
# Load the API key as a Secret, so that it is not logged in any traces or saved if the component is exported.
DOC_INTEL_API_KEY_SECRET = Secret.from_env_var("DOC_INTEL_API_KEY")
AOAI_API_KEY_SECRET = Secret.from_env_var("AOAI_API_KEY")

# Setup Haystack components
azure_generator = AzureOpenAIChatGenerator(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    api_key=AOAI_API_KEY_SECRET,
    api_version="2024-06-01",
)


# Setup Pydantic models for validation of the request and response
class FunctionRequestModel(BaseModel):
    """
    Defines the schema that will be expected in the request body. We'll use this to
    ensure that the request contains the correct values and structure, and to allow
    a partially filled request to be processed in case of an error.
    """

    text: str = Field(description="The text to be summarized")
    summary_style: str = Field(description="The style of the summary to be generated")
    num_sentences: int = Field(
        description="The desired number of sentences for the summary", gt=0
    )


LLM_SYSTEM_PROMPT = (
    "You are an expert in summarizing and adapting information. "
    "Your task is to take the following text and summarize it into "
    "exactly {} sentences, and with the result written in the style of '{}'."
)


@bp_summarize_text.route(route="summarize_text")
def summarize_text(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    try:
        try:
            # Check the request body
            req_body = req.get_json()
            request_obj = FunctionRequestModel(**req_body)
        except Exception as _e:
            return func.HttpResponse(
                (
                    "The request body was not in the expected format. Please ensure that it is "
                    "a valid JSON object with the following fields: {}"
                ).format(list(FunctionRequestModel.model_fields.keys())),
                status_code=400,
            )
        # Create the messages to send to the LLM
        input_messages = [
            ChatMessage.from_system(
                LLM_SYSTEM_PROMPT.format(
                    request_obj.num_sentences, request_obj.summary_style
                )
            ),
            ChatMessage.from_user(request_obj.text),
        ]
        # Send request to LLM
        llm_result = azure_generator.run(messages=input_messages)
        llm_text_response = llm_result["replies"][0].content
        # All steps completed successfully, set success=True and return the final result
        return func.HttpResponse(
            body=llm_text_response,
            status_code=200,
        )
    except Exception as _e:
        logging.exception("An error occurred during processing.")
        return func.HttpResponse(
            "An error occurred during processing.",
            status_code=500,
        )
