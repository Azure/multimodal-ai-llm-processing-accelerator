import json
import logging
import os
from enum import Enum
from typing import Optional

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel, Field
from src.components.speech import (
    AOAI_WHISPER_MIME_TYPE_MAPPER,
    BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    AzureSpeechTranscriber,
    is_phrase_start_time_match,
)
from src.components.utils import base64_bytes_to_buffer, get_file_ext_and_mime_type
from src.helpers.common import MeasureRunTime
from src.result_enrichment.common import is_value_in_content
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_call_center_audio_analysis = func.Blueprint()
FUNCTION_ROUTE = "call_center_audio_analysis"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_WHISPER_DEPLOYMENT = os.getenv("AOAI_WHISPER_DEPLOYMENT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")

### Setup components
aoai_whisper_async_client = AsyncAzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_WHISPER_DEPLOYMENT,
    azure_ad_token_provider=token_provider,
    api_version="2024-06-01",
)
transcriber = AzureSpeechTranscriber(
    speech_endpoint=SPEECH_ENDPOINT,
    azure_ad_token_provider=token_provider,
    aoai_whisper_async_client=aoai_whisper_async_client,
)
# Define the configuration for the transcription job
# More info: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create
# And: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription-create?pivots=rest-api#request-configuration-options
fast_transcription_definition = {
    "locales": ["en-US"],
    "profanityFilterMode": "Masked",
    "diarizationEnabled": False,  # Not available for fast transcription as of August 2024
    "wordLevelTimestampsEnabled": True,
}
# More info: https://platform.openai.com/docs/guides/speech-to-text
aoai_whisper_kwargs = {
    "language": "en",
    "prompt": None,
    "temperature": None,
    "timeout": 60,
}

aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=token_provider,
    api_version="2024-06-01",
    timeout=30,
    max_retries=0,
)

# Create mappers to handle different types of transcription methods
TRANSCRIPTION_METHOD_TO_MIME_MAPPER = {
    "fast": BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    "aoai_whisper": AOAI_WHISPER_MIME_TYPE_MAPPER,
}


### Setup Pydantic models for validation of LLM calls, and the Function response itself
# Classification fields
class CustomerSatisfactionEnum(Enum):
    Satisfied = "Satisfied"
    Dissatisfied = "Dissatisfied"


CUSTOMER_SATISFACTION_VALUES = [e.value for e in CustomerSatisfactionEnum]


class CustomerSentimentEnum(Enum):
    Positive = "Positive"
    Neutral = "Neutral"
    Negative = "Negative"


CUSTOMER_SENTIMENT_VALUES = [e.value for e in CustomerSentimentEnum]


# Setup a class for the raw keywords returned by the LLM, and then the enriched version (after we match the keywords to the transcription)
class RawKeyword(LLMResponseBaseModel):
    keyword: str = Field(
        description="A keyword extracted from the call. This should be a direct match to a word or phrase in the transcription without modification of the spelling or grammar.",
        examples=["credit card account"],
    )
    timestamp: str = Field(
        description="The timestamp of the sentence from which the keyword was uttered.",
        examples=["0:18"],
    )


class ProcessedKeyWord(RawKeyword):
    keyword_matched_to_transcription_sentence: bool = Field(
        description="Whether the keyword was matched to a single sentence in the transcription.",
    )
    full_sentence_text: Optional[str] = Field(
        default=None,
        description="The full text of the sentence in which the keyword was uttered.",
    )
    sentence_confidence: Optional[float] = Field(
        default=None,
        description="The confidence score of the sentence from which the keyword was extracted.",
    )
    sentence_start_time_secs: Optional[float] = Field(
        default=None,
        description="The start time of the sentence in the audio recording.",
    )
    sentence_end_time_secs: Optional[float] = Field(
        default=None,
        description="The end time of the sentence in the audio recording.",
    )


# Define the full schema for the LLM's response. We inherit from LLMResponseBaseModel to get the prompt generation functionality.
class LLMRawResponseModel(LLMResponseBaseModel):
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

    call_summary: str = Field(
        description="A summary of the call, including the topics and key action items. This should be no more than 20 words long.",
        examples=[
            "The customer called to close their credit card account. The agent closed the account and a confirmation email was sent."
        ],
    )
    customer_satisfaction: CustomerSatisfactionEnum = Field(
        description=f"Is the customer satisfied with the agent interaction. It must only be one of these options: {CUSTOMER_SATISFACTION_VALUES}.",
        examples=[CUSTOMER_SATISFACTION_VALUES[0]],
    )
    customer_sentiment: CustomerSentimentEnum = Field(
        description=f"The sentiment of the customer on the call. It must be one of these options:  {CUSTOMER_SENTIMENT_VALUES}.",
        examples=[CUSTOMER_SATISFACTION_VALUES[-1]],
    )
    next_action: Optional[str] = Field(
        description="The next action that needs to be taken, if there is one. This should be no more than 20 words long. If no action is necessary, return null.",
        examples=["The agent will send a follow-up email to the customer."],
    )
    next_action_sentence_timestamp: Optional[str] = Field(
        description="The timestamp of the sentence where the next action was mentioned. This should be the timestamp written in the transcription.",
        examples=["6:12"],
    )
    keywords: list[RawKeyword] = Field(
        description=(
            "A list of keywords related to the purpose of the call, the products they are interested in, or any issues that have occurred. "
            "Each result should include the exact keyword and a timestamp of the sentence where it was uttered. "
            "Each keyword should match a word or phrase in the transcription without modification of the spelling or grammar)."
        ),
        examples=[
            [
                {"keyword": "credit card account", "timestamp": "0:18"},
                {"keyword": "bank account", "timestamp": "0:46"},
                {"keyword": "insurance", "timestamp": "2:37"},
                {"keyword": "complaint", "timestamp": "4:52"},
            ]
        ],
    )


class ProcessedResultModel(LLMRawResponseModel):
    """
    Defined the schema for the processed result that will be returned by the
    function. This class inherits from LLMRawResponseModel but overwrites the
    `keywords` field to use the ProcessedKeyWord model instead of the
    RawKeyword. This way we can return the processed keywords with additional
    metadata.
    """

    keywords: list[ProcessedKeyWord] = Field(
        description=(
            "A list of key phrases related to the purpose of the call, the products they are interested in, or any issues that have occurred. "
            "Each item includes the keyword and timestamp of the sentence as extracted by the LLM, along with additional metadata "
            "that is merged from the Transcription result."
        ),
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
    result: Optional[ProcessedResultModel] = Field(
        default=None, description="The final result of the pipeline."
    )
    error_text: Optional[str] = Field(
        default=None,
        description="If an error occurred, this field will contain the error message.",
    )
    speech_extracted_text: Optional[str] = Field(
        default=None,
        description="The raw & formatted text content extracted by Azure AI Speech.",
    )
    speech_raw_response: Optional[list | dict] = Field(
        default=None, description="The raw API response from Azure AI Speech."
    )
    speech_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to transcribe the text using Azure AI Speech.",
    )
    llm_input_messages: Optional[list[dict]] = Field(
        default=None, description="The messages that were sent to the LLM."
    )
    llm_reply_message: Optional[dict] = Field(
        default=None, description="The message that was received from the LLM."
    )
    llm_raw_response: Optional[str] = Field(
        default=None, description="The raw text response from the LLM."
    )
    llm_time_taken_secs: Optional[float] = Field(
        default=None, description="The time taken to receive a response from the LLM."
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )


# Create the system prompt for the LLM, dynamically including the JSON schema
# of the expected response so that any changes to the schema are automatically
# reflected in the prompt, and in a JSON format that is similar in structure
# to the training data on which the LLM was trained (increasing reliability of
# the result).
LLM_SYSTEM_PROMPT = (
    "You are a customer service contact center agent, and you specialize in summarizing and classifying "
    "the content of customer service call recordings.\n"
    "Your task is to review a customer service call and extract all of the key information from the call recording.\n"
    f"{LLMRawResponseModel.get_prompt_json_example(include_preceding_json_instructions=True)}"
)


@bp_call_center_audio_analysis.route(route=FUNCTION_ROUTE)
async def call_center_audio_analysis(
    req: func.HttpRequest,
) -> func.HttpResponse:
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    # Create the object to hold all intermediate and final values. We will progressively update
    # values as each stage of the pipeline is completed, allowing us to return a partial
    # response in case of an error at any stage.
    output_model = FunctionReponseModel(success=False)
    try:
        # Create error_text and error_code variables. These will be updated as
        # we move through the pipeline so that if a step fails, the vars reflect
        # what has failed. If all steps complete successfully, the vars are
        # never used.
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        ## Check the request body
        # Transcription method
        request_json_content = json.loads(req.files["json"].read().decode("utf-8"))
        transcription_method = request_json_content["method"]
        if transcription_method not in TRANSCRIPTION_METHOD_TO_MIME_MAPPER:
            output_model.error_text = f"Invalid transcription method `{transcription_method}`. Please use one of {TRANSCRIPTION_METHOD_TO_MIME_MAPPER.keys().tolist()}"
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=error_code,
            )
        # Audio file & type
        valid_mime_to_filetype_mapper = TRANSCRIPTION_METHOD_TO_MIME_MAPPER[
            transcription_method
        ]
        error_text = "Invalid audio file. Please sbumit a file with a valid filename and content type."
        audio_file = req.files["audio"]
        audio_file_b64 = audio_file.read()
        audio_file_ext, _audio_file_content_type = get_file_ext_and_mime_type(
            valid_mimes_to_file_ext_mapper=valid_mime_to_filetype_mapper,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
        )
        audio_filename = (
            audio_file.filename if audio_file.filename else f"file.{audio_file_ext}"
        )
        # Get the transcription result
        error_text = "An error occurred during audio transcription."
        error_code = 500
        with MeasureRunTime() as speech_timer:
            if transcription_method == "fast":
                transcription, raw_transcription_api_response = (
                    await transcriber.get_fast_transcription_async(
                        audio_file=audio_file_b64,
                        definition=fast_transcription_definition,
                    )
                )
            else:
                audio_file = base64_bytes_to_buffer(
                    b64_str=audio_file_b64, name=audio_filename
                )
                transcription, raw_transcription_api_response = (
                    await transcriber.get_aoai_whisper_transcription_async(
                        audio_file=audio_file,
                        **aoai_whisper_kwargs,
                    )
                )
        formatted_transcription_text = transcription.to_formatted_str(
            transcription_prefix_format="Language: {language}\nDuration: {formatted_duration} minutes\n\nConversation:\n",
            phrase_format="[{start_min}:{start_sub_sec}] {auto_phrase_source_name} {auto_phrase_source_id}: {display_text}",
        )
        output_model.speech_extracted_text = formatted_transcription_text
        output_model.speech_raw_response = raw_transcription_api_response
        output_model.speech_time_taken_secs = speech_timer.time_taken
        # Create the messages to send to the LLM in the following order:
        # 1. System prompt
        # 2. Audio transcription, formatted in a clear way
        error_text = "An error occurred while creating the LLM input messages."
        input_messages = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": formatted_transcription_text,
            },
        ]
        output_model.llm_input_messages = input_messages
        # Send request to LLM
        error_text = "An error occurred when sending the LLM request."
        with MeasureRunTime() as llm_timer:
            llm_result = aoai_client.chat.completions.create(
                messages=input_messages,
                model=AOAI_LLM_DEPLOYMENT,
                response_format={"type": "json_object"},  # Ensure we get JSON responses
            )
        output_model.llm_time_taken_secs = llm_timer.time_taken
        # Validate that the LLM response matches the expected schema
        error_text = "An error occurred when validating the LLM's returned response into the expected schema."
        output_model.llm_reply_message = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content
        llm_structured_response = LLMRawResponseModel(
            **json.loads(llm_result.choices[0].message.content)
        )
        # Process each keyword and add additional metadata from the transcription and return a processed result
        error_text = "An error occurred when post-processing the keywords."
        processed_keywords = list()
        for keyword in llm_structured_response.keywords:
            # Find the sentence in the transcription that contains the keyword.
            # Search for all sentences with the same timestamp and where the LLM's
            # text was contained in the sentence. If not available, mark the keyword as not matched.
            keyword_sentence_start_time_secs = int(
                keyword.timestamp.split(":")[0]
            ) * 60 + int(keyword.timestamp.split(":")[1])
            matching_phrases = [
                phrase
                for phrase in transcription.phrases
                if is_value_in_content(
                    keyword.keyword.lower(), phrase.display_text.lower()
                )
                and is_phrase_start_time_match(
                    expected_start_time_secs=keyword_sentence_start_time_secs,
                    phrase=phrase,
                    start_time_tolerance_secs=1,
                )
            ]
            if len(matching_phrases) == 1:
                processed_keywords.append(
                    ProcessedKeyWord(
                        **keyword.dict(),
                        keyword_matched_to_transcription_sentence=True,
                        full_sentence_text=matching_phrases[0].display_text,
                        sentence_confidence=matching_phrases[0].confidence,
                        sentence_start_time_secs=matching_phrases[0].start_secs,
                        sentence_end_time_secs=matching_phrases[0].end_secs,
                    )
                )
            else:
                processed_keywords.append(
                    ProcessedKeyWord(
                        **keyword.dict(),
                        keyword_matched_to_transcription_sentence=False,
                    )
                )
        # Construct processed model, replacing the raw keywords with the processed keywords
        llm_structured_response_dict = llm_structured_response.dict()
        llm_structured_response_dict.pop("keywords")
        output_model.result = ProcessedResultModel(
            **llm_structured_response_dict,
            keywords=processed_keywords,
        )
        # All steps completed successfully, set success=True and return the final result
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as _e:
        logging.exception("An error occurred during processing.")
        output_model.error_text = error_text
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=error_code,
        )
