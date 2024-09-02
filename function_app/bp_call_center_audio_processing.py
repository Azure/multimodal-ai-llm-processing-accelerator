import json
import logging
import os
from enum import Enum
from typing import Optional

import azure.functions as func
from dotenv import load_dotenv
from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field
from src.components.speech import (
    AOAI_WHISPER_MIME_TYPE_MAPPER,
    BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    AzureSpeechTranscriber,
    is_phrase_start_time_match,
)
from src.components.utils import (
    InvalidFileTypeError,
    base64_file_to_buffer,
    get_file_ext_and_mime_type,
)
from src.helpers.common import MeasureRunTime
from src.result_enrichment.common import is_value_in_content
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_call_center_audio_processing = func.Blueprint()

SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_WHISPER_DEPLOYMENT = os.getenv("AOAI_WHISPER_DEPLOYMENT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
# Load the API key as a Secret, so that it is not logged in any traces or saved if the Haystack component is exported.
AOAI_API_KEY_SECRET = Secret.from_token(AOAI_API_KEY)

### Setup components
aoai_whisper_async_client = AsyncAzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_WHISPER_DEPLOYMENT,
    api_key=AOAI_API_KEY,
    api_version="2024-06-01",
)
transcriber = AzureSpeechTranscriber(
    speech_endpoint=SPEECH_ENDPOINT,
    speech_key=SPEECH_API_KEY,
    aoai_whisper_async_client=aoai_whisper_async_client,
)

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

azure_generator = AzureOpenAIChatGenerator(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    api_key=AOAI_API_KEY_SECRET,
    api_version="2024-06-01",
    generation_kwargs={
        "response_format": {"type": "json_object"}
    },  # Ensure we get JSON responses
)

# Create mappers to handle different types of transcription methods
TRANSCRIPTION_METHOD_TO_MIME_MAPPER = {
    "fast": BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    "aoai_whisper": AOAI_WHISPER_MIME_TYPE_MAPPER,
}


# Setup Pydantic models for validation of LLM calls, and the Function response itself
class CustomerSatisfactionEnum(Enum):
    Satisfied = "Satisfied"
    Dissatisfied = "Dissatisfied"


CUSTOMER_SATISFACTION_VALUES = [e.value for e in CustomerSatisfactionEnum]


class CustomerSentimentEnum(Enum):
    Positive = "Positive"
    Neutral = "Neutral"
    Negative = "Negative"


CUSTOMER_SENTIMENT_VALUES = [e.value for e in CustomerSentimentEnum]


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
    llm_reply_messages: Optional[list[dict]] = Field(
        default=None, description="The messages that were received from the LLM."
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


LLM_SYSTEM_PROMPT = (
    "You are a customer service contact center agent, and you specialize in summarizing and classifying "
    "the content of customer service call recordings.\n"
    "Your task is to review a customer service call and extract all of the key information from the call recording.\n"
    f"{LLMRawResponseModel.get_prompt_json_example(include_preceding_json_instructions=True)}"
)


@bp_call_center_audio_processing.route(route="call_center_audio_processing")
async def call_center_audio_processing(
    req: func.HttpRequest,
) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    try:
        func_timer = MeasureRunTime()
        func_timer.start()
        # Create the object to hold all intermediate and final values. We will progressively update
        # values as each stage of the pipeline is completed, allowing us to return a partial
        # response in case of an error at any stage.
        output_model = FunctionReponseModel(success=False)

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
                status_code=500,
            )
        valid_mime_to_filetype_mapper = TRANSCRIPTION_METHOD_TO_MIME_MAPPER[
            transcription_method
        ]
        # Audio file & type
        audio_file = req.files["audio"]
        audio_file_b64 = audio_file.read()
        try:
            audio_file_ext, _audio_file_content_type = get_file_ext_and_mime_type(
                valid_mimes_to_file_ext_mapper=valid_mime_to_filetype_mapper,
                filename=audio_file.filename,
                content_type=audio_file.content_type,
            )
            audio_filename = (
                audio_file.filename if audio_file.filename else f"file.{audio_file_ext}"
            )
        except InvalidFileTypeError as e:
            output_model.error_text = (
                "Please sbumit a file with a valid filename or content type. " + str(e)
            )
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Get the transcription result
        try:
            with MeasureRunTime() as speech_timer:
                if transcription_method == "fast":
                    transcription, raw_transcription_api_response = (
                        await transcriber.get_fast_transcription_async(
                            audio_file=audio_file_b64,
                            definition=fast_transcription_definition,
                        )
                    )
                else:
                    audio_file = base64_file_to_buffer(
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
        except Exception as _e:
            output_model.error_text = "An error occurred during audio transcription."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Create the messages to send to the LLM in the following order:
        # 1. System prompt
        # 2. Audio transcription
        try:
            input_messages = [
                ChatMessage.from_system(LLM_SYSTEM_PROMPT),
                ChatMessage.from_user(formatted_transcription_text),
            ]
            output_model.llm_input_messages = [
                msg.to_openai_format() for msg in input_messages
            ]
        except Exception as _e:
            output_model.error_text = (
                "An error occurred while creating the LLM input messages."
            )
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Send request to LLM
        try:
            with MeasureRunTime() as llm_timer:
                llm_result = azure_generator.run(messages=input_messages)
            output_model.llm_time_taken_secs = llm_timer.time_taken
        except Exception as _e:
            output_model.error_text = "An error occurred when sending the LLM request."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Validate that the LLM response matches the expected schema
        try:
            output_model.llm_reply_messages = [
                msg.to_openai_format() for msg in llm_result["replies"]
            ]
            if len(llm_result["replies"]) != 1:
                raise ValueError(
                    "The LLM response did not contain exactly one message."
                )
            output_model.llm_raw_response = llm_result["replies"][0].content
            llm_structured_response = LLMRawResponseModel(
                **json.loads(llm_result["replies"][0].content)
            )
        except Exception as _e:
            output_model.error_text = "An error occurred when validating the LLM's returned response into the expected schema."
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Process each keyword and add additional metadata from the transcription and return a processed result
        try:
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
        except Exception as _e:
            output_model.error_text = (
                "An error occurred when post-processing the keywords."
            )
            output_model.func_time_taken_secs = func_timer.stop()
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
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
        return func.HttpResponse(
            "An error occurred during processing.",
            status_code=500,
        )
