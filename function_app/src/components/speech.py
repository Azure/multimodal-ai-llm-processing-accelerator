import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx
from haystack import Document
from pydantic import BaseModel, Field, computed_field


class AzureSpeechTranscriptionEnum(Enum):
    BATCH = "Batch"
    REALTIME = "Realtime"
    FAST = "Fast"
    AOAI_WHISPER = "Azure OpenAI Whisper"


def get_type_from_type_dict(type_dict: dict[str, str]) -> str:
    if "$ref" in type_dict:
        return "class:{}".format(type_dict["$ref"].split("/")[-1])
    return type_dict.get("type")


class TranscriptionBaseModel(BaseModel):
    """
    Data object for transcribed audio data.
    """

    def model_field_info(self) -> dict[str, str]:
        """
        Returns a dictionary of field names and their descriptions.
        """
        output = dict()
        for field, info_dict in self.model_json_schema(mode="serialization")[
            "properties"
        ].items():
            if "anyOf" in info_dict:
                type = " | ".join(
                    get_type_from_type_dict(t) for t in info_dict["anyOf"]
                )
            elif "allOf" in info_dict:
                type = " & ".join(
                    get_type_from_type_dict(t) for t in info_dict["allOf"]
                )
            elif "type" in info_dict:
                type = get_type_from_type_dict(info_dict)
            else:
                raise ValueError("Field type could not be processed.")
            output[field] = "[{}] - {}".format(type, info_dict["description"])
        return output

    def get_vars_dict(self, replace_none_with: Any = "Unknown") -> dict[str, Any]:
        """
        Returns a dictionary of all variables in the object.

        Args:
            replace_none_with: The value to replace None values with.

        Returns:
            A dictionary of all variables and properties in the object.
        """
        vars = dict()
        for key, value in self.dict().items():
            if value is None:
                vars[key] = replace_none_with
            else:
                vars[key] = value
        return vars


class TranscribedWord(TranscriptionBaseModel):
    """
    Data object representing a single word in a transcription.
    """

    offset_id: int = Field(
        ...,
        description="The ID of the word, offset from the start of the transcription.",
    )
    text: str = Field(..., description="The text of the word.")
    confidence: Optional[float] = Field(
        None, description="The confidence score of the word, if available."
    )
    start_secs: float = Field(..., description="The start time of the word in seconds.")
    duration_secs: float = Field(
        ..., description="The duration of the word in seconds."
    )
    end_secs: float = Field(..., description="The end time of the word in seconds.")

    @computed_field(
        description="The minute in which the phrase started. E.g. the 67th second becomes 1."
    )
    def start_min(self) -> int:
        return int(self.start_secs // 60)

    @computed_field(
        description="The sub-minute second of the phrase in minutes. E.g. the 67th second becomes 7."
    )
    def start_sub_sec(self) -> int:
        return int(self.start_secs % 60)


class TranscribedPhrase(TranscriptionBaseModel):
    """
    Data object representing a phrase/sentence in a transcription.
    """

    offset_id: int = Field(
        ...,
        description="The ID of the phrase, offset from the start of the transcription.",
    )
    lexical_text: Optional[str] = Field(
        None, description="The lexical form of the phrase."
    )
    itn_text: Optional[str] = Field(None, description="The ITN form of the phrase.")
    masked_itn_text: Optional[str] = Field(
        None, description="The masked ITN form of the phrase."
    )
    display_text: str = Field(..., description="The display form of the phrase.")
    channel: int = Field(
        ..., description="The audio channel from which the phrase was spoken."
    )
    speaker: Optional[Union[int, str]] = Field(
        None, description="The speaker ID of the phrase, if detected."
    )
    confidence: Optional[float] = Field(
        None, description="The confidence score of the phrase, if available."
    )
    detected_language: Optional[str] = Field(
        None, description="The detected language of the phrase, if available."
    )
    start_secs: float = Field(
        ..., description="The start time of the phrase in seconds."
    )
    duration_secs: float = Field(
        ..., description="The duration of the phrase in seconds."
    )
    end_secs: float = Field(..., description="The end time of the phrase in seconds.")
    words: List[TranscribedWord] = Field(
        ..., description="The list of individual words within the phrase, if available."
    )

    @computed_field(
        description="The minute in which the phrase started. E.g. the 67th second becomes '1'."
    )
    def start_min(self) -> int:
        return int(self.start_secs // 60)

    @computed_field(
        description="The sub-minute second of the phrase in minutes. E.g. the 67th second becomes '07'"
    )
    def start_sub_sec(self) -> str:
        return str(int(self.start_secs % 60)).zfill(2)

    @computed_field(
        description="Auto-generated speaker/channel ID based on the speaker ID or channel."
    )
    def auto_speaker_channel_id(self) -> str:
        if self.speaker is not None:
            return str(self.speaker)
        if self.channel is not None:
            return str(self.channel)
        return "Unknown"

    def to_formatted_str(
        self,
        format: str = "[{start_min}:{start_sub_sec}] Speaker {auto_speaker_channel_id}: {display_text}",
    ) -> str:
        """
        Converts the object to a formatted string using the provided format string.

        Args:
            format: The format string to use. For a full list of variables, run
                `model_field_info()` on the object.

        Usage examples:
        format = "[{start_min}:{start_sub_sec}] Speaker {auto_speaker_channel_id}: {display_text}"
        > [0:07] Speaker 0: Hello, how are you doing?

        format = "[{start_min}:{start_sub_sec}] Channel {channel}: {display_text} (Confidence: {confidence:.2f})"
        > [0:07] Channel 0: Hello, how are you doing? (Confidence: 0.95)
        """
        try:
            return format.format(**self.get_vars_dict())
        except KeyError as _e:
            raise ValueError(
                f"Variable {_e} does not exist - please use only variables that existing within the object: {list(self.get_vars_dict().keys())}"
            ) from None

    # def _validate_can_map_field(
    #     self, mapping_field: str, mapper: dict[Any, Any], on_error: str = "raise"
    # ) -> bool:

    #     is_mapping_field_valid = hasattr(self, mapping_field)
    #     if on_error == "raise" and not is_mapping_field_valid:
    #         raise ValueError(
    #             f"Mapping field '{mapping_field}' does not exist in the object. Please use one of: {list(self.get_vars_dict().keys())}"
    #         )
    #     existing_value = getattr(self, mapping_field)
    #     is_existing_value_in_mapper = existing_value in mapper.values()
    #     if on_error == "raise" and not is_existing_value_in_mapper:
    #         raise ValueError(
    #             f"Existing value of '{existing_value}' for '{mapping_field}' is not contained in value mapper. Please ensure all existing values can be mapped by."
    #         )
    #     return is_mapping_field_valid and is_existing_value_in_mapper

    # def map_field_to_speaker_id(
    #     self, mapping_field: str, mapper: dict[Any, Any]
    # ) -> None:
    #     """
    #     Updates the speaker ID based on a mapping dictionary.
    #     """
    #     self._validate_can_map_field(mapping_field, mapper, on_error="raise")
    #     self.speaker = mapper[getattr(self, mapping_field)]
    #     return


class PhraseIdMethod(Enum):
    SPEAKER = "speaker"
    CHANNEL = "channel"
    OFFSET_ID = "offset_id"


class Transcription(TranscriptionBaseModel):
    """
    Data object representing a full transcription.
    """

    transcription_type: AzureSpeechTranscriptionEnum = Field(
        ..., description="The type of transcription."
    )
    phrases: List[TranscribedPhrase] = Field(
        ..., description="The list of phrases in the transcription."
    )
    duration_secs: Optional[float] = Field(
        None, description="The duration of the transcription in seconds."
    )
    detected_language: Optional[str] = Field(
        None, description="The detected language of the transcription, if available."
    )
    raw_api_response: Optional[Dict | List] = Field(
        None, description="The raw API response from the transcription service."
    )

    @computed_field(
        description="The duration in a formatted version. E.g. the 127 seconds becomes '2:07'."
    )
    def formatted_duration(self) -> str:
        if self.duration_secs is None:
            return ""
        return "{}:{}".format(
            int(self.duration_secs // 60), str(int(self.duration_secs % 60)).zfill(2)
        )

    def to_formatted_str(
        self,
        transcription_prefix_format: str = "Detected Language: {detected_language}\nDuration: {formatted_duration} minutes\n\nConversation:\n",
        phrase_format: str = "[{start_min}:{start_sub_sec}] {auto_phrase_source_name} {auto_phrase_source_id}: {display_text}",
        auto_phrase_speaker_name: str = "Speaker",
        auto_phrase_channel_name: str = "Channel",
        auto_phrase_offset_id_name: str = "Phrase",
    ) -> str:
        """
        Converts the object to a formatted string using the provided format string.

        Args:
            transcription_prefix_format: The format string to use. For a full list of variables, run
                `.model_field_info()` on the object.
            phrase_format: The format string to use for each phrase. For a full list of variables, run
                `.model_field_info()` on a phrase object.
            auto_phrase_speaker_name: The name to use for the speaker ID in the phrase format. (e.g. "Speaker 0")
            auto_phrase_channel_name: The name to use for the channel ID in the phrase format. (e.g. "Channel 0")
            auto_phrase_offset_id_name: The name to use for the offset_id in the phrase format. (e.g. "Phrase 0")

        Usage examples:
        transcription_info_format: str = "Detected Language: {detected_language}\nDuration: {formatted_duration} minutes\n\nConversation:\n",
        phrase_format: str = "[{start_min}:{start_sub_sec}] Speaker {auto_speaker_channel_id}: {display_text}"
        > Detected Language: English
        > Duration: 1:07 minutes
        >
        > Conversation:
        > [0:02] Speaker 0: Hello, how are you doing?
        > [0:05] Speaker 1: I'm doing great, thank you.
        > [0:09] Speaker 0: How can I help you today?
        """
        transcription_info_str = transcription_prefix_format.format(
            **self.get_vars_dict()
        )
        # Determine the best method for auto-generating the speaker/channel ID
        auto_phrase_source_name = {
            PhraseIdMethod.SPEAKER: auto_phrase_speaker_name,
            PhraseIdMethod.CHANNEL: auto_phrase_channel_name,
            PhraseIdMethod.OFFSET_ID: auto_phrase_offset_id_name,
        }
        best_phrase_id_method, best_phrase_id_property = (
            self._determine_auto_phrase_id_format()
        )
        auto_phrase_source_name_str = auto_phrase_source_name[best_phrase_id_method]
        phrase_format = phrase_format.replace(
            "{auto_phrase_source_name}", auto_phrase_source_name_str
        )
        phrase_format = phrase_format.replace(
            "auto_phrase_source_id", best_phrase_id_property
        )
        phrases_str = "\n".join(
            [phrase.to_formatted_str(phrase_format) for phrase in self.phrases]
        )
        return transcription_info_str + phrases_str

    def _determine_auto_phrase_id_format(self) -> tuple[str, str]:
        """
        Determines the optimal format for the speaker ID in the phrase format.
        """
        speaker_ids = set(
            [
                str(phrase.speaker)
                for phrase in self.phrases
                if phrase.speaker is not None
            ]
        )
        if len(speaker_ids) > 1:
            return PhraseIdMethod.SPEAKER, "speaker"
        channel_ids = set(
            [
                str(phrase.channel)
                for phrase in self.phrases
                if phrase.channel is not None
            ]
        )
        if len(channel_ids) > 1:
            return PhraseIdMethod.CHANNEL, "channel"
        return PhraseIdMethod.OFFSET_ID, "offset_id"


# Set divisor when converting from ticks to seconds
TIME_DIVISORS = {
    AzureSpeechTranscriptionEnum.BATCH: 10000000,
    AzureSpeechTranscriptionEnum.REALTIME: 10000000,
    AzureSpeechTranscriptionEnum.FAST: 1000,
    AzureSpeechTranscriptionEnum.AOAI_WHISPER: 1,
}


def process_batch_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.BATCH]
    return TranscribedWord(
        offset_id=offset_id,
        text=word_dict["displayText"],
        confidence=word_dict.get(
            "confidence", None
        ),  # Only returned for mono audio input
        start_secs=word_dict["offsetInTicks"] / time_divisor,
        duration_secs=word_dict["durationInTicks"] / time_divisor,
        end_secs=(word_dict["offsetInTicks"] + word_dict["durationInTicks"])
        / time_divisor,
    )


def process_realtime_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.REALTIME]
    return TranscribedWord(
        offset_id=offset_id,
        text=word_dict["Word"],
        start_secs=word_dict["Offset"] / time_divisor,
        duration_secs=word_dict["Duration"] / time_divisor,
        end_secs=(word_dict["Offset"] + word_dict["Duration"]) / time_divisor,
    )


def process_fast_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.FAST]
    return TranscribedWord(
        offset_id=offset_id,
        text=word_dict["text"],
        start_secs=word_dict["offset"] / time_divisor,
        duration_secs=word_dict["duration"] / time_divisor,
        end_secs=(word_dict["offset"] + word_dict["duration"]) / time_divisor,
    )


def process_batch_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.BATCH]
    best_phrase = phrase_dict["nBest"][0]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=best_phrase["lexical"],
        itn_text=best_phrase["itn"],
        masked_itn_text=best_phrase["maskedITN"],
        display_text=best_phrase["display"],
        channel=phrase_dict["channel"],
        speaker=phrase_dict.get("speaker", None),
        detected_language=phrase_dict.get("locale", None),
        start_secs=phrase_dict["offsetInTicks"] / time_divisor,
        duration_secs=phrase_dict["durationInTicks"] / time_divisor,
        end_secs=(phrase_dict["offsetInTicks"] + phrase_dict["durationInTicks"])
        / time_divisor,
        confidence=best_phrase["confidence"],
        words=words,
    )


def process_realtime_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.REALTIME]
    best_phrase = phrase_dict["NBest"][0]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=best_phrase["Lexical"],
        itn_text=best_phrase["ITN"],
        masked_itn_text=best_phrase["MaskedITN"],
        display_text=phrase_dict["DisplayText"],
        channel=phrase_dict["Channel"],
        speaker=phrase_dict.get("SpeakerId", None),
        detected_language=phrase_dict.get("PrimaryLanguage", dict()).get(
            "Language", None
        ),
        start_secs=phrase_dict["Offset"] / time_divisor,
        duration_secs=phrase_dict["Duration"] / time_divisor,
        end_secs=(phrase_dict["Offset"] + phrase_dict["Duration"]) / time_divisor,
        confidence=best_phrase["Confidence"],
        words=words,
    )


def process_fast_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.FAST]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=None,
        itn_text=None,
        masked_itn_text=None,
        display_text=phrase_dict["text"],
        channel=phrase_dict["channel"],
        speaker=None,
        detected_language=phrase_dict.get("locale", None),
        start_secs=phrase_dict["offset"] / time_divisor,
        duration_secs=phrase_dict["duration"] / time_divisor,
        end_secs=(phrase_dict["offset"] + phrase_dict["duration"]) / time_divisor,
        confidence=phrase_dict["confidence"],
        words=words,
    )


def process_whisper_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.AOAI_WHISPER]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=None,
        itn_text=None,
        masked_itn_text=None,
        display_text=phrase_dict["text"],
        channel=0,
        speaker=None,
        detected_language=None,
        start_secs=phrase_dict["start"] / time_divisor,
        duration_secs=(phrase_dict["end"] - phrase_dict["start"]) / time_divisor,
        end_secs=phrase_dict["end"] / time_divisor,
        confidence=None,
        words=words,
    )


def process_batch_transcription(transcription: Dict) -> List[TranscribedPhrase]:
    word_count = 0
    # Phrases come sorted by channel -> offset. Sort by offset alone
    raw_phrases = sorted(
        transcription["recognizedPhrases"], key=lambda phrase: phrase["offsetInTicks"]
    )
    processed_phrases = list()
    for phrase_num, raw_phrase in enumerate(raw_phrases):
        raw_words = raw_phrase["nBest"][0]["displayWords"]
        processed_words = [
            process_batch_word_dict(raw_word, word_count + i)
            for i, raw_word in enumerate(raw_words)
        ]
        word_count += len(raw_words)
        processed_phrases.append(
            process_batch_phrase_dict(raw_phrase, phrase_num, processed_words)
        )
    return Transcription(
        phrases=processed_phrases,
        transcription_type=AzureSpeechTranscriptionEnum.BATCH,
        duration_secs=transcription["durationInTicks"]
        / TIME_DIVISORS[AzureSpeechTranscriptionEnum.BATCH],
        detected_language=None,
        raw_api_response=transcription,
    )


def process_realtime_transcription(transcription: Dict) -> List[TranscribedPhrase]:
    word_count = 0
    processed_phrases = list()
    for phrase_num, raw_phrase in enumerate(transcription):
        raw_words = raw_phrase["NBest"][0]["Words"]
        processed_words = [
            process_realtime_word_dict(raw_word, word_count + i)
            for i, raw_word in enumerate(raw_words)
        ]
        word_count += len(raw_words)
        processed_phrases.append(
            process_realtime_phrase_dict(raw_phrase, phrase_num, processed_words)
        )
    return Transcription(
        phrases=processed_phrases,
        transcription_type=AzureSpeechTranscriptionEnum.REALTIME,
        duration_secs=None,
        detected_language=None,
        raw_api_response=transcription,
    )


def process_fast_transcription(transcription: Dict) -> List[TranscribedPhrase]:
    word_count = 0
    # Phrases come sorted by channel -> offset. Sort by offset alone
    raw_phrases = sorted(transcription["phrases"], key=lambda phrase: phrase["offset"])
    processed_phrases = list()
    for phrase_num, raw_phrase in enumerate(raw_phrases):
        raw_words = raw_phrase["words"]
        processed_words = [
            process_fast_word_dict(raw_word, word_count + i)
            for i, raw_word in enumerate(raw_words)
        ]
        word_count += len(raw_words)
        processed_phrases.append(
            process_fast_phrase_dict(raw_phrase, phrase_num, processed_words)
        )
    return Transcription(
        phrases=processed_phrases,
        transcription_type=AzureSpeechTranscriptionEnum.FAST,
        duration_secs=round(
            transcription["duration"]
            / TIME_DIVISORS[AzureSpeechTranscriptionEnum.FAST],
            3,
        ),
        detected_language=None,
        raw_api_response=transcription,
    )


def process_whisper_transcription(transcription: Dict) -> List[TranscribedPhrase]:
    processed_phrases = list()
    for phrase_num, raw_phrase in enumerate(transcription["segments"]):
        # Whisper does not return word-level information
        processed_phrases.append(
            process_whisper_phrase_dict(raw_phrase, phrase_num, list())
        )
    return Transcription(
        phrases=processed_phrases,
        transcription_type=AzureSpeechTranscriptionEnum.AOAI_WHISPER,
        duration_secs=round(transcription["duration"], 3),
        detected_language=transcription.get("language", None),
        raw_api_response=transcription,
    )


class AzureSpeechTranscriber:

    def __init__(
        self,
        speech_key: str,
        speech_region: str,
        aoai_api_key: str = None,
        aoai_endpoint: str = None,
        aoai_whisper_deployment: str = None,
    ):
        self._speech_key = speech_key
        self._speech_region = speech_region
        self._aoai_api_key = aoai_api_key
        self._aoai_endpoint = aoai_endpoint
        self._aoai_whisper_deployment = aoai_whisper_deployment
        self._httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    def _get_speech_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self._speech_key,
        }

    def _get_aoai_whisper_headers(self) -> dict[str, str]:
        return {
            "api-key": self._aoai_api_key,
        }

    async def get_fast_transcription_async(
        self,
        audio_path: str,
        definition: Optional[Dict[str, Any]] = None,
    ) -> tuple[Transcription, dict]:
        url = f"https://{self._speech_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-05-15-preview"
        with open(audio_path, "rb") as audio_file:
            headers = self._get_speech_headers()
            files = {
                "audio": audio_file,
                "definition": (None, json.dumps(definition), "application/json"),
            }
            response = await self._httpx_client.post(
                url=url,
                # params=params,
                headers=headers,
                files=files,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    "Fast transcription failed with status code: {}. Reason: {}".format(
                        response.status_code,
                        response.text,
                    ),
                )
            return process_fast_transcription(response.json()), response.json()

    async def get_aoai_whisper_transcription_async(
        self,
        audio_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Transcription, dict]:
        url = os.path.join(
            self._aoai_endpoint,
            f"openai/deployments/{self._aoai_whisper_deployment}/audio/transcriptions",
        )
        headers = self._get_aoai_whisper_headers()
        with open(audio_path, "rb") as audio_file:
            params = {
                "api-version": "2024-06-01",
            }
            files = {
                "file": audio_file,
                "model": (None, model),
                "response_format": (None, "verbose_json"),
            }
            for key, value in {
                "language": language,
                "prompt": prompt,
                "temperature": temperature,
            }.items():
                if value is not None:
                    files[key] = (None, value)
            response = await self._httpx_client.post(
                url=url,
                params=params,
                headers=headers,
                files=files,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    "AOAI Whisper transcription failed with status code: {}. Reason: {}".format(
                        response.status_code,
                        response.text,
                    ),
                )
            return process_whisper_transcription(response.json()), response.json()

        # url = f"https://{self._speech_region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-05-15-preview"
        # with open(audio_path, "rb") as audio_file:
        #     files = {
        #         "audio": audio_file,
        #         "definition": (None, json.dumps(definition), "application/json"),
        #     }
        #     # data = aiohttp.FormData()
        #     # data.add_field("audio", audio_file, content_type="audio/wav")
        #     # data.add_field(
        #     #     "definition", json.dumps(definition), content_type="application/json"
        #     # )
        #     response = await client.post(url=url, headers=headers, files=files)
        #     return response

        #     async with self._session.post(url, headers=headers, data=data) as response:
        #         response_json = await response.json()
        #         print(response.status)
        #         return response_json
        #         return process_fast_transcription(response_json)

    # async def submit_batch_transcription_job_async(
    #     self,
    #     content_urls: List[str],
    #     display_name: Optional[str] = None,
    #     locale: Optional[str] = None,
    #     model: Optional[str] = None,
    #     properties: Optional[Dict[str, Any]] = None,
    # ) -> Transcription:
    #     url = f"https://{self._speech_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    #     headers = self._get_headers()
    #     payload = {
    #         "contentUrls": content_urls,
    #         "displayName": display_name or "Batch Transcription Job",
    #         "locale": locale,
    #         "model": model,
    #         "properties": properties,
    #     }
    #     response = await self._httpx_client.post(
    #         url=url,
    #         # params=params,
    #         headers=headers,
    #         json=payload,
    #     )
    #     return response
    #     if response.status_code != 200:
    #         raise RuntimeError(
    #             "Fast transcription failed with status code: {}. Reason: {}".format(
    #                 response.status_code,
    #                 response.text,
    #             ),
    #         )
    #     return process_batch_transcription(response.json()), response.json()

    #     async with self._session.post(url, headers=headers, json=payload) as response:
    #         response_json = await response.json()
    #         return response_json
    #         return process_fast_transcription(response_json)


# def TranscriptionToDocuments(
#     transcription: Transcription,
#     phrase_format: str = "[{start_min}:{start_sub_sec}] {auto_phrase_source_name} {auto_phrase_source_id}: {display_text}",
#     auto_phrase_speaker_name: str = "Speaker",
#     auto_phrase_channel_name: str = "Channel",
#     auto_phrase_offset_id_name: str = "Phrase",
# ) -> List[Document]:
#     """Converts a Transcription object to a list of Haystack Document objects."""
#     documents = list()
#     for phrase in Transcription.phrases:
#         documents.append(
#             Document(
#                 content=phrase.display_text,
#                 meta={
#                     "start_secs": phrase.start_secs,
#                     "end_secs": phrase.end_secs,
#                     "speaker": phrase.speaker,
#                     "channel": phrase.channel,
#                     "confidence": phrase.confidence,
#                     "detected_language": phrase.detected_language,
#                 },
#             )
#         )
