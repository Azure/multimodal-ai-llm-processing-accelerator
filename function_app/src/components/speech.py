import json
import os
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from httpx import AsyncClient, Timeout
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, computed_field

from ..result_enrichment.common import is_value_in_content


class AzureSpeechTranscriptionEnum(Enum):
    BATCH = "Batch"
    REALTIME = "Realtime"
    FAST = "Fast"
    AOAI_WHISPER = "Azure OpenAI Whisper"


# Maps valid content types to file extensions for the Azure OpenAI Whisper API
# https://platform.openai.com/docs/guides/speech-to-text
AOAI_WHISPER_MIME_TYPE_MAPPER = {
    "audio/mpeg": "mp3",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/webm": "webm",
}

# Maps valid content types to file extensions for the Azure Batch Transcription API
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription-audio-data?tabs=portal#supported-audio-formats-and-codecs
BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wave",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/wma": "wma",
    "audio/x-ms-wma": "wma",
    "audio/aac": "aac",
    "audio/amr": "amr",
    "audio/webm": "webm",
    "audio/m4a": "m4a",
    "audio/speex": "ogg",
    "audio/x-speex": "ogg",
}


def get_type_from_pydantic_type_dict(type_dict: dict[str, str]) -> str:
    """
    Gets the object type from a Pydantic type dictionary.

    :param type_dict: The Pydantic type dictionary.
    :type type_dict: dict[str, str]
    :return: The object type.
    :rtype: str
    """
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
                    get_type_from_pydantic_type_dict(t) for t in info_dict["anyOf"]
                )
            elif "allOf" in info_dict:
                type = " & ".join(
                    get_type_from_pydantic_type_dict(t) for t in info_dict["allOf"]
                )
            elif "type" in info_dict:
                type = get_type_from_pydantic_type_dict(info_dict)
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
    display_text: str = Field(..., description="The display text of the word.")
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
    channel: Optional[int] = Field(
        None,
        description="The audio channel from which the phrase was spoken, if available.",
    )
    speaker: Optional[Union[int, str]] = Field(
        None, description="The speaker ID of the phrase, if detected."
    )
    confidence: Optional[float] = Field(
        None, description="The confidence score of the phrase, if available."
    )
    language: Optional[str] = Field(
        None, description="The language of the phrase, if available."
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
    language: Optional[str] = Field(
        None, description="The language of the transcription, if available."
    )
    raw_api_response: Optional[Dict | List] = Field(
        None, description="The raw API response from the transcription service."
    )

    @computed_field(
        description="The duration in a formatted version. E.g. the 127 seconds becomes '2:07'."
    )
    def formatted_duration(self) -> str:
        """
        Returns the transcription duration as a formatted string minute:second
        (e.g. '2:15').
        """
        if self.duration_secs is None:
            return ""
        return "{}:{}".format(
            int(self.duration_secs // 60), str(int(self.duration_secs % 60)).zfill(2)
        )

    def to_formatted_str(
        self,
        transcription_prefix_format: str = "Language: {language}\nDuration: {formatted_duration} minutes\n\nConversation:\n",
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
        transcription_info_format: str = "Language: {language}\nDuration: {formatted_duration} minutes\n\nConversation:\n",
        phrase_format: str = "[{start_min}:{start_sub_sec}] Speaker {auto_speaker_channel_id}: {display_text}"
        > Language: English
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


def is_text_in_phrase_display_text(text: str, phrase: TranscribedPhrase) -> bool:
    """
    Checks if a text is contained within a phrase's display_text field.

    :param text: The text to check.
    :type text: str
    :param phrase: The phrase to check within.
    :type phrase: TranscribedPhrase
    :return: Whether the text is contained within the phrase.
    :rtype: bool
    """
    return is_value_in_content(text, phrase.display_text)


def is_phrase_start_time_match(
    expected_start_time_secs: int | float,
    phrase: TranscribedPhrase,
    start_time_tolerance_secs: int | float = 0,
) -> bool:
    """
    Checks if the start time of the phrase matches the expected start time.

    :param expected_start_time_secs: The expected start time in seconds.
    :type expected_start_time_secs: int | float
    :param phrase: The phrase to check
    :type phrase: TranscribedPhrase
    :param start_time_tolerance_secs: The tolerance (in seconds) for matching
        the start time. defaults to 0
    :type start_time_tolerance_secs: int | float, optional
    :return: Whether the phrase start time matches the expected start time.
    :rtype: bool
    """
    return (
        expected_start_time_secs - start_time_tolerance_secs
        <= phrase.start_secs
        <= expected_start_time_secs + start_time_tolerance_secs
    )


# Set divisor when converting from ticks to seconds
TIME_DIVISORS = {
    AzureSpeechTranscriptionEnum.BATCH: 10000000,
    AzureSpeechTranscriptionEnum.REALTIME: 10000000,
    AzureSpeechTranscriptionEnum.FAST: 1000,
    AzureSpeechTranscriptionEnum.AOAI_WHISPER: 1,
}


def process_batch_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    """
    Processes a word dictionary from the Azure AI Speech Batch Transcription
    API.

    :param word_dict: The word dictionary to process.
    :type word_dict: Dict
    :param offset_id: The ID of the word, offset from the start of the
        transcription.
    :type offset_id: int
    :return: A processed TranscribedWord object.
    :rtype: TranscribedWord
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.BATCH]
    return TranscribedWord(
        offset_id=offset_id,
        display_text=word_dict["displayText"],
        confidence=word_dict.get(
            "confidence", None
        ),  # Only returned for mono audio input
        start_secs=word_dict["offsetInTicks"] / time_divisor,
        duration_secs=word_dict["durationInTicks"] / time_divisor,
        end_secs=(word_dict["offsetInTicks"] + word_dict["durationInTicks"])
        / time_divisor,
    )


def process_realtime_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    """
    Processes a word dictionary from the Azure AI Speech Real-time Transcription
     API.

    :param word_dict: The word dictionary to process.
    :type word_dict: Dict
    :param offset_id: The ID of the word, offset from the start of the
        transcription.
    :type offset_id: int
    :return: A processed TranscribedWord object.
    :rtype: TranscribedWord
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.REALTIME]
    return TranscribedWord(
        offset_id=offset_id,
        display_text=word_dict["Word"],
        start_secs=word_dict["Offset"] / time_divisor,
        duration_secs=word_dict["Duration"] / time_divisor,
        end_secs=(word_dict["Offset"] + word_dict["Duration"]) / time_divisor,
    )


def process_fast_word_dict(word_dict: Dict, offset_id: int) -> TranscribedWord:
    """
    Processes a word dictionary from the Azure AI Speech Fast Transcription API

    :param word_dict: The word dictionary to process.
    :type word_dict: Dict
    :param offset_id: The ID of the word, offset from the start of the
        transcription.
    :type offset_id: int
    :return: A processed TranscribedWord object.
    :rtype: TranscribedWord
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.FAST]
    return TranscribedWord(
        offset_id=offset_id,
        display_text=word_dict["text"],
        start_secs=word_dict["offset"] / time_divisor,
        duration_secs=word_dict["duration"] / time_divisor,
        end_secs=(word_dict["offset"] + word_dict["duration"]) / time_divisor,
    )


def process_batch_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    """
    Processes a phrase dictionary from the Azure AI Speech Batch Transcription
    API.

    :param phrase_dict: The phrase dictionary to process.
    :type phrase_dict: Dict
    :param offset_id: The ID of the phrase, offset from the start of the
        transcription.
    :type offset_id: int
    :param words: The list of individual words within the phrase.
    :type words: List[TranscribedWord]
    :return: A processed TranscribedPhrase object.
    :rtype: TranscribedPhrase
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.BATCH]
    best_phrase = phrase_dict["nBest"][0]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=best_phrase.get("lexical", None),
        itn_text=best_phrase.get("itn", None),
        masked_itn_text=best_phrase["maskedITN"],
        display_text=best_phrase["display"],
        channel=phrase_dict.get("channel", None),
        speaker=phrase_dict.get("speaker", None),
        language=phrase_dict.get("locale", None),
        start_secs=phrase_dict["offsetInTicks"] / time_divisor,
        duration_secs=phrase_dict["durationInTicks"] / time_divisor,
        end_secs=(
            (phrase_dict["offsetInTicks"] + phrase_dict["durationInTicks"])
            / time_divisor
        ),
        confidence=best_phrase.get("confidence", None),
        words=words,
    )


def process_realtime_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    """
    Processes a phrase dictionary from the Azure AI Speech Real-time
    Transcription API.

    :param phrase_dict: The phrase dictionary to process.
    :type phrase_dict: Dict
    :param offset_id: The ID of the phrase, offset from the start of the
        transcription.
    :type offset_id: int
    :param words: The list of individual words within the phrase.
    :type words: List[TranscribedWord]
    :return: A processed TranscribedPhrase object.
    :rtype: TranscribedPhrase
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.REALTIME]
    best_phrase = phrase_dict["NBest"][0]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=best_phrase.get("Lexical", None),
        itn_text=best_phrase.get("ITN", None),
        masked_itn_text=best_phrase.get("MaskedITN", None),
        display_text=phrase_dict["DisplayText"],
        channel=phrase_dict.get("Channel", None),
        speaker=phrase_dict.get("SpeakerId", None),
        language=phrase_dict.get("PrimaryLanguage", dict()).get("Language", None),
        start_secs=phrase_dict["Offset"] / time_divisor,
        duration_secs=phrase_dict["Duration"] / time_divisor,
        end_secs=(phrase_dict["Offset"] + phrase_dict["Duration"]) / time_divisor,
        confidence=best_phrase["Confidence"],
        words=words,
    )


def process_fast_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    """
    Processes a phrase dictionary from the Azure AI Speech Fast Transcription
    API.

    :param phrase_dict: The phrase dictionary to process.
    :type phrase_dict: Dict
    :param offset_id: The ID of the phrase, offset from the start of the
        transcription.
    :type offset_id: int
    :param words: The list of individual words within the phrase.
    :type words: List[TranscribedWord]
    :return: A processed TranscribedPhrase object.
    :rtype: TranscribedPhrase
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.FAST]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=None,
        itn_text=None,
        masked_itn_text=None,
        display_text=phrase_dict["text"],
        channel=phrase_dict.get("channel", None),
        speaker=phrase_dict.get("speaker", None),
        language=phrase_dict.get("locale", None),
        start_secs=phrase_dict["offset"] / time_divisor,
        duration_secs=phrase_dict["duration"] / time_divisor,
        end_secs=(phrase_dict["offset"] + phrase_dict["duration"]) / time_divisor,
        confidence=phrase_dict["confidence"],
        words=words,
    )


def process_whisper_phrase_dict(
    phrase_dict: Dict, offset_id: int, words: List[TranscribedWord]
) -> TranscribedPhrase:
    """
    Processes a phrase dictionary from the Azure OpenAI Whisper
    Transcription API.

    :param phrase_dict: The phrase dictionary to process.
    :type phrase_dict: Dict
    :param offset_id: The ID of the phrase, offset from the start of the
        transcription.
    :type offset_id: int
    :param words: The list of individual words within the phrase.
    :type words: List[TranscribedWord]
    :return: A processed TranscribedPhrase object.
    :rtype: TranscribedPhrase
    """
    time_divisor = TIME_DIVISORS[AzureSpeechTranscriptionEnum.AOAI_WHISPER]
    return TranscribedPhrase(
        offset_id=offset_id,
        lexical_text=None,
        itn_text=None,
        masked_itn_text=None,
        display_text=phrase_dict["text"],
        channel=phrase_dict.get("channel", None),
        speaker=phrase_dict.get("speaker", None),
        language=None,
        start_secs=phrase_dict["start"] / time_divisor,
        duration_secs=(phrase_dict["end"] - phrase_dict["start"]) / time_divisor,
        end_secs=phrase_dict["end"] / time_divisor,
        confidence=None,
        words=words,
    )


def process_batch_transcription(transcription: Dict) -> Transcription:
    """
    Processes a transcription response from the Azure AI Speech Batch
    Transcription API.

    :param transcription: The raw API response from Azure AI Speech API.
    :type transcription: Dict
    :return: A processed Transcription object.
    :rtype: Transcription
    """
    word_count = 0
    # Phrases come sorted by channel -> offset. Sort by offset alone
    raw_phrases = sorted(
        transcription["recognizedPhrases"], key=lambda phrase: phrase["offsetInTicks"]
    )
    processed_phrases: list[TranscribedPhrase] = list()
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
        language=None,
        raw_api_response=transcription,
    )


def process_realtime_transcription(transcription: Dict) -> Transcription:
    """
    Processes a transcription response from the Azure AI Speech Real-time
    Transcription API.

    :param transcription: The raw API response from Azure AI Speech API.
    :type transcription: Dict
    :return: A processed Transcription object.
    :rtype: Transcription
    """
    word_count = 0
    processed_phrases: list[TranscribedPhrase] = list()
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
        duration_secs=processed_phrases[-1].end_secs,
        language=None,
        raw_api_response=transcription,
    )


def process_fast_transcription(transcription: Dict) -> Transcription:
    """
    Processes a transcription response from the Azure AI Speech Fast
    Transcription API.

    :param transcription: The raw API response from Azure AI Speech API.
    :type transcription: Dict
    :return: A processed Transcription object.
    :rtype: Transcription
    """
    word_count = 0
    # Phrases come sorted by channel -> offset. Sort by offset alone
    raw_phrases = sorted(transcription["phrases"], key=lambda phrase: phrase["offset"])
    processed_phrases: list[TranscribedPhrase] = list()
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
        language=None,
        raw_api_response=transcription,
    )


def process_aoai_whisper_transcription(transcription: Dict) -> Transcription:
    """
    Processes a transcription response from Azure OpenAI Whisper.

    :param transcription: The raw API response from Azure OpenAI Whisper.
    :type transcription: Dict
    :return: A processed Transcription object.
    :rtype: Transcription
    """
    processed_phrases: list[TranscribedPhrase] = list()
    for phrase_num, raw_phrase in enumerate(transcription["segments"]):
        # TODO: Align phrase and segment values
        processed_phrases.append(
            process_whisper_phrase_dict(raw_phrase, phrase_num, list())
        )
    return Transcription(
        phrases=processed_phrases,
        transcription_type=AzureSpeechTranscriptionEnum.AOAI_WHISPER,
        duration_secs=round(transcription["duration"], 3),
        language=transcription.get("language", None),
        raw_api_response=transcription,
    )


class AzureSpeechTranscriber:
    """
    A component for transcribing audio files using Azure AI Speech and OpenAI
    Services. This component handles running of transcription jobs and
    processing the raw API responses into a more usable format.

    :param speech_key: The Azure Speech API key.
    :type speech_key: str
    :param speech_region: The Azure Speech API resource region/location.
    :type speech_endpoint: str
    :param aoai_whisper_async_client: An optional Azure OpenAI client to use for
        making requests This is required in order to get transcriptions using
        Azure OpenAI Whisper deployments. Defaults to None.
    :type aoai_whisper_async_client: Optional[AsyncAzureOpenAI], optional
    :param httpx_async_client: An optional HTTPX AsyncClient object to use for
        making requests. Defaults to None.
    :type httpx_async_client: Optional[AsyncClient], optional
    """

    def __init__(
        self,
        speech_key: str,
        speech_region: str,
        aoai_whisper_async_client: Optional[AsyncAzureOpenAI] = None,
        httpx_async_client: Optional[AsyncClient] = None,
    ):
        self._speech_key = speech_key
        self._speech_region = speech_region
        self._aoai_whisper_async_client = aoai_whisper_async_client
        self._httpx_async_client = httpx_async_client or AsyncClient(
            timeout=Timeout(60)
        )

    def _validate_whisper_client_initialized(self) -> None:
        """Validates that the Azure OpenAI client is initialized."""
        if self._aoai_whisper_async_client is None:
            raise ValueError(
                "Azure OpenAI client is not initialized. Please provide 'aoai_whisper_async_client' when initializing the transcriber."
            )

    def _get_speech_headers(self) -> dict[str, str]:
        """Gets the headers for the Azure Speech API."""
        return {
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": self._speech_key,
        }

    def _get_regional_endpoint_url(self) -> str:
        """Gets the regional endpoint URL for the Azure Speech API."""
        return f"https://{self._speech_region}.api.cognitive.microsoft.com"

    async def get_fast_transcription_async(
        self,
        audio_file: Union[bytes, BytesIO],
        definition: Optional[Dict[str, Any]] = None,
    ) -> tuple[Transcription, dict]:
        """
        Submits and returns a transcription of an audio file using the Azure AI
        Speech's Fast Transcription API. More information on the API can be
        found here: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create

        :param audio_file: _description_
        :type audio_file: Union[bytes, BytesIO]
        :param definition: _description_, defaults to None
        :type definition: Optional[Dict[str, Any]], optional
        :raises RuntimeError: Raised in case an error occured during the request.
        :return: A tuple of the processed transcription object and the raw API
            response.
        :rtype: tuple[Transcription, dict]
        """

        url = os.path.join(
            self._get_regional_endpoint_url(),
            "speechtotext/transcriptions:transcribe?api-version=2024-05-15-preview",
        )
        headers = self._get_speech_headers()
        response = await self._httpx_async_client.post(
            url=url,
            headers=headers,
            files={
                "audio": audio_file,
                "definition": (None, json.dumps(definition), "application/json"),
            },
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
        audio_file: Union[str, os.PathLike, BytesIO] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        model: str = "whisper-1",
        response_format: str = "verbose_json",
        **kwargs,
    ) -> tuple[Transcription, dict]:
        """
        Gets a transcription of an audio file using OpenAI's Whisper model using
        the Azure OpenAI resource. More information on model parameters can be
        found at: https://platform.openai.com/docs/guides/speech-to-text

        :param audio_file: Audio path or file buffer to be transcribed. Defaults
            to None
        :type audio_file: Union[str, os.PathLike, BytesIO], optional
        :param language: Language of the recording. Providing this can reduce
            the latency of the request. Defaults to None
        :type language: Optional[str], optional
        :param prompt: A prompt to be used to guide the model. This can be used
            to help the model understand certain words and phrases, or to guide
            the style of the transcription. Defaults to None
        :type prompt: Optional[str], optional
        :param temperature: Model temperature parameter. Defaults to None
        :type temperature: Optional[float], optional
        :param timeout: Timeout (in seconds) before the request is aborted.
            Defaults to None
        :type timeout: Optional[float], optional
        :param model: Model name, defaults to "whisper-1"
        :type model: str, optional
        :param response_format: JSON format of the response, defaults to
            "verbose_json"
        :type response_format: str, optional
        :return: Tuple of the processed transcription and the raw API response.
        :rtype: tuple[Transcription, dict]
        """
        self._validate_whisper_client_initialized()
        result = await self._aoai_whisper_async_client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timeout=timeout,
            timestamp_granularities=[
                "segment"
            ],  # TODO: Update once processing of words is supported.
            **kwargs,
        )
        return process_aoai_whisper_transcription(result.dict()), result.dict()
