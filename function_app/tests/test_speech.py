import json
import os

import pytest
from src.components.speech import (
    AzureSpeechTranscriptionEnum,
    Transcription,
    process_aoai_whisper_transcription,
    process_batch_transcription,
    process_fast_transcription,
    process_realtime_transcription,
)

CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def aoai_whisper_raw_transcription_response() -> dict:
    with open(
        os.path.join(CWD, "data/speech/aoai_whisper_raw_response.json"), "rb"
    ) as f:
        return json.load(f)


@pytest.fixture
def batch_raw_transcription_response() -> dict:
    with open(os.path.join(CWD, "data/speech/batch_raw_response.json"), "rb") as f:
        return json.load(f)


@pytest.fixture
def batch_whisper_raw_transcription_response() -> dict:
    with open(
        os.path.join(CWD, "data/speech/batch_whisper_raw_response.json"), "rb"
    ) as f:
        return json.load(f)


@pytest.fixture
def fast_raw_transcription_response() -> dict:
    with open(os.path.join(CWD, "data/speech/fast_raw_response.json"), "rb") as f:
        return json.load(f)


@pytest.fixture
def realtime_raw_transcription_response() -> dict:
    with open(os.path.join(CWD, "data/speech/realtime_raw_response.json"), "rb") as f:
        return json.load(f)


def test_process_aoai_whisper_transcription(
    aoai_whisper_raw_transcription_response: dict,
):
    """
    A basic test to ensure that the AOAI whisper transcription API response is
    processed correctly and without failure.

    TODO: This test does not validate the individual values of the processed
    transcription. It is recommended to add more tests to validate all of the
    processed values.
    """
    raw_transcription_dict = aoai_whisper_raw_transcription_response
    raw_phrases = raw_transcription_dict["segments"]
    raw_words = raw_transcription_dict["words"]
    full_duration_secs = raw_transcription_dict["duration"]
    # Process the transcription
    processed_transcription = process_aoai_whisper_transcription(raw_transcription_dict)
    all_processed_words = [
        word for phrase in processed_transcription.phrases for word in phrase.words
    ]
    ### Checks
    # Check type of processed transcription
    assert isinstance(processed_transcription, Transcription)
    assert (
        processed_transcription.transcription_type
        == AzureSpeechTranscriptionEnum.AOAI_WHISPER
    )
    # Correct number of phrases and words
    assert len(processed_transcription.phrases) == len(raw_phrases)
    # TODO: Words are not implemented due to the need to realign word and phrase timestamps
    # (words within a phrase often begin before the phrase or end after the phrase)
    assert len(all_processed_words) == 0
    # All phrases and words have display_text/text at the very least (other versions are optional)
    assert all([phrase.display_text for phrase in processed_transcription.phrases])
    assert all([word.display_text for word in all_processed_words])
    ## Timestamps
    assert (
        pytest.approx(full_duration_secs, rel=0.1)
        == processed_transcription.duration_secs
    )
    # Between 0 and duration of transcription
    assert all([phrase.start_secs > 0 for phrase in processed_transcription.phrases])
    assert all(
        [
            phrase.start_secs <= full_duration_secs
            for phrase in processed_transcription.phrases
        ]
    )
    assert all(
        [
            phrase.start_secs <= phrase.end_secs
            for phrase in processed_transcription.phrases
        ]
    )

    assert all([word.start_secs > 0 for word in all_processed_words])
    assert all([word.start_secs <= full_duration_secs for word in all_processed_words])
    assert all([word.start_secs <= word.end_secs for word in all_processed_words])


@pytest.mark.parametrize(
    "transcription_fixture",
    [
        "batch_raw_transcription_response",
        "batch_whisper_raw_transcription_response",
    ],
)
def test_process_batch_transcription(transcription_fixture, request):
    """
    A basic test to ensure that the Azure AI Speech Batch transcription API
    response is processed correctly and without failure.

    TODO: This test does not validate the individual values of the processed
    transcription. It is recommended to add more tests to validate all of the
    processed values.
    """
    raw_transcription_dict = request.getfixturevalue(transcription_fixture)
    raw_phrases = raw_transcription_dict["recognizedPhrases"]
    raw_words = [
        word for phrase in raw_phrases for word in phrase["nBest"][0]["displayWords"]
    ]
    full_duration_secs = int(raw_transcription_dict["durationInTicks"]) / 10000000
    # Process the transcription
    processed_transcription = process_batch_transcription(raw_transcription_dict)
    all_processed_words = [
        word for phrase in processed_transcription.phrases for word in phrase.words
    ]
    ### Checks
    # Check type of processed transcription
    assert isinstance(processed_transcription, Transcription)
    assert (
        processed_transcription.transcription_type == AzureSpeechTranscriptionEnum.BATCH
    )
    # Correct number of phrases and words
    assert len(processed_transcription.phrases) == len(raw_phrases)
    assert len(all_processed_words) == len(raw_words)
    # All phrases and words have display_text/text at the very least (other versions are optional)
    assert all([phrase.display_text for phrase in processed_transcription.phrases])
    assert all([word.display_text for word in all_processed_words])
    ## Timestamps
    assert (
        pytest.approx(full_duration_secs, rel=0.1)
        == processed_transcription.duration_secs
    )
    # Between 0 and duration of transcription
    assert all([phrase.start_secs > 0 for phrase in processed_transcription.phrases])
    assert all(
        [
            phrase.start_secs <= full_duration_secs
            for phrase in processed_transcription.phrases
        ]
    )
    assert all(
        [
            phrase.start_secs <= phrase.end_secs
            for phrase in processed_transcription.phrases
        ]
    )

    assert all([word.start_secs > 0 for word in all_processed_words])
    assert all([word.start_secs <= full_duration_secs for word in all_processed_words])
    assert all([word.start_secs <= word.end_secs for word in all_processed_words])


def test_process_fast_transcription(fast_raw_transcription_response: dict):
    """
    A basic test to ensure that the Azure AI Speech Fast transcription API
    response is processed correctly and without failure.

    TODO: This test does not validate the individual values of the processed
    transcription. It is recommended to add more tests to validate all of the
    processed values.
    """
    raw_transcription_dict = fast_raw_transcription_response
    raw_phrases = raw_transcription_dict["phrases"]
    raw_words = [word for phrase in raw_phrases for word in phrase["words"]]
    full_duration_secs = int(raw_transcription_dict["duration"]) / 1000
    # Process the transcription
    processed_transcription = process_fast_transcription(raw_transcription_dict)
    all_processed_words = [
        word for phrase in processed_transcription.phrases for word in phrase.words
    ]
    print(raw_words[:5])
    print(all_processed_words[:5])
    ### Checks
    # Check type of processed transcription
    assert isinstance(processed_transcription, Transcription)
    assert (
        processed_transcription.transcription_type == AzureSpeechTranscriptionEnum.FAST
    )
    # Correct number of phrases and words
    assert len(processed_transcription.phrases) == len(raw_phrases)
    assert len(all_processed_words) == len(raw_words)
    # All phrases and words have display_text/text at the very least (other versions are optional)
    assert all([phrase.display_text for phrase in processed_transcription.phrases])
    assert all([word.display_text for word in all_processed_words])
    ## Timestamps
    assert (
        pytest.approx(full_duration_secs, rel=0.1)
        == processed_transcription.duration_secs
    )
    # Between 0 and duration of transcription
    assert all([phrase.start_secs > 0 for phrase in processed_transcription.phrases])
    assert all(
        [
            phrase.start_secs <= full_duration_secs
            for phrase in processed_transcription.phrases
        ]
    )
    assert all(
        [
            phrase.start_secs <= phrase.end_secs
            for phrase in processed_transcription.phrases
        ]
    )

    assert all([word.start_secs > 0 for word in all_processed_words])
    assert all([word.start_secs <= full_duration_secs for word in all_processed_words])
    assert all([word.start_secs <= word.end_secs for word in all_processed_words])


def test_process_realtime_transcription(realtime_raw_transcription_response: dict):
    """
    A basic test to ensure that the Azure AI Speech Real-time transcription API
    response is processed correctly and without failure.

    TODO: This test does not validate the individual values of the processed
    transcription. It is recommended to add more tests to validate all of the
    processed values.
    """
    raw_transcription_dict = realtime_raw_transcription_response
    raw_phrases = raw_transcription_dict
    raw_words = [word for phrase in raw_phrases for word in phrase["NBest"][0]["Words"]]
    full_duration_secs = (raw_transcription_dict[-1]["Offset"] / 10000000) + (
        raw_transcription_dict[-1]["Duration"] / 10000000
    )
    # Process the transcription
    processed_transcription = process_realtime_transcription(raw_transcription_dict)
    all_processed_words = [
        word for phrase in processed_transcription.phrases for word in phrase.words
    ]
    ### Checks
    # Check type of processed transcription
    assert isinstance(processed_transcription, Transcription)
    assert (
        processed_transcription.transcription_type
        == AzureSpeechTranscriptionEnum.REALTIME
    )
    # Correct number of phrases and words
    assert len(processed_transcription.phrases) == len(raw_phrases)
    assert len(all_processed_words) == len(raw_words)
    # All phrases and words have display_text/text at the very least (other versions are optional)
    assert all([phrase.display_text for phrase in processed_transcription.phrases])
    assert all([word.display_text for word in all_processed_words])
    ## Timestamps
    assert (
        pytest.approx(full_duration_secs, rel=0.1)
        == processed_transcription.duration_secs
    )
    # Between 0 and duration of transcription
    assert all([phrase.start_secs > 0 for phrase in processed_transcription.phrases])
    assert all(
        [
            phrase.start_secs <= full_duration_secs
            for phrase in processed_transcription.phrases
        ]
    )
    assert all(
        [
            phrase.start_secs <= phrase.end_secs
            for phrase in processed_transcription.phrases
        ]
    )

    assert all([word.start_secs > 0 for word in all_processed_words])
    assert all([word.start_secs <= full_duration_secs for word in all_processed_words])
    assert all([word.start_secs <= word.end_secs for word in all_processed_words])
