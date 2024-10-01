from time import perf_counter
from typing import Union

from haystack import Document
from haystack.dataclasses import ByteStream


def prep_dataclass_for_chat_message(
    haystack_data: Union[Document, ByteStream]
) -> Union[str, ByteStream]:
    """
    Converts a Haystack Document to the format expected by ChatMessage, which is
    either a string or a ByteStream object. This can be used to coerce other
    types (e.g. dataframe) to the expected format.

    :param haystack_data:
        The Haystack Document to convert.

    :return:
        A string or ByteStream object.
    """
    if isinstance(haystack_data, ByteStream):
        return haystack_data
    elif haystack_data.content is not None:
        return haystack_data.content
    elif haystack_data.dataframe is not None:
        haystack_data.dataframe.to_markdown()
    raise NotImplementedError("Document includes data of unknown type.")


def haystack_doc_to_string(doc: Document) -> str:
    """
    Converts a Haystack Document (which can contains different types of
    content) to a simple string.

    :param doc:
        The Haystack Document to convert.

    :return:
        A string representation of the document.
    """
    if doc.content is not None:
        return doc.content
    if doc.dataframe is not None:
        return doc.dataframe.to_markdown()
    raise ValueError("Document has no content or dataframe, cannot convert to string.")


def clean_openai_msg(message: dict) -> dict:
    """
    Cleans OpenAI messages prior to returning them via the API. This is useful
    for shortening long URLs in image messages.
    :param message:
        The message to clean.

    :return:
        A cleaned message.
    """
    new_msg = message.copy()
    if isinstance(new_msg["content"], list):
        for msg in new_msg["content"]:
            if msg["type"] == "image_url" and msg["image_url"]["url"].startswith(
                "data:image"
            ):
                msg["image_url"]["url"] = msg["image_url"]["url"][:100] + "..."
    return new_msg


class MeasureRunTime:
    """
    A helper class to record the time taken to run a block of code.

    1. Use this class inline:
    ```
    timer = MeasureRunTime(round_to=1)
    timer.start()
    time.sleep(1.23456789)
    time_taken = timer.stop()
    print(time_taken)
    ```
    Output: 1.2

    2. Use this class as a context manager:
    ```
    with MeasureRunTime(round_to=2) as timer:
        time.sleep(2.3456789)

    print(timer.time_taken)
    ```
    Output: 2.35

    """

    def __init__(self, round_to: int = 2):
        self._round_to = round_to

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.start_time = perf_counter()

    def stop(self) -> float:
        self.end_time = perf_counter()
        self.time_taken = round(self.end_time - self.start_time, self._round_to)
        return self.time_taken


def camel_to_snake(name: str) -> str:
    """Converts camel case to snake case."""
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")
