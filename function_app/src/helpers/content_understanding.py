import json
import logging
import tempfile
from copy import copy
from types import NoneType
from typing import Literal, Optional, Union

import cv2
import numpy as np
import webvtt
from PIL import Image as PILImage
from PIL.Image import Image

from .image import (
    draw_polygon_on_pil_img,
    flat_poly_list_to_poly_dict_list,
    scale_flat_poly_list,
)


def source_coord_str_to_page_and_flat_poly_list(
    source_str: str,
) -> tuple[int, list[float]]:
    """
    Process a source string to a page number and flat polygon list.

    :param source_str:
        The source polygon string. Usually of format 'D(page_num, x1, y1, x2, y2, ...)'.

    :return:
        Tuple of page number (1-index) and a flat polygon list.
    """
    # Remove the 'D(' and ')' from the string
    source_str = source_str[2:-1]
    # Split the string into a list of strings
    source_str_list = source_str.split(",")
    # Convert the list of strings to a list of floats
    page_num = int(source_str_list[0])
    poly_coordinates = [float(x) for x in source_str_list[1:]]
    return page_num, poly_coordinates


def get_page_size(
    result_contents: dict[str, dict], page_num: int
) -> tuple[float, float]:
    """
    Get the page width and height from the result contents.

    :param result_contents:
        The result contents.

    :return:
        The page size (width, height).
    """
    matching_page_infos = [
        page for page in result_contents["pages"] if page["pageNumber"] == page_num
    ]
    if len(matching_page_infos) != 1:
        raise ValueError(
            f"Expected exactly one page info for page {page_num}, but got {len(matching_page_infos)}"
        )
    page_info = matching_page_infos[0]
    return page_info["width"], page_info["height"]


def _enrich_extracted_cu_field(
    field_info: tuple[dict, list[dict]], result_contents: dict[str, dict]
) -> tuple[dict, list[dict]]:
    """
    Enrich extracted content understanding field information.

    :param field_info:
        The extracted field information.

    :return:
        The enriched field information.
    """
    enriched_field_info = copy(field_info)
    if field_info["type"] == "object":
        for field_name, object_field_info in field_info.get(
            "valueObject", dict()
        ).items():
            enriched_field_info["valueObject"][field_name] = _enrich_extracted_cu_field(
                object_field_info, result_contents
            )
    elif field_info["type"] == "array":
        for idx, array_item in enumerate(field_info.get("valueArray", list())):
            enriched_field_info["valueArray"][idx] = _enrich_extracted_cu_field(
                array_item, result_contents
            )
    else:
        if "source" in field_info:
            # Process the source string to get the page number and flat polygon list for each source polygon
            enriched_field_info["source_polygons"] = list()
            if field_info.get("source") is not None:
                for source_str in field_info["source"].split(";"):
                    page_num, page_based_poly_list = (
                        source_coord_str_to_page_and_flat_poly_list(source_str)
                    )
                    page_wh = get_page_size(result_contents, page_num)
                    normalized_flat_poly_list = scale_flat_poly_list(
                        page_based_poly_list,
                        existing_scale=page_wh,
                        new_scale=(1, 1),
                    )
                    polygon_info = {
                        "page_number": page_num,
                        "normalized_polygon": normalized_flat_poly_list,
                    }
                    enriched_field_info["source_polygons"].append(polygon_info)
    return enriched_field_info


def enrich_extracted_cu_fields(result_contents: dict[str, dict]) -> dict[str, dict]:
    """
    Enrich extracted content understanding fields.

    :param fields:
        The extracted fields.

    :return:
        The enriched fields.
    """
    result_contents = copy(result_contents)
    return {
        field_name: _enrich_extracted_cu_field(field_info, result_contents)
        for field_name, field_info in result_contents["fields"].items()
    }


def _draw_field_on_imgs(
    enriched_field_info: tuple[dict, list[dict]],
    pil_imgs: dict[int, Image],
    outline_color: Union[str, tuple[int, int, int]],
    outline_width: int,
) -> dict[int, Image]:
    """
    Draw field outlines on PIL images (inplace).

    :param field_info:
        The field information as extracted by Azure Content Understanding and
            enriched by the helpers in this repo.
    :param pil_imgs:
        A dictionary mapping page number to PIL image.
    :param outline_color:
        The color of the outline.
    :param outline_width:
        The width of the outline.

    :return:
        The image dictionary with all fields drawn in place.
    """
    if enriched_field_info["type"] == "object":
        for object_field_info in enriched_field_info.get("valueObject", {}).values():
            pil_imgs = _draw_field_on_imgs(
                object_field_info, pil_imgs, outline_color, outline_width
            )
    elif enriched_field_info["type"] == "array":
        for array_item in enriched_field_info.get("valueArray", []):
            pil_imgs = _draw_field_on_imgs(
                array_item, pil_imgs, outline_color, outline_width
            )
    else:
        for source_polygon in enriched_field_info.get("source_polygons", []):
            # Scale the normalized polygon from 0->1 to the PIL image size
            pixel_based_flat_poly_list = scale_flat_poly_list(
                source_polygon["normalized_polygon"],
                existing_scale=(1, 1),
                new_scale=pil_imgs[source_polygon["page_number"]].size,
            )
            pixel_based_polygon_dict = flat_poly_list_to_poly_dict_list(
                pixel_based_flat_poly_list
            )
            pil_imgs[source_polygon["page_number"]] = draw_polygon_on_pil_img(
                pil_img=pil_imgs[source_polygon["page_number"]],
                polygon=pixel_based_polygon_dict,
                outline_color=outline_color,
                outline_width=outline_width,
            )
    return pil_imgs


def draw_fields_on_imgs(
    enriched_fields: dict[str, dict],
    pil_imgs: dict[int, Image],
    outline_color: Union[str, tuple[int, int, int]] = "blue",
    outline_width: int = 3,
) -> dict[int, Image]:
    """
    Draw field outlines on PIL images (inplace).

    :param enriched_fields:
        The enriched fields.
    :param pil_imgs:
        The PIL images.

    :return:
        The images with fields drawn in place.
    """
    for _field_name, field_info in enriched_fields.items():
        pil_imgs = _draw_field_on_imgs(
            field_info, pil_imgs, outline_color, outline_width
        )
    return pil_imgs


def condense_webvtt_transcription(webvtt_str: str) -> str:
    """
    Condense a WEBVTT transcription into a more readable format.

    :param transcription_md:
        The transcription in markdown format.

    :return:
        The condensed transcription string.
    """
    processed_captions = []
    for caption in webvtt.from_string(webvtt_str):
        _start_hour, start_min, start_sec = caption.start.split(":")
        start_sec = start_sec.split(".")[0]
        processed_captions.append(
            f"[{start_min}:{start_sec}] {caption.voice}: {caption.text}"
        )
    return "\n".join(processed_captions)


def extract_frames_from_video_bytes(
    video_bytes: bytes,
    frame_times_ms: list[int],
    img_type: Literal["pil", "cv2"] = "pil",
) -> dict[int, Union[np.ndarray, NoneType]]:
    """
    Extract frames from a video in memory at specified times.

    :param video_bytes: Video file in bytes.
    :param frame_times_ms: List of times in milliseconds to extract frames.
    :return: List of extracted frames.
    """
    # OpenCV does not support reading from bytes - use a temporary file
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(video_bytes)

        video_capture = cv2.VideoCapture(temp.name)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    time_to_frame_mapper = {}

    for time_ms in frame_times_ms:
        frame_number = int((time_ms / 1000) * fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video_capture.read()
        if success:
            if img_type == "pil":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = PILImage.fromarray(frame)
            elif img_type == "cv2":
                # frame is already in cv2 format
                pass
        else:
            logging.exception(f"Failed to extract frame at {time_ms} ms")
            frame = None
        time_to_frame_mapper[time_ms] = frame

    video_capture.release()
    return time_to_frame_mapper


def get_cu_value_field_name(field_info: dict) -> str:
    """Gets the name of the dictionary field containing the resulting value
    from a content understanding field result."""
    return next(
        (field_name for field_name in field_info if field_name.startswith("value")),
        None,
    )


def format_extracted_field_output(
    text: Optional[str],
    confidence: Optional[float] = None,
    null_text_fill_value: str = "\<NO VALUE\>",
) -> str:
    """
    Formats the text and confidence score (if available) of a Content
    Understanding field to keep only the field. Returns the result as a string.

    :param text: The extracted text value.
    :type text: Optional[str]
    :param confidence: The confidence score of the extracted text value.
    :type confidence: Optional[float]
    :param null_text_fill_value: The value to use if the extracted text is None.
        This should be a markdown-compatible string, defaults to "\<NO VALUE\>".
    :type null_text_fill_value: str
    """
    formatted_text = f"'{text}'" if text is not None else null_text_fill_value
    if confidence is None:
        return formatted_text
    return f"{formatted_text} [Confidence: {round(confidence * 100, 1)}%]"


def format_simple_cu_field_output(
    field_info: dict, include_confidence: bool = True
) -> str:
    """
    Formats the output of a simple Content Understanding field (fields other
    than array or object) to keep only the field value and confidence score.
    Returns the result as a string.

    :param field_info: Content Understanding field information.
    :type field_info: dict
    :param include_confidence: Whether to include confidence score information
        in the result (if available), defaults to True
    :type include_confidence: bool, optional
    :return: Formatted field output.
    :rtype: str
    """
    field_name = get_cu_value_field_name(field_info)
    confidence = field_info.get("confidence")
    return format_extracted_field_output(
        field_info.get(field_name),
        confidence if (confidence is not None and include_confidence) else None,
    )


def apply_md_text_formatting(text: str, bold: bool) -> str:
    """Applies Markdown formatting to a given string"""
    return "**{}**".format(text) if bold else text


def simplify_cu_field_dict(field_info: dict) -> dict:
    """
    Simplifies a Content Understanding field dictionary to keep only the field
    values and confidence scores. Returns the result as a dictionary.

    :param field_info: Content Understanding field information.
    :type field_info: dict
    :return: Simplified field dictionary.
    :rtype: dict
    """
    if field_info["type"] == "object":
        return {
            apply_md_text_formatting(k, True): simplify_cu_field_dict(v)
            for k, v in field_info.get("valueObject", {}).items()
        }
    elif field_info["type"] == "array":
        return [simplify_cu_field_dict(v) for v in field_info.get("valueArray", [])]
    else:
        field_name = get_cu_value_field_name(field_info)
        return format_extracted_field_output(
            field_info.get(field_name), field_info.get("confidence")
        )


def cu_fields_dict_to_markdown(fields_dict: dict) -> str:
    """
    Converts a dictionary of Content Understanding fields to a
    markdown-formatted string.

    :param cu_fields_dict: Dictionary of Content Understanding fields.
    :type cu_fields_dict: dict
    :return: Markdown-formatted string.
    :rtype: str
    """
    markdown_lines = []
    for field_name, field_info in fields_dict.items():
        if field_info.get("type") == "object":
            markdown_lines.append(
                f"{apply_md_text_formatting(field_name, True)}: {json.dumps(simplify_cu_field_dict(field_info), indent=4)}"
            )
        elif field_info.get("type") == "array":
            if field_info.get("valueArray", []) and field_info["valueArray"][0][
                "type"
            ] not in [
                "object",
                "array",
            ]:
                value_str = [
                    format_simple_cu_field_output(item, True)
                    for item in field_info.get("valueArray", [])
                ]
                markdown_lines.append(
                    f"{apply_md_text_formatting(field_name, True)}: {value_str}"
                )
            else:
                markdown_lines.append(
                    f"{apply_md_text_formatting(field_name, True)}: {json.dumps(simplify_cu_field_dict(field_info), indent=4)}"
                )
        else:
            formatted_output = format_simple_cu_field_output(field_info, True)
            if formatted_output.startswith("'|") and formatted_output.endswith("|'"):
                # Output is a table - add a newline before and after
                formatted_output = f"\n{formatted_output[1:-1]}\n"
            markdown_lines.append(
                f"{apply_md_text_formatting(field_name, True)}: {formatted_output}"
            )
    return "\n".join(markdown_lines)
