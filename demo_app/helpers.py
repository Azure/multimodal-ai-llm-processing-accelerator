import base64
import logging
import tempfile
from io import BytesIO
from types import NoneType
from typing import Literal, Optional, Union

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage


def pil_img_to_base64_bytes(pil_img: PILImage) -> str:
    """
    Converts a PIL image to a base64 encoded string.

    :param pil_img: The PIL image to convert.
    :type pil_img: PIL.Image.Image
    :return: A base64 encoded string.
    :rtype: str
    """
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


def base64_img_str_to_url(base64_img_str: str, mime_type: str) -> str:
    """
    Converts a base64 image string into a data URL.

    :param base64_img_str: A base64 image string.
    :type base64_img_str: str
    :param mime_type: The MIME type of the image.
    :type mime_type: str
    :return: A data URL for the image.
    :rtype: str
    """
    return f"data:{mime_type};base64,{base64_img_str}"


def resize_img_by_max(
    pil_img: PILImage,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> PILImage:
    """
    Resizes a PIL image to fit within the specified maximum width and height.
    The resulting image will maintain the aspect ratio of the original image.

    :param pil_img: The PIL image to resize.
    :type pil_img: PIL.Image.Image
    :param max_width: The maximum width of the image.
    :type max_width: Optional[int]
    :param max_height: The maximum height of the image.
    :type max_height: Optional[int]
    :return: The resized PIL image.
    :rtype: PIL.Image.Image
    """
    if max_width is None and max_height is None:
        return pil_img
    height_reduction_factor = pil_img.height / max_height if max_height else 1
    width_reduction_factor = pil_img.width / max_width if max_width else 1
    reduction_factor = max(height_reduction_factor, width_reduction_factor)
    return pil_img.resize(
        (int(pil_img.width / reduction_factor), int(pil_img.height / reduction_factor))
    )


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
                frame = Image.fromarray(frame)
            elif img_type == "cv2":
                # frame is already in cv2 format
                pass
        else:
            logging.exception(f"Failed to extract frame at {time_ms} ms")
            frame = None
        time_to_frame_mapper[time_ms] = frame

    video_capture.release()
    return time_to_frame_mapper
