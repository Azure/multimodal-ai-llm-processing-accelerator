import base64
from io import BytesIO
from typing import List, Optional

from PIL import Image, ImageDraw


def pil_img_to_base64(pil_img: Image.Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.

    :param pil_img:
        The PIL image to convert.

    :return:
        A base64 encoded string.
    """
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


def base64_to_pil_img(base64_img: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def convert_normalized_to_pixel_based_polygon(
    normalized_polygon: List[dict], page_width: int, page_height: int
) -> List[dict]:
    pixel_polygon = list()
    for point in normalized_polygon:
        x = round(point["x"] * page_width)
        y = round(point["y"] * page_height)
        pixel_polygon.append({"x": x, "y": y})
    return pixel_polygon


def resize_img_by_max(
    pil_img: Image.Image,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> Image.Image:
    """
    Resizes a PIL image to fit within the specified maximum width and height.
    The resulting image will maintain the aspect ratio of the original image.

    :param pil_img:
        The PIL image to resize.
    :param max_width:
        The maximum width of the image.
    :param max_height:
        The maximum height of the image.

    :return:
        The resized PIL image.
    """
    if max_width is None and max_height is None:
        return pil_img
    height_reduction_factor = pil_img.height / max_height if max_height else 1
    width_reduction_factor = pil_img.width / max_width if max_width else 1
    reduction_factor = max(height_reduction_factor, width_reduction_factor)
    return pil_img.resize(
        (int(pil_img.width / reduction_factor), int(pil_img.height / reduction_factor))
    )


def draw_polygon_on_pil_img(
    pil_img: Image.Image,
    polygon: List[dict],
    polygon_type: str = "normalized",
    outline: str = "red",
    width: int = 1,
) -> Image.Image:
    """
    Draws a polygon on a PIL image.

    :param pil_img:
        The PIL image to draw the polygon on.
    :param polygon:
        The polygon to draw. This can be a list of dictionaries with "x" and "y"
        keys, representing the points of the polygon.
    :param polygon_type:
        The type of polygon provided. Can be "normalized" (0-1) or "pixel"
        (pixel-based).
    :param outline:
        The color of the polygon outline.
    :param width:
        The width of the polygon outline.

    :return:
        The PIL image with the polygon drawn on it.
    """
    # Take a clean copy of the original
    pil_img = pil_img.copy()
    draw = ImageDraw.Draw(pil_img)
    if polygon_type == "normalized":
        # Convert normalized polygon (0-1) to pixel-based
        polygon = convert_normalized_to_pixel_based_polygon(
            polygon, pil_img.width, pil_img.height
        )
    # Convert to list of tuples, as expected by PIL
    draw_polygon = [(point["x"], point["y"]) for point in polygon]
    draw.polygon(draw_polygon, outline=outline, width=width)
    return pil_img
