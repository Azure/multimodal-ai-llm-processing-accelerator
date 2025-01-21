import base64
import itertools
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
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


def base64_to_pil_img(base64_img: str) -> PILImage:
    """Convert a base64 encoded string to a PIL image.

    :param: base64_img: The base64 encoded image.
    :type: base64_img: str
    :return: The PIL image.
    :rtype: PIL.Image.Image
    """
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def rotate_img_arr(
    image: np.ndarray,
    angle: float,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Rotates an image array by a given angle, expanding the dimensions to fit the
    rotated image.

    :param image: Image to be rotated, as a numpy array.
    :type image: np.ndarray
    :param angle: Angle to rotate the image by in the counter-clockwise
        direction.
    :type angle: float
    :param border_color: If the page dimensions are expanded to fit the rotated
        image, set the fill color of the new pixels. Defaults to (255, 255, 255)
    :type border_color: tuple[int, int, int], optional
    :return: Rotated image array.
    :rtype: np.ndarray
    """
    # Code is adapted from https://cristianpb.github.io/blog/image-rotation-opencv
    angle = angle % 360
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(
        image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color
    )


def rotate_img_pil(
    image: PILImage,
    angle: float,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> PILImage:
    """
    Rotates a PIL image by a given angle, expanding the dimensions to fit the
    rotated image.

    :param image: Image to be rotated, as a PIL object.
    :type image: PIL.Image.Image
    :param angle: Angle to rotate the image by in the counter-clockwise
        direction.
    :type angle: float
    :param border_color: If the page dimensions are expanded to fit the rotated
        image, set the fill color of the new pixels. Defaults to (255, 255, 255)
    :type border_color: tuple[int, int, int], optional
    :return: Rotated PIL image.
    :rtype: PIL.Image.Image
    """
    img_arr = np.array(image)
    rotated_img_arr = rotate_img_arr(img_arr, angle, border_color=border_color)
    return Image.fromarray(rotated_img_arr)


def rotate_coord(
    coord: tuple[float, float], angle: float, cx: int, cy: int, height: int, width: int
) -> tuple[float, float]:
    """
    Rotates a coordinate around a center point by a given angle, accounting for
    the new dimensions of the rotated image (including when the new image is
    expanded to accomodate the image content - e.g. cases where angle is not
    a multiple of 90).

    :param coord: Tuple of the coordinate to rotate (x, y).
    :type coord: tuple[float, float]
    :param angle: Angle to rotate the coordinate by in the counter-clockwise
        direction.
    :type angle: float
    :param cx: X-coordinate of the center point.
    :type cx: int
    :param cy: Y-coordinate of the center point.
    :type cy: int
    :param height: Height of the original image.
    :type height: int
    :param width: Width of the original image.
    :type width: int
    :return: Rotated coordinate.
    :rtype: tuple[float, float]
    """
    # Code is adapted from https://cristianpb.github.io/blog/image-rotation-opencv
    angle = angle % 360
    # opencv calculates standard transformation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Grab  the rotation components of the matrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    v = [coord[0], coord[1], 1]
    # Perform the actual rotation and return the image
    calculated = np.dot(M, v)
    return calculated[0], calculated[1]


def rotate_polygon(
    polygon: List[float], angle: float, img_width: int, img_height: int
) -> List[float]:
    """
    Rotates the coordinates of a polygon around the center of the image by a
    given angle, accounting for the new dimensions of the rotated image (
    including when the new image is expanded to accomodate the image content -
    e.g. cases where angle is not a multiple of 90).

    :param polygon: List of coordinates of the polygon (x0, y0, x1, y1, etc.)
    :type polygon: List[float]
    :param angle: Angle to rotate the polygon by in the counter-clockwise
        direction.
    :type angle: float
    :param img_width: Width of the original image.
    :type img_width: int
    :param img_height: Height of the original image.
    :type img_height: int
    :return: Rotated polygon coordinates (x0, y0, x1, y1, etc.).
    :rtype: List[float]
    """
    # Code is adapted from https://cristianpb.github.io/blog/image-rotation-opencv
    # Calculate the shape of rotated images
    cx, cy = img_width // 2, img_height // 2
    coord_pairs = [polygon[i : i + 2] for i in range(0, len(polygon), 2)]
    new_coords = []
    for x, y in coord_pairs:
        new_x, new_y = rotate_coord((x, y), angle, cx, cy, img_height, img_width)
        new_coords.extend([new_x, new_y])
    return new_coords


@dataclass
class TransformedImage:
    """
    Dataclass to store a transformed image and metadata related to the
    transformations that have occurred, making it possible to apply those same
    transformations to other objects (e.g. transform coordinates from the
    original image to the transformed image).
    """

    image: PILImage
    orig_image: PILImage
    rotation_applied: float


def crop_img(img: PILImage, crop_poly: list[float]) -> PILImage:
    """
    Crops an image based on the coordinates of an [x0, y0, x1, y1, ...] polygon.

    :param img: Image to crop.
    :type img: PIL.Image.Image
    :param crop_poly: List of coordinates of the polygon (x0, y0, x1, y1, etc.)
    :type crop_poly: list[float]
    :return: The cropped image.
    :rtype: PIL.Image.Image
    """
    top_left = (min(crop_poly[::2]), min(crop_poly[1::2]))
    bottom_right = (max(crop_poly[::2]), max(crop_poly[1::2]))
    return img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))


def scale_flat_poly_list(
    polygon: list[float],
    existing_scale: tuple[float, float],
    new_scale: tuple[float, float],
) -> list[float]:
    """
    Scales a flat polygon list from one scale to a new one.

    :param polygon: List of coordinates of the polygon (x0, y0, x1, y1, etc.)
    :type polygon: list[float]
    :param existing_scale: Dimensions of the existing scale (width, height)
    :type existing_scale: tuple[float, float]
    :param new_scale: Dimensions of the new scale (width, height)
    :type new_scale: tuple[float, float]
    :return: Normalized polygon scaled to the new dimensions.
    :rtype: list[float]
    """
    x_coords = polygon[::2]
    x_coords = [x / existing_scale[0] * new_scale[0] for x in x_coords]
    y_coords = polygon[1::2]
    y_coords = [y / existing_scale[1] * new_scale[1] for y in y_coords]
    return list(itertools.chain.from_iterable(zip(x_coords, y_coords)))


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


def flat_poly_list_to_poly_dict_list(
    flat_poly_list: List[float],
) -> List[Dict[str, float]]:
    """
    Converts a flat poly list (e.g. [x0, y0, x1, y1, ...]) to a list of
    dictionaries with "x" and "y" keys.

    :param flat_poly_list: The flat polygon list.
    :type flat_poly_list: List[float]
    :return: The polygon list as a list of dictionaries.
    :rtype: List[Dict[str, float]]
    """
    return [
        {"x": flat_poly_list[i], "y": flat_poly_list[i + 1]}
        for i in range(0, len(flat_poly_list), 2)
    ]


def poly_dict_list_to_flat_poly_list(
    poly_dict_list: List[Dict[str, float]]
) -> List[float]:
    """
    Converts a list of dictionaries with "x" and "y" keys to a flat poly list
    (e.g. [x0, y0, x1, y1, ...]).

    :param poly_dict_list: The polygon list as a list of dictionaries.
    :type poly_dict_list: List[Dict[str, float]]
    :return: The flat polygon list.
    :rtype: List[float]
    """
    return [coord for poly in poly_dict_list for coord in [poly["x"], poly["y"]]]


def draw_polygon_on_pil_img(
    pil_img: PILImage,
    polygon: List[dict],
    outline_color: str = "red",
    outline_width: int = 1,
) -> PILImage:
    """
    Draws a polygon on a PIL image.

    :param pil_img: The PIL image to draw the polygon on.
    :type pil_img: PIL.Image.Image
    :param polygon: The polygon to draw. This can be a list of dictionaries
        with "x" and "y" keys, representing the points of the polygon.
    :type polygon: List[dict]
    :param outline_color: The color of the polygon outline.
    :type outline_color: str
    :param outline_width: The width of the polygon outline, in pixels.
    :type outline_width: int
    :return: The PIL image with the polygon drawn on it.
    :rtype: PIL.Image.Image
    """
    # Take a clean copy of the original
    pil_img = pil_img.copy()
    draw = ImageDraw.Draw(pil_img)
    # Convert to list of tuples, as expected by PIL
    draw_polygon = [(point["x"], point["y"]) for point in polygon]
    draw.polygon(draw_polygon, outline=outline_color, width=outline_width)
    return pil_img


def get_flat_poly_lists_convex_hull(poly_lists: list[list[float]]) -> list[int]:
    """
    Get the convex hull of a list of polygons.

    :param poly_lists: List of (x0, y0, x1, y1) polygon lists.
    :type poly_lists: list[list[float]]
    :return: Flattened (x0, y0, x1, y1) convex hull polygon.
    :rtype: list[int]
    """
    list_of_points = list(itertools.chain.from_iterable(poly_lists))
    # Multiply points by 1000 to avoid issues with casting floats to ints
    list_of_points = [point * 1000 for point in list_of_points]
    contour = np.array(list_of_points).reshape((-1, 1, 2)).astype(np.int32)
    return contour.reshape(-1) / 1000  # Divide by 1000 to get back to original scale
