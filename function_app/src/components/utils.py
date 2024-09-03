import io
import mimetypes
import os
from pathlib import Path
from typing import Optional, Union

from haystack.dataclasses import ByteStream


def get_bytestream_from_source_with_mimetype(
    source: Union[str, Path, ByteStream]
) -> ByteStream:
    """
    Creates a ByteStream object from a source, while always enriching the
    object with the mime_type.

    :param source:
        A source to convert to a ByteStream. Can be a string (path to a file),
        a Path object, or a ByteStream.
    :return:
        A ByteStream object.
    """

    if isinstance(source, ByteStream):
        return source
    if isinstance(source, (str, Path)):
        bs = ByteStream.from_file_path(Path(source))
        bs.meta["file_path"] = str(source)
        mime_type = mimetypes.guess_type(Path(source).as_posix())[0]
        if mime_type:
            bs.mime_type = mime_type
        return bs


def base64_file_to_buffer(b64_str: bytes, name: Optional[str] = None) -> io.BytesIO:
    """Convert a base64 string to a BytesIO object."""
    buffer = io.BytesIO(b64_str)
    if name is not None:
        buffer.name = name
    return buffer


class InvalidFileTypeError(Exception):
    pass


def reverse_dict(d: dict) -> dict:
    """
    Reverse the keys and values of a dictionary.
    """
    return {v: k for k, v in d.items()}


def get_file_ext_and_mime_type(
    valid_mimes_to_file_ext_mapper: dict[str, str],
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> tuple[str, str]:
    """
    Get the file extension and MIME type for a given file, while also validating
    that it exists within the set of valid values. This function will
    first check the filename if available, and then the content type. If neither
    are available or are not valid, it will raise an InvalidFileTypeError.

    :param valid_mimes_to_file_ext_mapper:
        A mapper of valid mime types (e.g. ['audio/wav', 'audio/mp3']) to their
        associated file extension (e.g. ['wav', 'mp3']).
    :param filename:
        The filename of the file.
    :param content_type:
        The content type of the file.
    """
    valid_file_ext_to_mime_mapper = reverse_dict(valid_mimes_to_file_ext_mapper)
    valid_file_extensions = list(valid_file_ext_to_mime_mapper.keys())
    valid_content_types = list(valid_file_ext_to_mime_mapper.values())
    if filename:
        file_ext = os.path.splitext(filename)[1][1:]
        if file_ext.lower() not in valid_file_extensions:
            raise InvalidFileTypeError(
                f"The file extension `{file_ext}` is not supported. "
                f"Please use one of the following supported extensions: {valid_file_extensions}"
            )
        content_type = valid_file_ext_to_mime_mapper[file_ext.lower()]
    elif content_type:
        if content_type not in valid_content_types:
            raise InvalidFileTypeError(
                f"The content type `{content_type}` is not supported. "
                f"Please use one of the following supported content types: {valid_content_types}"
            )
        file_ext = valid_file_ext_to_mime_mapper[content_type]
    else:
        raise InvalidFileTypeError("The file must have a filename or content type.")
    return file_ext, content_type
