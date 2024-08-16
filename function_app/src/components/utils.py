import mimetypes
from pathlib import Path
from typing import Union

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
