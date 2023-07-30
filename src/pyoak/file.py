import logging
from pathlib import Path

import chardet

logger = logging.getLogger(__name__)


def read_text_unknown_encoding(file: Path) -> str | None:
    """Read a file and return the contents as a string. Will guess the encoding if it is not UTF-8.

    Args:
        file (Path): Path to the file

    Returns:
        str | None: Contents of the file if the file was read successfully, None otherwise

    """
    logger.debug(f"Reading file <{file}>")
    raw_text = ""
    with file.open(encoding="utf-8") as f:
        try:
            raw_text = f.read()
        except UnicodeDecodeError:
            logger.debug(f"File {file} is not in UTF-8 encoding. Trying to detect encoding.")
            with file.open("rb") as fb:
                encs = chardet.detect(fb.read())
                logger.debug(f"Detected encoding {encs['encoding']} for file {file}.")
                with file.open(encoding=encs["encoding"]) as fn:
                    try:
                        raw_text = fn.read()
                    except Exception:
                        logger.exception(f"Could not detect encoding for file {file}.")
                        return None

    return raw_text
