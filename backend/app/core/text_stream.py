############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# text_stream.py: Decode a byte stream to text without
#     breaking on characters split across network chunks
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Incremental UTF-8 decoding for streamed responses.

A backend's stream arrives as arbitrary byte chunks. A character that takes
more than one byte (an emoji, a dash, any non-Latin text) can be cut in two by
a chunk boundary, and ``chunk.decode("utf-8")`` on the first half raises
``UnicodeDecodeError: ... unexpected end of data``. That failed the whole
request mid-stream (seen in production on Kimi K3 and GLM 5.3, always just
under a 4,096-byte boundary).

``Utf8StreamDecoder`` keeps the incomplete tail and prepends it to the next
chunk, so text comes out whole. Bytes that are not valid UTF-8 at all become
U+FFFD instead of ending the stream: one bad byte from a model is not worth a
failed request.
"""

import codecs
from typing import Union


class Utf8StreamDecoder:
    """Feed byte chunks, get text; a split character waits for its other half."""

    __slots__ = ("_decoder",)

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, chunk: Union[bytes, bytearray, memoryview, str]) -> str:
        """Text for this chunk, minus any incomplete character at its end.
        A ``str`` passes through unchanged (some inner streams already yield text)."""
        if isinstance(chunk, str):
            return chunk
        return self._decoder.decode(bytes(chunk), final=False)

    def flush(self) -> str:
        """Call when the stream ends. A character still incomplete then is
        malformed, and comes out as U+FFFD."""
        return self._decoder.decode(b"", final=True)
