"""zstd compression for full original texts."""

from __future__ import annotations

import zstandard

ZSTD_LEVEL = 10

_compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
_decompressor = zstandard.ZstdDecompressor()


def compress(text: str) -> bytes:
    return _compressor.compress(text.encode("utf-8"))


def decompress(blob: bytes) -> str:
    return _decompressor.decompress(blob).decode("utf-8")
