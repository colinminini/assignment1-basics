"""Lean, fast tokenization of a large text file -> numpy .npy token array on disk.

Built on HuggingFace `tokenizers` (Rust). `encode_batch` releases the GIL and
parallelizes across all cores internally, so NO Python multiprocessing is needed.

Input is many documents separated by a marker (default `<|endoftext|>`). We
stream-split on it, encode each document, and re-emit the marker id between them
-> `... docA, <eot>, docB, <eot>, ...`.

Output is a real .npy: we reserve a fixed-size header, append token bytes while
streaming (so the full array never has to fit in RAM), then patch the true shape
into the header at the end. Read it back with np.load(..., mmap_mode="r") --
self-describing (dtype + shape baked in), no sidecar, lazily memmapped.

    pip install tokenizers numpy
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# magic(8) + length-field(2) + header-string(118) = 128 bytes, a multiple of 64.
_HEADER_LEN = 118


def _header_str(dtype, n: int) -> bytes:
    descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    d = "{'descr': %r, 'fortran_order': False, 'shape': (%d,), }" % (descr, n)
    assert len(d) + 1 <= _HEADER_LEN, "shape too long for reserved header"
    return (d + " " * (_HEADER_LEN - len(d) - 1) + "\n").encode("latin1")


class FastTokenizer:
    def __init__(
        self,
        pretrained: str = "gpt2",
        sep_token: str | None = "<|endoftext|>",
        special_tokens: list[str] | None = None,
    ):
        self.tok = Tokenizer.from_pretrained(pretrained)  # first call downloads + caches

        extra = list(special_tokens or [])
        if sep_token and sep_token not in extra:
            extra.append(sep_token)
        if extra:
            self.tok.add_special_tokens(extra)            # atomic, single-id, in-string

        self.sep = sep_token
        self.sep_id = self.tok.token_to_id(sep_token) if sep_token else None
        if sep_token is not None:
            assert self.sep_id is not None, f"{sep_token!r} is not a single token"

        self.vocab_size = self.tok.get_vocab_size()
        # uint16 halves storage; valid while vocab < 65536 (e.g. GPT-2 = 50257).
        self.dtype = np.uint16 if self.vocab_size < (1 << 16) else np.uint32
        self._sep_arr = (
            np.array([self.sep_id], dtype=self.dtype) if self.sep_id is not None else None
        )

    # --- helpers -----------------------------------------------------------
    def token_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tok.encode(text).ids, dtype=self.dtype)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tok.decode(
            np.asarray(ids).ravel().tolist(), skip_special_tokens=skip_special_tokens
        )

    # --- big file -> .npy --------------------------------------------------
    def encode_file(
        self,
        in_path,
        out_path,
        batch_docs: int = 2048,
        read_chars: int = 1 << 20,   # 1 MB read buffer; keeps RAM bounded
        verbose: bool = True,
    ) -> int:
        in_path, out_path = Path(in_path), Path(out_path)
        total = 0
        with open(in_path, "r", encoding="utf-8") as f, open(out_path, "wb") as out:
            # reserve the header (real shape patched in at the end)
            out.write(b"\x93NUMPY\x01\x00")
            out.write(struct.pack("<H", _HEADER_LEN))
            out.write(_header_str(self.dtype, 0))

            batch: list[str] = []
            for doc in self._documents(f, read_chars):
                if doc == "":
                    continue
                batch.append(doc)
                if len(batch) >= batch_docs:
                    total += self._flush(batch, out)
                    batch.clear()
                    if verbose:
                        print(f"\r{total:,} tokens", end="", flush=True)
            if batch:
                total += self._flush(batch, out)

        with open(out_path, "r+b") as out:       # patch true shape into the header
            out.seek(10)
            out.write(_header_str(self.dtype, total))

        if verbose:
            mb = out_path.stat().st_size / 1e6
            print(f"\r{total:,} tokens  ->  {out_path}  ({mb:.1f} MB, {np.dtype(self.dtype).name})")
        return total

    def _documents(self, f, read_chars):
        """Stream documents split on the separator, without loading the whole file."""
        if not self.sep:
            yield from f
            return
        buf = ""
        while True:
            block = f.read(read_chars)
            if not block:
                break
            buf += block
            docs = buf.split(self.sep)
            buf = docs.pop()                      # last piece may be incomplete -> keep it
            yield from docs
        if buf:
            yield buf

    def _flush(self, batch, out) -> int:
        encs = self.tok.encode_batch(batch)       # parallel across cores, GIL released
        parts = []
        for e in encs:
            parts.append(np.asarray(e.ids, dtype=self.dtype))
            if self._sep_arr is not None:
                parts.append(self._sep_arr)
        arr = np.concatenate(parts)
        arr.tofile(out)                           # raw append, after the reserved header
        return arr.size

    # --- read back ---------------------------------------------------------
    @staticmethod
    def load(out_path) -> np.memmap:
        return np.load(out_path, mmap_mode="r")   # self-describing; lazily paged