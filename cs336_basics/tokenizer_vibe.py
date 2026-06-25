"""Lean, fast trainable tokenizer + large text file -> numpy .npy token array.

Features:
- Train a GPT-2-style byte-level BPE tokenizer on local text files.
- Save tokenizer.json, vocab.json, merges.txt.
- Load from pretrained HF tokenizer, tokenizer.json, or vocab+merges.
- Stream-tokenize a huge text file into a real .npy without holding all tokens in RAM.

    pip install tokenizers numpy
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


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
        pretrained: str | None = "gpt2",
        tokenizer_json: str | Path | None = None,
        vocab_file: str | Path | None = None,
        merges_file: str | Path | None = None,
        sep_token: str | None = "<|endoftext|>",
        special_tokens: list[str] | None = None,
    ):
        """
        Load a tokenizer from one of:
        - tokenizer_json
        - vocab_file + merges_file
        - pretrained HF name, default "gpt2"

        Priority:
            tokenizer_json > vocab+merges > pretrained
        """
        if tokenizer_json is not None:
            self.tok = Tokenizer.from_file(str(tokenizer_json))

        elif vocab_file is not None or merges_file is not None:
            if vocab_file is None or merges_file is None:
                raise ValueError("Pass both vocab_file and merges_file.")
            self.tok = Tokenizer(
                BPE.from_file(
                    vocab=str(vocab_file),
                    merges=str(merges_file),
                )
            )
            self.tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tok.decoder = ByteLevelDecoder()

        else:
            if pretrained is None:
                raise ValueError(
                    "Provide one of: pretrained, tokenizer_json, or vocab_file+merges_file."
                )
            self.tok = Tokenizer.from_pretrained(pretrained)

        extra = list(special_tokens or [])
        if sep_token and sep_token not in extra:
            extra.append(sep_token)

        if extra:
            self.tok.add_special_tokens(extra)

        self.sep = sep_token
        self.sep_id = self.tok.token_to_id(sep_token) if sep_token else None

        if sep_token is not None:
            assert self.sep_id is not None, f"{sep_token!r} is not a single token"

        self.vocab_size = self.tok.get_vocab_size()

        # uint16 halves storage; valid while vocab < 65536.
        self.dtype = np.uint16 if self.vocab_size < (1 << 16) else np.uint32
        self._sep_arr = (
            np.array([self.sep_id], dtype=self.dtype)
            if self.sep_id is not None
            else None
        )

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        files: str | Path | Iterable[str | Path],
        out_dir: str | Path,
        vocab_size: int = 50_257,
        min_frequency: int = 2,
        sep_token: str | None = "<|endoftext|>",
        special_tokens: list[str] | None = None,
        add_prefix_space: bool = False,
        show_progress: bool = True,
    ) -> "FastTokenizer":
        """
        Train a GPT-2-style byte-level BPE tokenizer.

        Saves:
            out_dir/tokenizer.json
            out_dir/vocab.json
            out_dir/merges.txt

        Notes:
        - vocab_size includes byte alphabet + special tokens + learned merges.
        - For very small models/corpora, use smaller vocab_size, e.g. 4096, 8192, 16384.
        - sep_token is registered as a special token, so it remains atomic.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(files, (str, Path)):
            file_list = [str(files)]
        else:
            file_list = [str(p) for p in files]

        all_specials = list(special_tokens or [])
        if sep_token and sep_token not in all_specials:
            all_specials.append(sep_token)

        tokenizer = Tokenizer(BPE())

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_specials,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        tokenizer.train(files=file_list, trainer=trainer)

        # Save full tokenizer config.
        tokenizer_json = out_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_json))

        # Save legacy GPT-2-style files: vocab.json + merges.txt.
        tokenizer.model.save(str(out_dir))

        return cls(
            tokenizer_json=tokenizer_json,
            sep_token=sep_token,
            special_tokens=special_tokens,
        )

    def save(self, out_dir: str | Path) -> None:
        """
        Save the current tokenizer to:
            tokenizer.json
            vocab.json
            merges.txt
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.tok.save(str(out_dir / "tokenizer.json"))
        self.tok.model.save(str(out_dir))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def token_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tok.encode(text).ids, dtype=self.dtype)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tok.decode(
            np.asarray(ids).ravel().tolist(),
            skip_special_tokens=skip_special_tokens,
        )

    # ---------------------------------------------------------------------
    # Big file -> .npy
    # ---------------------------------------------------------------------
    def encode_file(
        self,
        in_path,
        out_path,
        batch_docs: int = 2048,
        read_chars: int = 1 << 20,
        verbose: bool = True,
    ) -> int:
        in_path, out_path = Path(in_path), Path(out_path)
        total = 0

        with open(in_path, "r", encoding="utf-8") as f, open(out_path, "wb") as out:
            # Reserve the header. True shape is patched at the end.
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

        with open(out_path, "r+b") as out:
            out.seek(10)
            out.write(_header_str(self.dtype, total))

        if verbose:
            mb = out_path.stat().st_size / 1e6
            print(
                f"\r{total:,} tokens  ->  {out_path}  "
                f"({mb:.1f} MB, {np.dtype(self.dtype).name})"
            )

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
            buf = docs.pop()

            yield from docs

        if buf:
            yield buf

    def _flush(self, batch, out) -> int:
        encs = self.tok.encode_batch(batch)

        parts = []

        for e in encs:
            parts.append(np.asarray(e.ids, dtype=self.dtype))

            if self._sep_arr is not None:
                parts.append(self._sep_arr)

        arr = np.concatenate(parts)
        arr.tofile(out)

        return arr.size

    # ---------------------------------------------------------------------
    # Read back
    # ---------------------------------------------------------------------
    @staticmethod
    def load(out_path) -> np.memmap:
        return np.load(out_path, mmap_mode="r")

if __name__ == '__main__':

    tok = FastTokenizer.train(files="./data/owt-train.txt", out_dir="./data/owt-my_tokenizer", vocab_size=32000, min_frequency=2, sep_token="<|endoftext|>")

    tok = FastTokenizer(tokenizer_json="./data/owt-my_tokenizer/tokenizer.json", sep_token="<|endoftext|>")

    tok.encode_file(in_path="./data/owt-train.txt", out_path="./data/owt-train.npy")

    tok.encode_file(in_path="./data/owt-valid.txt", out_path="./data/owt-valid.npy")