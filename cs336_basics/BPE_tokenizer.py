import regex as re
from collections import Counter, defaultdict
import json

class BPE:

    def __init__(self):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def pre_tokenize(self, text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
        """Split text into pre-tokens; return {byte_tuple: count}."""
        special_set = set(special_tokens)
        # Longer specials first, so <|endoftext|> matches before any hypothetical <|end|>.
        sorted_specials = sorted(special_set, key=len, reverse=True)
        split_pat = '(' + '|'.join(re.escape(s) for s in sorted_specials) + ')'
        chunks = re.split(split_pat, text)

        counts: Counter[tuple[int, ...]] = Counter()
        for chunk in chunks:
            if chunk in special_set:
                continue  # specials get added to the vocab directly, never merged
            for m in re.finditer(self.PAT, chunk):
                counts[tuple(m.group().encode('utf-8'))] += 1
        return counts

    def train_bpe(self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        with open(input_path, 'r') as f:
            text = f.read()
        
        # 1. Seed vocab: 256 byte values + specials
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for tok in special_tokens:
            vocab[len(vocab)] = tok.encode('utf-8')

        # 2. Pre-tokenize. We'll mutate the words during training,
        #    so store them as lists with a parallel array of frequencies.
        word_counts = self.pre_tokenize(text, special_tokens)
        words: list[list[int]] = [list(w) for w in word_counts]
        freqs: list[int] = list(word_counts.values())

        merges: list[tuple[bytes, bytes]] = []
        num_merges = vocab_size - len(vocab)

        for _ in range(num_merges):
            # 3. Count all adjacent pairs, weighted by word frequency.
            pair_freqs: Counter[tuple[int, int]] = Counter()
            for word, freq in zip(words, freqs):
                for pair in zip(word, word[1:]):
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # 4. Pick the best pair. Tie-break on the bytes of the pair,
            #    preferring the lexicographically GREATER pair.
            best = max(
                pair_freqs,
                key=lambda p: (pair_freqs[p], (vocab[p[0]], vocab[p[1]])),
            )

            # 5. Add merged token to vocab and record the merge.
            new_id = len(vocab)
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
            merges.append((vocab[best[0]], vocab[best[1]]))

            # 6. Apply the merge in every word.
            a, b = best
            for i, word in enumerate(words):
                if len(word) < 2:
                    continue
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                        new_word.append(new_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                words[i] = new_word

        return vocab, merges

import json
import base64

def save_vocab_and_merges(vocab: dict[int, bytes],
                          merges: list[tuple[bytes, bytes]],
                          vocab_path: str,
                          merges_path: str) -> None:
    # bytes -> base64 ascii string
    serializable_vocab = {
        token_id: base64.b64encode(token_bytes).decode('ascii')
        for token_id, token_bytes in vocab.items()
    }
    serializable_merges = [
        [base64.b64encode(a).decode('ascii'),
         base64.b64encode(b).decode('ascii')]
        for a, b in merges
    ]

    with open(vocab_path, 'w') as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    with open(merges_path, 'w') as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)


def load_vocab_and_merges(vocab_path: str,
                          merges_path: str
                          ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path) as f:
        raw_vocab = json.load(f)
    with open(merges_path) as f:
        raw_merges = json.load(f)

    vocab = {
        int(token_id): base64.b64decode(b64_str)        # str -> int, b64 -> bytes
        for token_id, b64_str in raw_vocab.items()
    }
    merges = [
        (base64.b64decode(a), base64.b64decode(b))      # list -> tuple, b64 -> bytes
        for a, b in raw_merges
    ]
    return vocab, merges

if __name__ == '__main__':
    bpe = BPE()
    vocab, merges = bpe.train_bpe('./data/TinyStoriesV2-GPT4-valid.txt', vocab_size=500, special_tokens=['<|endoftext|>', '<|startoftext|>'])
    save_vocab_and_merges(vocab, merges, './data/vocab.json', './data/merges.json')