"""Learn a byte-pair-encoding subword vocabulary for Astrax input encoding.

This is an input-side encoder only. The output side of Astrax keeps the full
UTF-8 codepoint space (see include/astrax/charset.hpp); subwords give the
input a learned, shared representation for synonyms, morphology, and mixed
Chinese/English text.

Produces two files under data/:
  - astrax_subwords.txt : one token per line, most frequent merge first
  - astrax_subwords.bin : binary merge table for the C++ loader

The vocabulary starts from raw Unicode codepoints so it works for any script.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

END_OF_WORD = "\u0000"


def read_documents(paths: list[pathlib.Path], max_bytes: int) -> list[str]:
    documents: list[str] = []
    total = 0
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                documents.append(line)
                total += len(line.encode("utf-8"))
                if total >= max_bytes:
                    return documents
    return documents


def word_frequencies(documents: list[str]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for document in documents:
        for word in document.split():
            counts[word] += 1
    return counts


def learn_bpe(counts: collections.Counter, merges: int, min_frequency: int):
    # Efficient BPE: keep pair statistics and only re-scan words that contain
    # the chosen pair. A naive re-scan of every word for every merge is
    # O(merges * words) and is far too slow on tens of megabytes.
    words = [list(word) for word in counts]
    frequencies = list(counts.values())
    merge_list: list[tuple[str, str]] = []

    # Pair -> set of word indices containing it, and weighted pair counts.
    pair_counts: collections.Counter = collections.Counter()
    pair_words: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    for index, symbols in enumerate(words):
        for position in range(len(symbols) - 1):
            pair = (symbols[position], symbols[position + 1])
            pair_counts[pair] += frequencies[index]
            pair_words[pair].add(index)

    for _ in range(merges):
        best_pair = None
        best_count = 0
        for pair, count in pair_counts.items():
            if count > best_count:
                best_count = count
                best_pair = pair
        if best_pair is None or best_count < min_frequency:
            break
        merge_list.append(best_pair)
        merged = best_pair[0] + best_pair[1]

        affected = sorted(pair_words.get(best_pair, ()))
        pair_words.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

        for index in affected:
            symbols = words[index]
            # Remove this word's contribution to all its current pairs.
            for position in range(len(symbols) - 1):
                pair = (symbols[position], symbols[position + 1])
                if pair in pair_counts:
                    pair_counts[pair] -= frequencies[index]
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                    pair_words.get(pair, set()).discard(index)

            # Apply the merge to this word.
            output = []
            position = 0
            while position < len(symbols):
                if (position + 1 < len(symbols)
                        and symbols[position] == best_pair[0]
                        and symbols[position + 1] == best_pair[1]):
                    output.append(merged)
                    position += 2
                else:
                    output.append(symbols[position])
                    position += 1
            words[index] = output

            # Re-add this word's contribution with the merged symbols.
            for position in range(len(output) - 1):
                pair = (output[position], output[position + 1])
                pair_counts[pair] += frequencies[index]
                pair_words[pair].add(index)

    return merge_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merges", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-megabytes", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=200000)
    args = parser.parse_args()

    corpus_paths = [DATA / "wiki_zh.txt", DATA / "wiki_en.txt"]
    documents = read_documents(corpus_paths, args.max_megabytes * 1024 * 1024)
    if not documents:
        print("no corpus found under data/", file=sys.stderr)
        raise SystemExit(1)

    counts = word_frequencies(documents)
    # Keep the most frequent words; rare words contribute little to merges and
    # would make the pair index dominate memory.
    counts = collections.Counter(dict(counts.most_common(args.max_words)))
    merges = learn_bpe(counts, args.merges, args.min_frequency)

    text_path = DATA / "astrax_subwords.txt"
    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Astrax BPE merges, most frequent first\n")
        for left, right in merges:
            handle.write(f"{left}\t{right}\n")

    binary_path = DATA / "astrax_subwords.bin"
    with binary_path.open("wb") as handle:
        handle.write(b"ASTRAX-BPE-1")
        handle.write(struct.pack("<I", len(merges)))
        for left, right in merges:
            left_bytes = left.encode("utf-8")
            right_bytes = right.encode("utf-8")
            handle.write(struct.pack("<H", len(left_bytes)))
            handle.write(left_bytes)
            handle.write(struct.pack("<H", len(right_bytes)))
            handle.write(right_bytes)

    print(f"documents={len(documents)} unique_words={len(counts)} merges={len(merges)}")
    print(f"text={text_path} binary={binary_path}")


if __name__ == "__main__":
    main()
