"""
Compare tokenization outputs between two tokenizer backends/directories.
"""

import argparse
import os

from nanochat.common import get_base_dir
from nanochat.tokenizer import HuggingFaceTokenizer, RustBPETokenizer


def load_tokenizer(backend: str, tokenizer_dir: str):
    if backend == "huggingface":
        expected = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.exists(expected):
            raise FileNotFoundError(
                f"Missing HuggingFace tokenizer artifact: {expected}\n"
                "Train one with: uv run python -m scripts.tok_train --tokenizer-backend huggingface\n"
                "Or pass --tokenizer-dir-a/--tokenizer-dir-b to a directory that contains tokenizer.json."
            )
        return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    if backend == "rustbpe":
        expected = os.path.join(tokenizer_dir, "tokenizer.pkl")
        if not os.path.exists(expected):
            raise FileNotFoundError(
                f"Missing rustbpe tokenizer artifact: {expected}\n"
                "Train one with: uv run python -m scripts.tok_train --tokenizer-backend rustbpe\n"
                "Or pass --tokenizer-dir-a/--tokenizer-dir-b to a directory that contains tokenizer.pkl."
            )
        return RustBPETokenizer.from_directory(tokenizer_dir)
    raise ValueError(f"Unknown backend: {backend}")


def first_diff_index(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def token_preview(tokenizer, token_ids, max_items=10):
    preview_ids = token_ids[:max_items]
    preview_tokens = [repr(tokenizer.decode([tid])) for tid in preview_ids]
    return ", ".join(preview_tokens)


def main():
    parser = argparse.ArgumentParser(description="Compare token IDs between two tokenizers")
    parser.add_argument("--backend-a", choices=["huggingface", "rustbpe"], required=True)
    parser.add_argument("--backend-b", choices=["huggingface", "rustbpe"], required=True)
    parser.add_argument("--tokenizer-dir-a", type=str, default=None, help="Directory for tokenizer A")
    parser.add_argument("--tokenizer-dir-b", type=str, default=None, help="Directory for tokenizer B")
    parser.add_argument("--text", action="append", default=[], help="Text to compare (repeatable)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    tok_dir_a = args.tokenizer_dir_a or os.path.join(base_dir, "tokenizer")
    tok_dir_b = args.tokenizer_dir_b or os.path.join(base_dir, "tokenizer")

    tokenizer_a = load_tokenizer(args.backend_a, tok_dir_a)
    tokenizer_b = load_tokenizer(args.backend_b, tok_dir_b)

    texts = args.text or [
        "Hello world! 12345",
        "Numbers: 1 23 456 7890",
        "Unicode: 你好世界 🌍",
        "A quick brown fox jumps over the lazy dog.",
    ]

    all_equal = True
    print(f"A: {args.backend_a} @ {tok_dir_a}")
    print(f"B: {args.backend_b} @ {tok_dir_b}")
    print(f"Comparing {len(texts)} text sample(s)...")
    print("-" * 80)

    for i, text in enumerate(texts, start=1):
        ids_a = tokenizer_a.encode(text)
        ids_b = tokenizer_b.encode(text)
        diff_i = first_diff_index(ids_a, ids_b)
        equal = diff_i == -1
        all_equal = all_equal and equal

        print(f"[{i}] equal={equal} len_a={len(ids_a)} len_b={len(ids_b)}")
        if equal:
            continue

        print(f"    first_diff_index={diff_i}")
        if diff_i < len(ids_a):
            print(f"    a_id={ids_a[diff_i]} a_tok={repr(tokenizer_a.decode([ids_a[diff_i]]))}")
        else:
            print("    a_id=<eos>")
        if diff_i < len(ids_b):
            print(f"    b_id={ids_b[diff_i]} b_tok={repr(tokenizer_b.decode([ids_b[diff_i]]))}")
        else:
            print("    b_id=<eos>")
        print(f"    a_preview={token_preview(tokenizer_a, ids_a)}")
        print(f"    b_preview={token_preview(tokenizer_b, ids_b)}")

    print("-" * 80)
    if all_equal:
        print("All compared token ID sequences are identical.")
    else:
        print("Differences found.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
