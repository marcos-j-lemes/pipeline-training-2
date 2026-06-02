"""Pipeline minimal para BPE usando o binding `rustbpe`.

Etapas implementadas:
- Treinar `Tokenizer` a partir de `dataset.txt` (streaming por linhas)
- Exportar `vocab.json` (id -> token string/hex) e `merges.txt` (pares de ids)
- Tokenizar dataset em `train.bin` e `val.bin` (uint32 raw binary)
- Funções de inferência: `encode_text` e `decode_ids`

Uso: python pipeline.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np

from rustbpe import Tokenizer


def train_tokenizer(dataset_path: str, vocab_size: int, outdir: str) -> Tokenizer:
    tok = Tokenizer()
    # train_from_iterator aceita qualquer iterável Python; usar streaming por linhas
    with open(dataset_path, "r", encoding="utf8") as f:
        tok.train_from_iterator(f, vocab_size)

    os.makedirs(outdir, exist_ok=True)

    # export mergeable ranks and reconstruct token bytes by id
    ranks = tok.get_mergeable_ranks()  # list of (bytes, id)

    # Build token_bytes list (index -> bytes) and a map bytes->id for lookup
    token_bytes: List[bytes] = [bytes([i]) for i in range(256)]
    bytes_to_id = {token_bytes[i]: i for i in range(256)}

    # ranks may include the 0..255 entries first; process sorted by id
    ranks_sorted = sorted(ranks, key=lambda x: x[1])

    merges: List[Tuple[int, int]] = []
    for b, idx in ranks_sorted:
        # ensure token_bytes is large enough
        if idx < len(token_bytes):
            # already present (likely the 0..255 entries)
            token_bytes[idx] = b
            bytes_to_id[b] = idx
            continue

        # try to find a split b = left_bytes + right_bytes where both sides known
        found = False
        for split in range(1, len(b)):
            left_b = b[:split]
            right_b = b[split:]
            if left_b in bytes_to_id and right_b in bytes_to_id:
                left_id = bytes_to_id[left_b]
                right_id = bytes_to_id[right_b]
                merges.append((left_id, right_id))
                # ensure token_bytes list has space
                if idx >= len(token_bytes):
                    token_bytes.extend([b""] * (idx - len(token_bytes) + 1))
                token_bytes[idx] = b
                bytes_to_id[b] = idx
                found = True
                break

        if not found:
            # Fallback: just register the bytes under idx (might happen for weird cases)
            if idx >= len(token_bytes):
                token_bytes.extend([b""] * (idx - len(token_bytes) + 1))
            token_bytes[idx] = b
            bytes_to_id[b] = idx

    # Save vocab.json (id -> string if valid UTF-8 else hex)
    vocab = {}
    for i, b in enumerate(token_bytes):
        if not b:
            continue
        try:
            s = b.decode("utf-8")
        except Exception:
            s = b.hex()
        vocab[i] = s

    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Save merges as pairs of ids (one pair per line: left right)
    with open(os.path.join(outdir, "merges.txt"), "w", encoding="utf8") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")

    # Also save ranks for debugging / reproducibility
    ranks_serializable = [(b.hex(), i) for b, i in ranks_sorted]
    with open(os.path.join(outdir, "tokenizer_ranks.json"), "w", encoding="utf8") as f:
        json.dump(ranks_serializable, f, ensure_ascii=False)

    return tok


def tokenize_dataset(tok: Tokenizer, dataset_path: str, outdir: str, split_ratio: float = 0.1) -> None:
    # Read lines, encode each line, concatenate ids; split into train/val by lines
    train_ids: List[int] = []
    val_ids: List[int] = []

    with open(dataset_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids = tok.encode(line)
            if random.random() < split_ratio:
                val_ids.extend(ids)
            else:
                train_ids.extend(ids)

    os.makedirs(outdir, exist_ok=True)

    # Save as raw uint32 binary
    np.array(train_ids, dtype=np.uint32).tofile(os.path.join(outdir, "train.bin"))
    #np.array(val_ids, dtype=np.uint32).tofile(os.path.join(outdir, "val.bin"))


def encode_text(tok: Tokenizer, text: str) -> List[int]:
    return tok.encode(text)


def decode_ids(tok: Tokenizer, ids: List[int]) -> str:
    return tok.decode(ids)


# def main():
#     parser = argparse.ArgumentParser(description="BPE pipeline using rustbpe")
#     parser.add_argument("--dataset", default="dataset.txt")
#     parser.add_argument("--outdir", default="artifacts")
#     parser.add_argument("--vocab-size", type=int, default=5000)
#     parser.add_argument("--run", choices=["all", "train", "tokenize", "infer"], default="all")
#     args = parser.parse_args()

#     if args.run in ("all", "train"):
#         print("Training tokenizer...")
#         tok = train_tokenizer(args.dataset, args.vocab_size, args.outdir)
#     else:
#         # If not training, still create a Tokenizer instance and try to load ranks if present
#         tok = Tokenizer()
#         ranks_file = os.path.join(args.outdir, "tokenizer_ranks.json")
#         if os.path.exists(ranks_file):
#             with open(ranks_file, "r", encoding="utf8") as f:
#                 ranks = json.load(f)
#             # ranks: list of (hex, id)
#             # call get_mergeable_ranks is preferable; if not available, skip
#             try:
#                 _ = tok.get_mergeable_ranks()
#             except Exception:
#                 pass

#     if args.run in ("all", "tokenize"):
#         print("Tokenizing dataset and writing train/val binaries...")
#         tokenize_dataset(tok, args.dataset, args.outdir)

#     if args.run in ("all", "infer"):
#         # Demonstrate encode/decode
#         samples = ["Hello world!", "Exemplo de tokenização em português."]
#         for s in samples:
#             ids = encode_text(tok, s)
#             text = decode_ids(tok, ids)
#             print("Sample:", s)
#             print("Encoded len:", len(ids), "First ids:", ids[:20])
#             print("Decoded:", text)


if __name__ == "__main__":
    #main()

    # Treino do tokenizer ======================
    DATASET_PATH = "./dadosBrutos/dados_training_bpe.txt"
    VOCAB_SIZE = 5000
    OUTDIR = "artifacts"

    tokenizer = train_tokenizer(DATASET_PATH, VOCAB_SIZE, OUTDIR)

    #print("Mergeable ranks:", tokenizer.get_mergeable_ranks())
    print("Sample encoding:", tokenizer.encode("hello world"))
    print()
    print("Sample decoding:", tokenizer.decode(tokenizer.encode("olá mundo")))

    print()

    # Tokenização do dataset ==========================
    OUTDIR_TOKENIZER = "dataset"
    DATASET_PATH_TRAIN = "./dadosBrutos/dados_training_modelo.txt"

    tokenize_dataset(tokenizer, DATASET_PATH_TRAIN, OUTDIR_TOKENIZER)

    print("Tokenization complete. Train/val binaries saved in", OUTDIR_TOKENIZER)

    # Encode e decode de exemplo ========================

    text_sample = "Exemplo de tokenização em português."
    encoded_ids = tokenizer.encode(text_sample)
    decoded_text = tokenizer.decode(encoded_ids)

    print("Original text:", text_sample)
    print("Encoded IDs:", encoded_ids)
    print("Decoded text:", decoded_text)

    





    # tokenizer = Tokenizer()

    # tokenizer.train_from_iterator(["hello world", "hello rustbpe", "hello tokenizer"], vocab_size=256)

   # print("Mergeable ranks:", tokenizer.get_mergeable_ranks())

