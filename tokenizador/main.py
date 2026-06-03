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
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config, resolve_path


def train_tokenizer(dataset_path: str, vocab_size: int, outdir: str):
    from rustbpe import Tokenizer

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


def tokenize_dataset(tok, dataset_path: str, outdir: str, split_ratio: float = 0.1) -> dict:
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
    np.array(val_ids, dtype=np.uint32).tofile(os.path.join(outdir, "val.bin"))
    return {
        "dataset_path": dataset_path,
        "train_bin_path": os.path.join(outdir, "train.bin"),
        "val_bin_path": os.path.join(outdir, "val.bin"),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "split_ratio": split_ratio,
    }


def encode_text(tok, text: str) -> List[int]:
    return tok.encode(text)


def decode_ids(tok, ids: List[int]) -> str:
    return tok.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina BPE e gera train.bin/val.bin.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "train-bpe", "tokenize"],
        default="all",
        help="Use all para treinar BPE e tokenizar em uma unica execucao.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer_config = config["tokenizer"]

    bpe_dataset_path = resolve_path(tokenizer_config["bpe_dataset_path"])
    model_dataset_path = resolve_path(tokenizer_config["model_dataset_path"])
    artifacts_dir = resolve_path(tokenizer_config["artifacts_dir"])
    dataset_dir = resolve_path(tokenizer_config["dataset_dir"])
    split_ratio = float(tokenizer_config.get("split_ratio", 0.1))
    seed = int(tokenizer_config.get("seed", 42))

    random.seed(seed)

    tokenizer = None
    if args.stage in {"all", "train-bpe", "tokenize"}:
        print(f"Treinando BPE com: {bpe_dataset_path}")
        tokenizer = train_tokenizer(
            str(bpe_dataset_path),
            int(tokenizer_config["vocab_size"]),
            str(artifacts_dir),
        )
        print(f"Artifacts do BPE salvos em: {artifacts_dir}")

    if args.stage in {"all", "tokenize"}:
        print(f"Tokenizando dados do modelo: {model_dataset_path}")
        metadata = tokenize_dataset(tokenizer, str(model_dataset_path), str(dataset_dir), split_ratio)
        metadata["vocab_size"] = int(tokenizer_config["vocab_size"])
        metadata_path = dataset_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"train.bin/val.bin salvos em: {dataset_dir}")
        print(f"Metadata salva em: {metadata_path}")


if __name__ == "__main__":
    main()
