from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"Linha invalida no YAML: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        parsed_value = _parse_scalar(value)
        parent[key] = parsed_value

        if isinstance(parsed_value, dict):
            stack.append((indent, parsed_value))

    return root


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_path(config_path or DEFAULT_CONFIG_PATH)
    text = path.read_text(encoding="utf8")

    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(text)

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config invalida em {path}")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    if model.get("vocab_size") == "auto":
        model["vocab_size"] = int(config["tokenizer"]["vocab_size"])
    return model


def training_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["training"])


def legacy_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    model = model_config(config)
    tokenizer = config["tokenizer"]
    return {
        "tokenizer": {
            "dataset_path": tokenizer["bpe_dataset_path"],
            "vocab_size": int(tokenizer["vocab_size"]),
            "outdir": tokenizer["artifacts_dir"],
            "dataset_text_path": tokenizer["model_dataset_path"],
            "train_bin_path": tokenizer["train_bin_path"],
            "val_bin_path": tokenizer["val_bin_path"],
            "outdir_tokenizer": tokenizer["dataset_dir"],
            "regenerate_dataset": bool(
                config["training"].get("regenerate_tokenizer_and_dataset", False)
            ),
        },
        "modelo": model,
        "treino": training_config(config),
    }
