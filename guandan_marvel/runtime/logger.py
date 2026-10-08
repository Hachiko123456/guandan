# SPDX-License-Identifier: MIT
# Adapted from AtaraxosAI/doudizhu@acdcf199cf422e2b683cb22142b8e7cb33b14460:marvel/runtime/logger.py.
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import torch


class MetricSerializer:
    @staticmethod
    def serialize(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        if isinstance(value, Mapping):
            return {str(key): MetricSerializer.serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [MetricSerializer.serialize(item) for item in value]
        return str(value)


class MetricSinks:
    """Explicit JSONL/CSV metric sinks; no remote logger is enabled implicitly."""

    def __init__(self, *, jsonl_path: Path | None = None, csv_path: Path | None = None) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.csv_path = Path(csv_path) if csv_path else None
        for path in (self.jsonl_path, self.csv_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Mapping[str, object], *, step: int) -> None:
        serialized = {str(key): MetricSerializer.serialize(value) for key, value in payload.items()}
        if self.jsonl_path is not None:
            record = {"schema": "guandan-marvel-metrics-v1", "step": int(step), **serialized}
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        if self.csv_path is not None:
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "key", "value"])
                if write_header:
                    writer.writeheader()
                for key, value in serialized.items():
                    cell = value if isinstance(value, (str, int, float, bool)) or value is None else json.dumps(value, ensure_ascii=False)
                    writer.writerow({"step": int(step), "key": key, "value": "" if cell is None else cell})


__all__ = ["MetricSerializer", "MetricSinks"]
