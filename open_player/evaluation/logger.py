"""ExperimentLogger: CSV + JSONL experiment records (no heavy platform)."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Appends structured records to both a CSV and a JSONL file."""

    def __init__(self, log_dir: str, name: str = "experiment") -> None:
        self.log_dir = str(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.name = name
        self.csv_path = os.path.join(self.log_dir, f"{name}.csv")
        self.jsonl_path = os.path.join(self.log_dir, f"{name}.jsonl")
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=None)
        self._fieldnames: Optional[List[str]] = None

    def record(self, step: int, **metrics: Any) -> None:
        row: Dict[str, Any] = {"step": int(step)}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    row[f"{k}_{kk}"] = vv
            else:
                row[k] = v
        # keep column order stable across records
        if self._fieldnames is None:
            self._fieldnames = sorted(row.keys())
            self._csv.fieldnames = self._fieldnames
            self._csv.writeheader()
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
                self._csv.fieldnames = self._fieldnames
        for k in self._fieldnames:
            row.setdefault(k, "")
        self._csv.writerow(row)
        self._csv_file.flush()
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def close(self) -> None:
        self._csv_file.close()

    @staticmethod
    def load_csv(path: str) -> List[Dict[str, float]]:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out = []
            for row in reader:
                parsed: Dict[str, float] = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = 0.0
                out.append(parsed)
            return out
