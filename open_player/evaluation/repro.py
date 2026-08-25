"""Reproducibility helpers: experiment ids, config snapshots, git hash."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def git_hash(short: int = 8) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"
    return out[:short] if short else out


def config_fingerprint(config_dict: Dict[str, Any], length: int = 8) -> str:
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def new_experiment_id(prefix: str = "p1.5", config: Optional[Dict[str, Any]] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg8 = config_fingerprint(config or {}, 8)
    return f"{prefix}-{git_hash(8)}-{cfg8}-{ts}"


def make_run_dir(base: str = "results", config: Optional[Dict[str, Any]] = None, exp_id: Optional[str] = None, note: str = "") -> str:
    """Create results/<exp_id>/ with a run_info.json snapshot."""
    exp_id = exp_id or new_experiment_id(config=config)
    run_dir = os.path.join(base, exp_id)
    os.makedirs(run_dir, exist_ok=True)
    info = {
        "exp_id": exp_id,
        "git_hash": git_hash(64),
        "config": config or {},
        "created_at": datetime.now().isoformat(),
        "note": note,
    }
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=str)
    with open(os.path.join(base, "latest_run.txt"), "w", encoding="utf-8") as fh:
        fh.write(exp_id)
    return run_dir


def save_json(run_dir: str, name: str, payload: Any) -> str:
    path = os.path.join(run_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


def save_csv(run_dir: str, name: str, rows: list) -> str:
    """rows: list of flat dicts; first row defines the columns."""
    import csv
    path = os.path.join(run_dir, name)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return path
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path
