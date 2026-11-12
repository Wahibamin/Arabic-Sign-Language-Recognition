"""Verify files before deployment; does not load TensorFlow models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models"
REQUIRED = ["model_0.keras", "model_1.keras", "v5_classifier.joblib", "labels.json", "inference_config.json", "metrics_v5.json"]


def run(strict: bool = False) -> int:
    failures = []
    for name in REQUIRED:
        path = ASSETS / name
        if not path.is_file():
            failures.append(f"MISSING: {name}")
            continue
        print(f"✓ {name} ({path.stat().st_size / (1024 * 1024):.2f} MiB)")
        if name.endswith(".keras"):
            with ZipFile(path) as zf:
                if not {"config.json", "metadata.json", "model.weights.h5"}.issubset(zf.namelist()):
                    failures.append(f"INCOMPLETE KERAS FILE: {name}")
    if not failures:
        labels = json.loads((ASSETS / "labels.json").read_text(encoding="utf-8"))
        config = json.loads((ASSETS / "inference_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((ASSETS / "metrics_v5.json").read_text(encoding="utf-8"))
        if len(labels["class_names"]) != 31 or config["class_count"] != 31:
            failures.append("Class count must be 31")
        if config["mode"] != "V3_50_specialist_50":
            failures.append("Expected accepted V5 ensemble setting")
        if (metrics["test_if_run"]["correct"], metrics["test_if_run"]["samples"]) != (657, 744):
            failures.append("Recorded test summary does not match the supplied V5 report")

    manifest = ROOT / "MODEL_SHA256SUMS.txt"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            digest, relpath = line.split("  ", 1)
            path = ROOT / relpath
            if not path.is_file():
                failures.append(f"MISSING HASHED FILE: {relpath}")
            else:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != got:
                    failures.append(f"HASH MISMATCH: {relpath}")

    for error in failures:
        print("ERROR:", error)
    if failures:
        return 1
    print("Assets and V5 metadata verified successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Fail if any model asset is missing")
    args = parser.parse_args()
    raise SystemExit(run(strict=args.strict))
