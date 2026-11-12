"""Dependency-light tests: no TensorFlow or GPU required."""
from pathlib import Path
import json
import numpy as np
import pytest

from src.arsl.predictor import (
    ASSETS_DIR,
    combine_scores,
    read_metadata,
    validate_assets,
)


def test_accepted_v5_weighted_rule():
    a = np.array([0.7, 0.3])
    b = np.array([0.3, 0.7])
    svm = np.array([0.9, 0.1])
    result = combine_scores(a, b, svm, "V3_50_specialist_50")
    np.testing.assert_allclose(result, [0.7, 0.3])


def test_v3_rule_and_unknown_mode():
    a = np.array([0.6, 0.4]); b = np.array([0.2, 0.8]); s = np.array([0.5, 0.5])
    np.testing.assert_allclose(combine_scores(a, b, None, "V3_original"), [0.4, 0.6])
    with pytest.raises(ValueError):
        combine_scores(a, b, s, "invalid")


def test_metadata_and_assets_exist():
    assert validate_assets() == []
    labels, config = read_metadata(ASSETS_DIR)
    assert len(labels["class_names"]) == 31
    assert len(labels["arabic_labels"]) == 31
    assert config["mode"] == "V3_50_specialist_50"
    assert config["image_sizes"] == [[160, 160], [192, 192]]


def test_v5_results_are_labeled_exploratory():
    metrics = json.loads((ASSETS_DIR / "metrics_v5.json").read_text(encoding="utf-8"))
    assert metrics["test_if_run"]["correct"] == 657
    assert metrics["test_if_run"]["samples"] == 744
    assert metrics["test_if_run"]["status"] == "exploratory_reused_test"
