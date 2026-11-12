"""Inference pipeline for the validated ArSL V5 ensemble.

The preprocessing and ensemble weights intentionally match the model
artifacts bundled with ArSL_V5_Final_Results.zip.  This module does not
train or update model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

ASSETS_DIR = Path(__file__).resolve().parents[2] / "models"
REQUIRED_ASSETS = (
    "model_0.keras",
    "model_1.keras",
    "v5_classifier.joblib",
    "labels.json",
    "inference_config.json",
)


@dataclass(frozen=True)
class SignPrediction:
    """Ordered top classes and their relative model scores."""

    best_class: str
    best_arabic: str
    top_five: tuple[tuple[str, str, float], ...]
    confidence: float


def validate_assets(asset_dir: Path = ASSETS_DIR) -> list[str]:
    """Return missing assets; safe to call without importing TensorFlow."""
    return [name for name in REQUIRED_ASSETS if not (asset_dir / name).is_file()]


def combine_scores(
    cnn0: np.ndarray,
    cnn1: np.ndarray,
    specialist: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    """Reproduce the final V5 decision rule (not a newly tuned ensemble)."""
    cnn0 = np.asarray(cnn0, dtype=np.float64)
    cnn1 = np.asarray(cnn1, dtype=np.float64)
    if cnn0.shape != cnn1.shape or cnn0.ndim != 1:
        raise ValueError("أبعاد احتمالات الشبكتين غير متطابقة")

    baseline = 0.5 * cnn0 + 0.5 * cnn1
    if mode == "V3_original":
        scores = baseline
    else:
        if specialist is None or np.asarray(specialist).shape != baseline.shape:
            raise ValueError("أبعاد احتمالات SVM غير صالحة")
        specialist = np.asarray(specialist, dtype=np.float64)
        if mode == "specialist_only":
            scores = specialist
        elif mode == "V3_75_specialist_25":
            scores = 0.75 * baseline + 0.25 * specialist
        elif mode == "V3_50_specialist_50":
            scores = 0.5 * baseline + 0.5 * specialist
        else:
            raise ValueError(f"وضع دمج غير معروف: {mode}")

    if not np.all(np.isfinite(scores)) or np.any(scores < -1e-6):
        raise ValueError("قيم التنبؤ غير صالحة")
    return scores


def read_metadata(assets_dir: Path = ASSETS_DIR) -> tuple[dict, dict]:
    """Read recorded labels/configuration without loading model weights."""
    labels = json.loads((assets_dir / "labels.json").read_text(encoding="utf-8"))
    config = json.loads((assets_dir / "inference_config.json").read_text(encoding="utf-8"))
    class_names = labels.get("class_names", [])
    if len(class_names) != 31 or len(set(class_names)) != 31:
        raise ValueError("عدد أو ترتيب فئات النموذج غير صالح")
    if config.get("class_count") != len(class_names):
        raise ValueError("عدد الفئات غير متوافق مع إعدادات النموذج")
    if config.get("image_sizes") != [[160, 160], [192, 192]]:
        raise ValueError("أبعاد الصور تختلف عن النموذج المحفوظ")
    return labels, config


class ArSLPredictor:
    """Loads the two trained CNNs and the original PCA+SVM classifier."""

    def __init__(self, assets_dir: Path = ASSETS_DIR) -> None:
        missing = validate_assets(assets_dir)
        if missing:
            raise FileNotFoundError(
                "ملفات النماذج غير موجودة: " + ", ".join(missing)
            )

        # Heavy imports are deferred to the first inference request.
        import joblib
        import tensorflow as tf

        self.tf = tf
        self.labels, self.config = read_metadata(assets_dir)
        self.names = self.labels["class_names"]
        self.arabic_names: Mapping[str, str] = self.labels.get("arabic_labels", {})
        self.models = [
            tf.keras.models.load_model(str(assets_dir / f"model_{i}.keras"), compile=False)
            for i in range(2)
        ]
        self.extractors = [
            tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer("global_average_pooling2d").output, model.outputs[0]],
            )
            for model in self.models
        ]
        self.learned = joblib.load(assets_dir / "v5_classifier.joblib")
        self.reducer = self.learned["reducer"]
        self.classifier = self.learned["classifier"]

        if not all(int(model.output_shape[-1]) == 31 for model in self.models):
            raise ValueError("مخرجات الشبكات لا تطابق 31 فئة")
        if getattr(self.reducer, "n_features_in_", None) != 2560:
            raise ValueError("PCA يجب أن يستقبل 2560 ميزة")
        if getattr(self.reducer, "n_components_", None) != 128:
            raise ValueError("PCA يجب أن ينتج 128 ميزة")
        if not np.array_equal(self.classifier.classes_, np.arange(31)):
            raise ValueError("ترتيب فئات SVM لا يطابق تسميات الصور")

    def predict(self, image: Image.Image) -> SignPrediction:
        if image is None:
            raise ValueError("يرجى رفع صورة قبل التحليل")
        if not isinstance(image, Image.Image):
            raise TypeError("المدخل المتوقع هو صورة PIL")
        if image.width * image.height > 18_000_000:
            raise ValueError("الصورة كبيرة جداً؛ استخدم صورة أصغر من 18 ميجابكسل")

        # Maintain original RGB [0, 255] preprocessing: each CNN handles its
        # own normalization. Do not rescale or horizontally flip the photo.
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
        feats: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for extractor, size in zip(self.extractors, self.config["image_sizes"]):
            resized = self.tf.image.resize_with_pad(
                self.tf.convert_to_tensor(rgb), int(size[0]), int(size[1])
            )
            embed, probs = extractor.predict(self.tf.expand_dims(resized, 0), verbose=0)
            feats.append(np.asarray(embed, dtype=np.float32))
            predictions.append(np.asarray(probs[0], dtype=np.float64))

        specialist = None
        if self.config["mode"] != "V3_original":
            # Exactly two L2-normalized 1280-value embeddings -> 2560 values.
            combined_features = np.concatenate([normalize(f) for f in feats], axis=1)
            reduced_features = self.reducer.transform(combined_features)
            specialist = np.asarray(
                self.classifier.predict_proba(reduced_features)[0], dtype=np.float64
            )

        scores = combine_scores(predictions[0], predictions[1], specialist, self.config["mode"])
        indices = np.argsort(scores)[::-1][:5]
        top_five = tuple(
            (self.names[int(idx)], self.arabic_names.get(self.names[int(idx)], self.names[int(idx)]), float(scores[idx]))
            for idx in indices
        )
        return SignPrediction(
            best_class=top_five[0][0],
            best_arabic=top_five[0][1],
            confidence=top_five[0][2],
            top_five=top_five,
        )


@lru_cache(maxsize=1)
def get_predictor() -> ArSLPredictor:
    """Load models only once per running server."""
    return ArSLPredictor()
