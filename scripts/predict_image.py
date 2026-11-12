"""Classify a local image using the saved V5 model without running Gradio.

Usage: python scripts/predict_image.py path/to/photo.jpg
"""

from pathlib import Path
import argparse
import json
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.arsl.predictor import get_predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="ArSL V5 single-image prediction")
    parser.add_argument("image", type=Path, help="Image file to classify")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    with Image.open(args.image) as image:
        result = get_predictor().predict(image)
    output = {
        "best_class": result.best_class,
        "best_arabic": result.best_arabic,
        "top_five": [
            {"class": en, "arabic": ar, "model_score": score}
            for en, ar, score in result.top_five
        ],
        "note": "Exploratory, uncalibrated scores; static alphabet signs only",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
