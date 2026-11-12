# Arabic Sign Language Alphabet Recognition — V5

An academic **31-class static Arabic sign alphabet** image classifier with a live **Gradio** interface (photo upload or webcam).

**Recorded exploratory evaluation:** 88.31% accuracy (657/744), Macro F1 88.24%, validation accuracy 90.19% (671/744). This is **not an independent, signer-disjoint test**; the test set was reused during earlier project iterations. Performance on new signers must be assessed separately.

## Inference architecture

Two previously fine-tuned CNNs — MobileNetV2 (160×160) and EfficientNetV2B0 (192×192) — provide 1280-d embeddings each and 31-class probability vectors. Each embedding is L2-normalized, concatenated into 2560 features, compressed with a previously fitted PCA to 128 dimensions, and classified with an RBF-SVM. The final score is the unmodified V5 decision rule:

`0.50 × (0.50 × MobileNetV2 + 0.50 × EfficientNetV2B0) + 0.50 × SVM`

## Start locally

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_assets.py --strict
python app.py
```

Open http://127.0.0.1:7860.

## Deployment

- GitHub stores the source code and trained artifacts. GitHub Pages alone **cannot** run TensorFlow inference.
- Deploy on a Python-capable service such as [Hugging Face Spaces (Gradio)](https://huggingface.co/docs/hub/spaces-sdks-gradio).
- `model_1.keras` exceeds GitHub browser's 25 MiB per-file upload limit. Push via Git CLI (files up to 100 MiB) or use Git LFS.

See [Arabic deployment guide](docs/DEPLOYMENT_AR.md) and [Arabic model card](docs/MODEL_CARD_AR.md).

## Attribution

Data: [RGB Arabic Alphabets Sign Language (AASL)](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset); [dataset paper](https://arxiv.org/abs/2301.11932). No raw training images are redistributed in this repository. The MIT license only covers new application code; trained model and data rights must be reviewed separately before public redistribution.
