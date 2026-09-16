# Handwritten OCR Project

## Features
- Handwritten digit recognition using the supplied MNIST Keras model.
- Handwritten text recognition using the supplied CRNN + CTC model.
- Tesseract OCR is used as a fallback when the CRNN cannot be loaded.

## Project structure

```text
handwritten_ocr/
├── streamlit_app.py
├── ocr_utils.py
├── handwritten_ocr.py
├── requirements.txt
├── packages.txt
└── models/
    ├── digit_model.keras
    └── crnn_model.h5
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important fix

The Streamlit image display no longer uses the removed/deprecated
`use_column_width` argument. It uses a fixed pixel width instead, which is
compatible with current Streamlit versions.

The model loader also checks both `models/` and the project root, so the
models are found reliably after deployment.
