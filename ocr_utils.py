import os
import numpy as np
import cv2
import tensorflow as tf

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The app now works whether models are inside models/ or in the project root.
MODELS_DIR = os.path.join(BASE_DIR, "models")

DIGIT_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "digit_model.keras"),
    os.path.join(BASE_DIR, "digit_model.keras"),
]

CRNN_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "crnn_model.h5"),
    os.path.join(BASE_DIR, "crnn_model.h5"),
]

_digit_model = None
_crnn_model = None


def _first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_digit_model():
    global _digit_model

    if _digit_model is not None:
        return _digit_model

    model_path = _first_existing(DIGIT_MODEL_PATHS)
    if model_path is None:
        raise FileNotFoundError(
            "Digit model not found. Expected digit_model.keras in "
            "models/ or the project root."
        )

    _digit_model = tf.keras.models.load_model(model_path, compile=False)
    return _digit_model


def load_crnn_model():
    global _crnn_model

    if _crnn_model is not None:
        return _crnn_model

    model_path = _first_existing(CRNN_MODEL_PATHS)
    if model_path is None:
        return None

    try:
        _crnn_model = tf.keras.models.load_model(model_path, compile=False)
        return _crnn_model
    except Exception as e:
        print("Could not load CRNN model:", e)
        return None


def tesseract_available():
    if not PYTESSERACT_AVAILABLE:
        return False

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _to_gray(image):
    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image.copy()


def is_valid_digit_image(image):
    gray = _to_gray(image)
    gray = cv2.resize(gray, (200, 200))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 30]

    if len(contours) == 0 or len(contours) > 5:
        return False

    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    total_area = sum(cv2.contourArea(c) for c in contours)

    if total_area == 0 or largest_area / total_area < 0.60:
        return False

    x, y, w, h = cv2.boundingRect(largest)
    image_area = binary.shape[0] * binary.shape[1]
    size_ratio = (w * h) / image_area

    if size_ratio < 0.005 or size_ratio > 0.75:
        return False

    aspect_ratio = w / float(h)
    return 0.15 <= aspect_ratio <= 5.0


def preprocess_digit_image(image):
    gray = _to_gray(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    useful = [c for c in contours if cv2.contourArea(c) > 10]

    if useful:
        largest = max(useful, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        padding = int(max(w, h) * 0.25)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(binary.shape[1], x + w + padding)
        y2 = min(binary.shape[0], y + h + padding)
        binary = binary[y1:y2, x1:x2]

    h, w = binary.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = binary

    resized = cv2.resize(
        square, (28, 28), interpolation=cv2.INTER_AREA
    )
    resized = resized.astype("float32") / 255.0

    model = load_digit_model()
    input_shape = model.input_shape

    if len(input_shape) == 4:
        return resized.reshape(1, 28, 28, 1)
    if len(input_shape) == 3:
        return resized.reshape(1, 28, 28)
    if len(input_shape) == 2:
        return resized.reshape(1, 784)

    raise ValueError(f"Unsupported model input shape: {input_shape}")


def predict_digit(image):
    if not is_valid_digit_image(image):
        raise ValueError(
            "Invalid image. Please upload one clear handwritten digit (0-9)."
        )

    model = load_digit_model()
    processed = preprocess_digit_image(image)

    probabilities = model.predict(processed, verbose=0)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "engine": "MNIST TensorFlow/Keras Model",
    }


def _get_characters():
    try:
        from handwritten_ocr import CHARACTERS
        return CHARACTERS
    except Exception:
        return (
            ' !"#&\'()*+,-./0123456789:;?'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        )


def predict_text_crnn(gray):
    model = load_crnn_model()
    if model is None:
        return None

    try:
        # The supplied CRNN was trained for 64 x 256 grayscale images.
        image = cv2.resize(gray, (256, 64))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image, verbose=0)

        input_length = np.ones(
            predictions.shape[0], dtype=np.int32
        ) * predictions.shape[1]

        decoded, _ = tf.keras.backend.ctc_decode(
            predictions,
            input_length,
            greedy=True,
        )

        decoded = decoded[0].numpy()[0]
        characters = _get_characters()

        result = ""
        for index in decoded:
            index = int(index)
            if 0 <= index < len(characters):
                result += characters[index]

        return result.strip()

    except Exception as e:
        print("CRNN prediction error:", e)
        return None


def predict_text(image):
    gray = _to_gray(image)

    if load_crnn_model() is not None:
        prediction = predict_text_crnn(gray)
        if prediction is not None:
            return {
                "prediction": prediction,
                "engine": "CRNN + CTC",
            }

    if tesseract_available():
        prediction = pytesseract.image_to_string(
            gray,
            config="--psm 6"
        )
        return {
            "prediction": prediction.strip(),
            "engine": "Tesseract OCR",
        }

    raise RuntimeError(
        "No text recognition engine available. Add crnn_model.h5 "
        "or install Tesseract OCR."
    )
