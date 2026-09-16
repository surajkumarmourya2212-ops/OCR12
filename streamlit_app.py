import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import ocr_utils


st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ Handwritten OCR System")
st.write("Recognise handwritten digits, words and short sentences.")


@st.cache_resource
def load_models():
    digit_model = ocr_utils.load_digit_model()
    crnn_model = ocr_utils.load_crnn_model()
    return digit_model, crnn_model


try:
    digit_model, crnn_model = load_models()
except Exception as e:
    st.error("❌ Could not load the digit model.")
    st.code(str(e))
    st.stop()


if crnn_model is not None:
    text_engine = "CRNN + CTC"
elif ocr_utils.tesseract_available():
    text_engine = "Tesseract OCR"
else:
    text_engine = "No text engine available"

st.caption(
    f"Digit Model: MNIST TensorFlow/Keras | Text Engine: {text_engine}"
)

tab_digit, tab_text = st.tabs(["🔢 Digit Recognition", "📝 Text Recognition"])


def read_uploaded_image(uploaded_file):
    """Safely convert a Streamlit UploadedFile to a PIL RGB image."""
    try:
        data = uploaded_file.getvalue()
        if not data:
            raise ValueError("The uploaded file is empty.")
        return Image.open(__import__("io").BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"Could not read the image: {e}") from e


with tab_digit:
    st.subheader("Handwritten Digit Recognition")
    st.write("Upload an image containing one handwritten digit (0–9).")

    uploaded_digit = st.file_uploader(
        "📷 Upload handwritten digit",
        type=["png", "jpg", "jpeg", "webp"],
        key="digit_upload",
    )

    if uploaded_digit is not None:
        try:
            image = read_uploaded_image(uploaded_digit)

            # Fixed: no deprecated use_column_width argument.
            st.image(image, caption="Uploaded Image", width=600)

            if st.button(
                "🔍 Predict Digit",
                type="primary",
                key="predict_digit",
            ):
                with st.spinner("Predicting digit..."):
                    result = ocr_utils.predict_digit(np.array(image))

                st.success(f"✅ Predicted Digit: {result['prediction']}")
                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("Upload an image containing one handwritten digit.")


with tab_text:
    st.subheader("Handwritten Text Recognition")
    st.write("Upload an image containing a handwritten word or short sentence.")

    st.info(f"Text engine currently available: **{text_engine}**")

    uploaded_text = st.file_uploader(
        "📷 Upload handwritten text",
        type=["png", "jpg", "jpeg", "webp"],
        key="text_upload",
    )

    if uploaded_text is not None:
        try:
            image = read_uploaded_image(uploaded_text)

            # Fixed: no deprecated use_column_width argument.
            st.image(image, caption="Uploaded Text Image", width=600)

            if st.button(
                "📝 Recognise Text",
                type="primary",
                key="predict_text",
            ):
                with st.spinner("Recognising text..."):
                    result = ocr_utils.predict_text(np.array(image))

                st.success("Recognition Complete")
                st.text_area(
                    "Recognised Text:",
                    value=result["prediction"],
                    height=150,
                )
                st.caption(f"Engine: {result['engine']}")

        except Exception as e:
            st.error(f"❌ Error: {e}")


st.divider()
st.caption(
    "Digit Recognition: MNIST TensorFlow/Keras Model | "
    "Text Recognition: CRNN + CTC or Tesseract OCR"
)
