import io
import re

import torch
from PIL import Image
import streamlit as st
import easyocr
import numpy as np
import requests

from transformers import AutoTokenizer, AutoImageProcessor

from config import (
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
    BEST_MODEL_PATH,
    DEVICE,
)
from model import MultimodalHatefulMemeModel  



@st.cache_resource
def load_components():

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)

    model = MultimodalHatefulMemeModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
    )
    try:
        state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error(
            f"Checkpoint not found at {BEST_MODEL_PATH}. "
            "Please train the model first by running `python train.py`."
        )
        st.stop()
    model.to(DEVICE)
    model.eval()

    return tokenizer, image_processor, model


@st.cache_resource
def load_ocr_reader():

    return easyocr.Reader(['en']) 



def extract_text_from_image(image: Image.Image) -> str:

    reader = load_ocr_reader()
    np_img = np.array(image)
    results = reader.readtext(np_img, detail=0)  
    extracted = " ".join(results).strip()
    return extracted


KEYWORD_WEIGHTS = {
    "isis": 0.5,
    "terrorist": 0.5,
    "terrorists": 0.5,
    "jihad": 0.5,
    "bomb": 0.5,
    "suicide bomber": 0.5,
    "suicide": 0.5,
    "kill": 0.5,
    "strip club": 0.5,
    "nazi": 0.5,
    "hitler": 0.5,
    "racist": 0.5,
    "nigger": 0.5,
    "nigga": 0.5
}


def predict(
    model: MultimodalHatefulMemeModel,
    tokenizer,
    image_processor,
    image: Image.Image,
    text: str,
) -> float:

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    
    
    img_enc = image_processor(
        image,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    pixel_values = img_enc["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )  
        prob = torch.sigmoid(logits).item()

    return prob


def apply_textual_heuristics(prob: float, text: str):
    """
    Lightweight rule-based boost for obviously harmful keywords.
    """
    text_lower = text.lower()
    matched_keywords = []
    boost = 0.0

    for keyword, weight in KEYWORD_WEIGHTS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            matched_keywords.append(keyword)
            boost += weight

    boost = min(boost, 0.4)
    adjusted_prob = min(1.0, prob + boost)
    return adjusted_prob, matched_keywords, boost


def load_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a URL and return it as a PIL Image.
    Raises ValueError if download fails or response is not an image.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Could not download image: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        st.warning(
            "The provided URL did not return an image content type. "
            "Attempting to process it anyway."
        )

    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise ValueError("Downloaded file is not a valid image.") from exc

    return image


def main():
    st.title("Multimodal Harmful Meme Detection System")
    st.write(
        "Upload a meme image. The system will **read the text from the image** "
        "and use both the text and the image to estimate harmfulness."
    )

    tokenizer, image_processor, model = load_components()

    uploaded_file = st.file_uploader("Upload meme image", type=["png", "jpg", "jpeg"])
    image_url = st.text_input("Or enter meme image URL")

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Meme", use_container_width=True)
    elif image_url.strip():
        with st.spinner("Downloading image from URL..."):
            try:
                image = load_image_from_url(image_url.strip())
                st.image(image, caption="Meme from URL", use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))

    if image is not None:
        with st.spinner("Reading text from the meme (OCR)..."):
            ocr_text = extract_text_from_image(image)

        if not ocr_text:
            st.warning(
                "OCR could not detect any text. You can type the meme text below "
                "if you want to still run a prediction."
            )

        meme_text = st.text_area(
            "Extracted meme text (you can edit if OCR is wrong)",
            value=ocr_text,
        )

        if st.button("Analyze Meme"):
            if meme_text.strip() == "":
                st.error(
                    "No text available. Please type the meme text or upload an "
                    "image with visible text."
                )
            else:
                prob_harmful = predict(model, tokenizer, image_processor, image, meme_text)
                adjusted_prob, keyword_hits, extra = apply_textual_heuristics(
                    prob_harmful, meme_text
                )

                st.write(f"**Base model probability:** {prob_harmful:.3f}")
                if keyword_hits:
                    st.info(
                        "Keyword heuristic boost applied for: "
                        + ", ".join(sorted(set(keyword_hits)))
                        + f" (+{extra:.2f})"
                    )
                st.write(f"**Final harmful probability:** {adjusted_prob:.3f}")

                if adjusted_prob >= 0.5:
                    st.error("The meme is likely **harmful**.")
                else:
                    st.success("The meme is likely **not harmful**.")
    elif not image_url.strip():
        st.info("Upload an image or provide an image URL to start the analysis.")


if __name__ == "__main__":
    main()
