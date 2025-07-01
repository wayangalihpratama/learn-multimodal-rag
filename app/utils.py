import clip
import torch
from PIL import Image
import logging

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPTokenizer,
    CLIPModel,
)

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Load CLIP model ---
try:
    clip_model, preprocess = clip.load("ViT-B/32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    logger.info(f"CLIP model loaded on device: {device}")
except Exception as e:
    logger.exception(f"Failed to load CLIP model: {e}")

# --- Load CLIP model for text ---
text_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_clip_tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# --- BLIP Setup (runs once) ---
try:
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    logger.info("BLIP model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load BLIP model.")
    raise e


def get_image_embedding(image_file):
    """
    Generate CLIP image embedding from an uploaded image file.

    Args:
        image_file: File-like object or image path (Streamlit uploader
        or PIL-compatible)

    Returns:
        Numpy array of image embedding
    """
    try:
        image = Image.open(image_file).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)

        logger.info("Image embedding generated successfully.")
        return embedding[0].cpu().numpy()

    except Exception as e:
        logger.exception("Failed to generate image embedding.")
        raise e


def generate_caption(image_file):
    """
    Generate a natural language caption for the image using BLIP.
    Args:
        image_file: File-like object or PIL Image
    Returns:
        String caption
    """
    try:
        image = Image.open(image_file).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)

        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        logger.info(f"BLIP caption generated: {caption}")
        return caption

    except Exception as e:
        logger.exception(f"Failed to generate caption: {e}")
        return "No caption"


def get_text_embedding(text: str):
    """
    Generate CLIP text embedding from an image caption.

    Args:
        text: string

    Returns:
        Numpy array of text embedding
    """
    inputs = text_clip_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = text_clip_model.get_text_features(**inputs)
    return outputs[0].cpu().numpy()
