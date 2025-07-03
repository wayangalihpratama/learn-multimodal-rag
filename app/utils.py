import clip
import torch
from PIL import Image
import logging
import numpy as np

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


def normalize(vec):
    """
    Normalize a vector to unit length (L2 norm = 1).

    This ensures that the vector has a consistent magnitude,
    which is important for similarity comparisons using cosine distance.

    Args:
        vec (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    return vec / np.linalg.norm(vec)


def get_fused_embedding(
    image_file=None, text=None, image_weight=0.5, text_weight=0.5
):
    """
    Generate a fused embedding from an image and/or text query.

    This function allows you to:
    - Use only an image,
    - Use only a text query,
    - Or combine both (multimodal query).

    When both are provided, it computes a weighted average of the image
    and text embeddings, then normalizes the fused result to unit length
    for stable similarity comparison.

    Args:
        image_file (file-like or None): Uploaded image file.
        text (str or None): Optional text input from user.
        image_weight (float): Weight for image embedding in fusion
        (default 0.5).
        text_weight (float): Weight for text embedding in fusion
        (default 0.5).

    Returns:
        list: Fused (or individual) embedding as a list of floats.

    Raises:
        ValueError: If neither image nor text is provided.
    """
    # Get normalized image embedding if image is provided
    image_emb = (
        normalize(get_image_embedding(image_file)) if image_file else None
    )

    # Get normalized text embedding if text is provided
    text_emb = normalize(get_text_embedding(text)) if text else None

    # If both image and text are provided, compute weighted fusion
    if image_emb is not None and text_emb is not None:
        fused = normalize(image_weight * image_emb + text_weight * text_emb)
        return fused.tolist()

    # If only image is provided
    elif image_emb is not None:
        return image_emb.tolist()

    # If only text is provided
    elif text_emb is not None:
        return text_emb.tolist()

    # If neither image nor text is provided, raise error
    else:
        raise ValueError("At least one of image or text must be provided.")
