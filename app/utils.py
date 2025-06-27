import clip
import torch
from PIL import Image
import logging

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
