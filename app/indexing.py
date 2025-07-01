import os
from tqdm import tqdm
import logging
import hashlib

from chromadb import HttpClient
from utils import get_image_embedding, generate_caption, get_text_embedding

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ChromaDB client ---
chroma_client = HttpClient(host="chromadb", port=8000)
collection = chroma_client.get_or_create_collection(
    name="pest_disease",
    metadata={"hnsw:space": "cosine"},  # 👈 ensures cosine distance
)
logger.info(f"✅ Chroma collection count: {collection.count()}")


# --- Dataset path ---
DATA_DIR = "data/pest_disease"


# --- Indexing ---
def generate_image_id(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()


def normalize_label(path: str) -> str:
    # Convert path like "Tomato_Blight-Leaf" -> "tomato blight leaf"
    return path.replace("_", " ").replace("-", " ").lower()


def index_images():
    logger.info(f"📂 Indexing images from: {DATA_DIR}")

    for root, _, files in os.walk(DATA_DIR):
        for file in tqdm(files, desc="Indexing images"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = os.path.join(root, file)
            try:
                label = normalize_label(os.path.relpath(root, DATA_DIR))
                shared_id = generate_image_id(file_path=file_path)

                # Check if this image has already been indexed
                existing_imgs = collection.get(ids=[f"{shared_id}_img"])
                if existing_imgs["ids"]:
                    logger.info(f"🔄 Updating existing entry for: {file_path}")
                    collection.delete(ids=[f"{shared_id}_img"])

                # Check if this text has already been indexed
                existing_txts = collection.get(ids=[f"{shared_id}_txt"])
                if existing_txts["ids"]:
                    logger.info(f"🔄 Updating existing entry for: {file_path}")
                    collection.delete(ids=[f"{shared_id}_txt"])

                with open(file_path, "rb") as img_file:
                    embedding_img = get_image_embedding(img_file)
                    img_file.seek(0)  # rewind file before reusing
                    blip_caption = generate_caption(img_file)

                # Combine BLIP + label into a better caption
                caption = (
                    f"{blip_caption}. This image shows symptoms of {label}."
                )
                embedding_text = get_text_embedding(caption)

                # Add image embedding
                collection.add(
                    ids=[f"{shared_id}_img"],
                    embeddings=[embedding_img.tolist()],
                    metadatas=[
                        {
                            "group_id": shared_id,
                            "type": "image",
                            "label": label,
                            "path": file_path,
                            "caption": caption,
                        }
                    ],
                )

                # Add text embedding
                collection.add(
                    ids=[f"{shared_id}_txt"],
                    embeddings=[embedding_text.tolist()],
                    metadatas=[
                        {
                            "group_id": shared_id,
                            "type": "caption",
                            "label": label,
                            "path": file_path,
                            "caption": caption,
                        }
                    ],
                )

                logger.info(f"✅ Indexed: {file_path} [label: {label}]")
                logger.info(
                    f"📦 Final collection size: {collection.count()} items."
                )

            except Exception as e:
                logger.warning(f"⚠️ Failed to process {file_path}: {e}")


if __name__ == "__main__":
    index_images()
    logger.info("🎉 Indexing completed.")
