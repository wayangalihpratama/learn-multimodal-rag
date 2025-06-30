import os
from tqdm import tqdm
import chromadb
import logging
import hashlib

from chromadb.config import Settings
from utils import get_image_embedding, generate_caption

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ChromaDB client ---
chroma_client = chromadb.Client(
    Settings(persist_directory="app/chroma", anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="pest_disease")
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
                uid = generate_image_id(file_path=file_path)

                # Check if this image has already been indexed
                existing = collection.get(ids=[uid])
                if existing["ids"]:
                    logger.info(f"🔄 Updating existing entry for: {file_path}")
                    collection.delete(ids=[uid])

                with open(file_path, "rb") as img_file:
                    embedding = get_image_embedding(img_file)
                    img_file.seek(0)  # rewind file before reusing
                    blip_caption = generate_caption(img_file)

                # Combine BLIP + label into a better caption
                caption = (
                    f"{blip_caption}. This image shows symptoms of {label}."
                )

                collection.add(
                    ids=[uid],
                    embeddings=[embedding.tolist()],
                    metadatas=[
                        {"label": label, "path": file_path, "caption": caption}
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
