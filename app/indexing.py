import os
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from utils import get_image_embedding

# from PIL import Image
import uuid
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ChromaDB client ---
chroma_client = chromadb.Client(
    Settings(persist_directory="app/chroma", anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="pest_disease")

# --- Dataset path ---
DATA_DIR = "data/pest_disease"


# --- Indexing ---
def index_images():
    logger.info(f"📂 Indexing images from: {DATA_DIR}")
    # indexing.py (partial, inside loop)


for root, _, files in os.walk(DATA_DIR):
    for file in tqdm(files, desc="Indexing images"):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(root, file)
        try:
            # Get label from subfolder name
            label = os.path.relpath(root, DATA_DIR)

            uid = str(uuid.uuid4())

            with open(file_path, "rb") as img_file:
                embedding = get_image_embedding(img_file)

            caption = "No caption yet"

            collection.add(
                ids=[uid],
                embeddings=[embedding.tolist()],
                metadatas=[
                    {"label": label, "path": file_path, "caption": caption}
                ],
            )

            logger.info(f"✅ Indexed: {file_path} [label: {label}]")

        except Exception as e:
            logger.warning(f"⚠️ Failed to process {file_path}: {e}")


if __name__ == "__main__":
    index_images()
    logger.info("🎉 Indexing completed.")
