import streamlit as st

from utils import get_image_embedding
import chromadb
from chromadb.config import Settings
import logging
import os

# --- Setup Logging ---
LOG_DIR = "app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("🔧 Streamlit app started.")

# --- Initialize ChromaDB ---
try:
    chroma_client = chromadb.Client(
        Settings(
            persist_directory="app/chroma",  # adjust if needed
            anonymized_telemetry=False,
        )
    )
    collection = chroma_client.get_or_create_collection(name="pest_disease")
    logger.info("✅ ChromaDB initialized.")
except Exception as e:
    logger.exception(f"❌ Failed to initialize ChromaDB: {e}")

# --- Streamlit UI ---
st.title("🌿 Multimodal RAG: Pest & Disease Image Search")
st.write(
    "Upload a plant or leaf image to find visually similar disease cases."
)

uploaded_file = st.file_uploader(
    "📤 Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    logger.info(f"📤 Image uploaded: {uploaded_file.name}")

    try:
        with st.spinner("🔍 Searching for similar images..."):
            query_embedding = get_image_embedding(uploaded_file)
            logger.info("✅ Image embedding generated.")

            results = collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=5
            )
            logger.info("✅ Query to ChromaDB completed.")

        st.subheader("🔎 Top Similar Results")
        for metadata in results["metadatas"][0]:
            label = metadata.get("label", "Unknown")
            caption = metadata.get("caption", "-")
            path = metadata.get("path", "N/A")

            st.markdown(f"**Label:** {label}")
            st.markdown(f"_Caption:_ {caption}")
            try:
                st.image(path, width=300)
                logger.info(f"🖼️ Displayed image from: {path}")
            except Exception as img_err:
                st.warning(f"Unable to display image: {path}")
                logger.warning(
                    f"⚠️ Failed to display image at {path}: {img_err}"
                )

    except Exception as err:
        st.error("Something went wrong during image search.")
        logger.exception(f"❌ Error during image processing or search: {err}")
