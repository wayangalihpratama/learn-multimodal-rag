import streamlit as st
from utils import (
    get_image_embedding,
    get_text_embedding,
)  # make sure you implement get_text_embedding
from chromadb import HttpClient
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

DISTANCE_THRESHOLD = 0.1  # adjust empirically

# --- Initialize ChromaDB ---
try:
    chroma_client = HttpClient(host="chromadb", port=8000)
    collection = chroma_client.get_or_create_collection(
        name="pest_disease",
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("✅ ChromaDB initialized.")
    logger.info(f"✅ Chroma collection count: {collection.count()}")

    test_result = collection.peek()
    if not test_result["ids"]:
        logger.warning("⚠️ No data found in ChromaDB collection.")
        st.warning(
            "⚠️ No image data indexed yet. Please run the indexing script."
        )
        st.stop()
    else:
        logger.info(f"📦 ChromaDB contains {len(test_result['ids'])} items.")

except Exception as e:
    logger.exception(f"❌ Failed to initialize ChromaDB: {e}")
    st.error("❌ Failed to initialize ChromaDB.")
    st.stop()

# --- Streamlit UI ---
st.title("🌿 Multimodal RAG: Pest & Disease Image Search")
st.write(
    "Upload a plant or leaf image or enter a text description to find similar disease cases."
)

uploaded_file = st.file_uploader(
    "📤 Upload an image", type=["jpg", "jpeg", "png"]
)
text_query = st.text_input("💬 Or search by text:")
logger.info(f"Text query: {text_query}")

query_embedding = None
query_type = None

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    logger.info(f"📤 Image uploaded: {uploaded_file.name}")
    query_embedding = get_image_embedding(uploaded_file)
    query_type = "image"
    DISTANCE_THRESHOLD = 0.1

elif text_query:
    query_embedding = get_text_embedding(text_query)
    query_type = "text"
    DISTANCE_THRESHOLD = 0.2

if query_embedding is not None:
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10,
            include=["distances", "metadatas"],
        )
        logger.info(f"✅ Query returned {len(results['ids'][0])} results.")
        distances = results["distances"][0]
        logger.info(f"All distances: {distances}")
    except Exception as e:
        logger.exception(f"❌ Query to ChromaDB failed: {e}")
        st.error("❌ ChromaDB query failed.")
        st.stop()

    # Check for weak matches
    if all(d > DISTANCE_THRESHOLD for d in distances):
        st.warning(
            "⚠️ No close matches found. Try a more specific query or different image."
        )
        logger.info(f"❌ All distances above threshold: {distances}")
        st.stop()

    st.subheader("🔎 Top Similar Results")

    if results["metadatas"] and results["metadatas"][0]:
        seen = set()
        for i, metadata in enumerate(results["metadatas"][0]):
            group_id = metadata.get("group_id")
            if group_id in seen:
                continue
            seen.add(group_id)

            col1, col2 = st.columns([1, 3])
            label = metadata.get("label", "Unknown")
            caption = metadata.get("caption", "-")
            path = metadata.get("path", "N/A")
            distance = distances[i]

            try:
                with col1:
                    if path and os.path.exists(path):
                        st.image(path, width=175)
                    else:
                        st.text("📁 Image not found")
                with col2:
                    st.markdown(f"**Label:** {label}")
                    st.markdown(f"**Caption:** {caption}")
                    st.markdown(f"**Distance:** `{distance:.3f}`")
            except Exception as img_err:
                st.warning(f"⚠️ Unable to display result: {path}")
                logger.warning(
                    f"⚠️ Failed to display image at {path}: {img_err}"
                )
    else:
        st.warning("No similar results found.")
        logger.info("🔍 No matching metadata found in results.")
