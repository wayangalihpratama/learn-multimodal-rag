import streamlit as st
from utils import get_image_embedding
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
        metadata={"hnsw:space": "cosine"},  # 👈 ensures cosine distance
    )
    logger.info("✅ ChromaDB initialized.")
    logger.info(f"✅ Chroma collection count: {collection.count()}")

    # Check if collection has data
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
    "Upload a plant or leaf image to find visually similar disease cases."
)

uploaded_file = st.file_uploader(
    "📤 Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    logger.info(f"📤 Image uploaded: {uploaded_file.name}")

    try:
        with st.spinner("🔍 Generating embedding and querying ChromaDB..."):
            query_embedding = get_image_embedding(uploaded_file)
            logger.info("✅ Image embedding generated.")
            logger.info(f"🔢 Embedding length: {len(query_embedding)}")

            try:
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=10,
                    include=["distances", "metadatas"],
                )
                logger.info(
                    f"✅ Query returned {len(results['ids'][0])} results."
                )

                distances = results["distances"][0]
                logger.info(f"All distances: {distances}")

            except Exception as e:
                logger.exception(f"❌ Query to ChromaDB failed: {e}")
                st.error("❌ ChromaDB query failed.")
                st.stop()

        # ✅ Spinner ends here — now display results or warning
        if all(d > DISTANCE_THRESHOLD for d in distances):
            st.warning(
                "⚠️ No close matches found. This image may not belong to any known category."
            )
            logger.info(f"❌ All distances above threshold: {distances}")
            st.stop()

        st.subheader("🔎 Top Similar Results")
        # st.json(results)  # show raw results

        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                col1, col2 = st.columns([1, 3])

                label = metadata.get("label", "Unknown")
                caption = metadata.get("caption", "-")
                path = metadata.get("path", "N/A")
                distance = distances[i]

                try:
                    with col1:
                        st.image(metadata["path"], width=175)
                        logger.info(f"🖼️ Displayed image from: {path}")
                    with col2:
                        st.markdown(f"**Label:** {label}")
                        st.markdown(f"**Caption:** {caption}")
                        st.markdown(f"**Distance:** `{distance:.3f}`")

                except Exception as img_err:
                    st.warning(f"⚠️ Unable to display image: {path}")
                    st.text(f"Image path: {path}")
                    logger.warning(
                        f"⚠️ Failed to display image at {path}: {img_err}"
                    )
        else:
            st.warning("No similar results found.")
            logger.info("🔍 No matching metadata found in results.")

    except Exception as err:
        st.error("Something went wrong during image search.")
        logger.exception(f"❌ Error during image processing or search: {err}")
