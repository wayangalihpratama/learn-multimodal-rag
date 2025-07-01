import chromadb
from app.indexing import index_images

chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
collection = chroma_client.get_or_create_collection(
    name="pest_disease",
    metadata={"hnsw:space": "cosine"},  # 👈 ensures cosine distance
)

count = collection.count()
print(f"ChromaDB collection count: {count}")
if collection.count() == 0:
    index_images()
else:
    print("✅ ChromaDB already contains data. Skipping indexing.")
