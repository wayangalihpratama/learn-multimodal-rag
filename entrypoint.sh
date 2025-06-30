#!/bin/sh

echo "⏳ Waiting for ChromaDB to be ready..."
until curl -s http://chromadb:8000/ > /dev/null; do
    echo "🔁 ChromaDB not ready yet..."
    sleep 1
done

echo "🛠️  Running indexing..."
python app/indexing.py

echo "🚀 Starting Streamlit..."
streamlit run app/main.py --server.port=8501
