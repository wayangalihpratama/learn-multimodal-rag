#!/bin/sh

echo "🛠️  Running indexing..."
python app/indexing.py

echo "🚀 Starting Streamlit..."
streamlit run app/main.py --server.port=8501
