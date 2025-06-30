# 🌿 Multimodal RAG: Pest & Disease Image Search


[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/your-username/learn-multimodal-rag) [![Docker](https://img.shields.io/badge/Built%20with-Docker-blue?logo=docker)](https://www.docker.com/) [![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-orange?logo=streamlit)](https://streamlit.io/) [![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple)](https://www.trychroma.com/) [![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/downloads/release/python-3100/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


This project is a multimodal Retrieval-Augmented Generation (RAG) application that lets users upload images of plant diseases and retrieve visually similar samples using image embeddings, captions, and ChromaDB vector search.

## 🧰 Features
- 🖼️ Upload an image of a pest/disease
- 🤖 BLIP captioning & image embedding
- 🔍 Find visually similar cases
- 💾 Vector database powered by ChromaDB server
- 🌐 Full Dockerized environment


## 📦 Tech Stack

| Component         | Description                         |
|------------------|-------------------------------------|
| **CLIP**         | Visual encoder for image embeddings |
| **BLIP**         | Vision-language model for captions  |
| **ChromaDB**     | Vector database for similarity search |
| **Streamlit**    | UI for uploading and searching images |
| **Docker Compose** | Environment orchestration            |

---

## 📁 Project Structure

```bash
.
├── app/
│   ├── main.py              # Streamlit UI
│   ├── indexing.py          # Dataset indexer
│   ├── utils.py             # BLIP + embedding logic
│   ├── logs/                # App logs
│   └── chroma/              # Chroma persistence (if using local)
├── data/                    # Input images to be indexed
│   └── pest_disease/
├── chroma-data/             # Persisted ChromaDB volume (optional)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


---

## 🚀 How to Run

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/your-username/learn-multimodal-rag.git
cd learn-multimodal-rag
```

### 2. 🖼️ Prepare the Dataset

Download the [Crop Pest and Disease Detection dataset](https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection) and extract it into:

```bash
data/pest_disease/
```

Ensure that the folder structure is like:

```bash
data/pest_disease/
├── Tomato leaf blight/
│   ├── image1.jpg
│   ├── image2.jpg
├── Aphid/
│   ├── image1.jpg
```

### 3. 🐳 Build and Start the Project

```bash
docker-compose up --build
```

This will:
- Run image indexing using indexing.py
- Start Streamlit on port 8501

Open in your browser:

```bash
http://localhost:8501
```

### 🔧 Docker Compose Overview

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - ./chroma-data:/chroma
    restart: unless-stopped

  multimodal-rag:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./app:/app/app
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - chromadb
    restart: unless-stopped

```

---

## 🔍 How It Works

1. At startup, the app indexes all images:
   - Generates CLIP embeddings
   - Uses BLIP to generate a caption
   - Combines the label and caption
   - Stores it in ChromaDB with metadata

2. User uploads a query image:
   - The image is embedded using CLIP
   - The vector is searched in ChromaDB
   - Similar images and metadata are displayed

---

## 🛠️ Useful Commands

Rebuild and rerun the app

```bash
docker-compose down
docker-compose up --build
```

View logs

```bash
docker-compose logs -f
```

---

## ⚠️ Troubleshooting

- No similar results?
  Make sure images are indexed by checking logs.
- Still empty?
  Check logs in app/logs/app.log.
- Embeddings or captions fail?
  Make sure your models are downloaded and supported.
- Collection empty in main.py?
  Ensure indexing completed before Streamlit starts.

---

## 📚 Acknowledgments
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [BLIP (Salesforce)](https://github.com/salesforce/BLIP)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

👨‍🔬 Author

Built with ❤️ by Wayan Galih Pratama, for educational and research purposes.
