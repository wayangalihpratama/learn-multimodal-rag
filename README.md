# 🌿 Learn Multimodal RAG: Pest & Disease Image Search

A practical implementation of **Multimodal Retrieval-Augmented Generation (RAG)** using **CLIP** and **BLIP** for pest and crop disease detection based on plant and leaf images.

This project uses:
- **CLIP** for image embeddings
- **BLIP** for automatic image captioning
- **ChromaDB** for vector storage
- **Streamlit** for a web UI to search similar images
- **Docker Compose** to containerize the application

---

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
learn-multimodal-rag/
├── app/
│ ├── indexing.py # Index dataset images into ChromaDB
│ ├── main.py # Streamlit app for searching
│ ├── utils.py # Embedding & captioning helpers
│ ├── chroma/ # ChromaDB persistence directory
├── data/
│ └── pest_disease/ # Dataset images organized by label
├── chroma/ # Mounted volume for ChromaDB
├── entrypoint.sh # Entrypoint script: indexing + UI
├── Dockerfile # Docker image config
├── docker-compose.yml # Container setup
└── requirements.txt # Python dependencies
```


---

## 🚀 How to Run

### 1. 📥 Clone the repository

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

- No results returned:
  Ensure that indexing ran before UI started. The entrypoint.sh script handles this, but you can run indexing manually inside the container:

```bash
docker exec -it <container_name> python app/indexing.py
```

- ChromaDB data not saved:
  Make sure the volume ./chroma:/app/app/chroma is properly mounted.

---

## 📚 References
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [BLIP (Salesforce)](https://github.com/salesforce/BLIP)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

👨‍🔬 Author

Built with ❤️ by Wayan Galih Pratama, for educational and research purposes.
