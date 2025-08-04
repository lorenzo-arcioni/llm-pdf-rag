import os
import json
from pathlib import Path
from tqdm import tqdm

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ==== CONFIG ====
PDF_DIR = "pdfs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = "data/chunks.jsonl"
INDEX_DIR = "data/faiss_index"
# ===============

def extract_and_chunk():
    print("📚 Caricamento PDF...")
    all_docs = []
    for file in Path(PDF_DIR).rglob("*.pdf"):
        loader = PyPDFLoader(str(file))
        all_docs.extend(loader.load())

    print("✂️ Segmentazione in chunk...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(all_docs)

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for doc in split_docs:
            f.write(json.dumps({"text": doc.page_content, "metadata": doc.metadata}) + "\n")

    return split_docs

def build_and_save_index(docs):
    print("📌 Costruzione indice FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_DIR)
    print("✅ Indice salvato in", INDEX_DIR)

if __name__ == "__main__":
    docs = extract_and_chunk()
    build_and_save_index(docs)
