import os
from llama_cpp import Llama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ==== CONFIG ====
MODEL_PATH = "models/gemma-3-12b-it-BF16.gguf"
INDEX_DIR = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONTEXT_SIZE = 2048
TOP_K = 5
# ===============

def load_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    print("🧠 Caricamento modello...")
    return Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_threads=8)

def rag_query(llm, db, query):
    #relevant_docs = db.similarity_search(query, k=TOP_K)
    #context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Sei un esperto di machine learning e AI. Rispondi alla seguente domanda basandoti SOLO sul contesto fornito.

### Contesto:
{query}

### Domanda:
{query}

### Risposta:"""

    response = llm("Ciao, come stai?", max_tokens=512, stop=["###"])
    return response["choices"][0]["text"].strip()

def main():
    llm = load_llm()
    #db = load_index()

    print("\n🤖 Pronto per rispondere. Digita una domanda o Ctrl+C per uscire.\n")
    while True:
        try:
            query = input("📝> ").strip()
            if not query:
                continue
            answer = rag_query(llm, llm, query)
            print("\n📌 Risposta:\n", answer, "\n")
        except KeyboardInterrupt:
            print("\n🛑 Uscita.")
            break

if __name__ == "__main__":
    main()