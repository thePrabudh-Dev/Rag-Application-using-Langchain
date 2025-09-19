import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Paths and constants
DATA_PATH = "data/knowledge_base.txt"
DB_FAISS_PATH = "vector_store/faiss_index"

def run_ingestion() -> None:
    """Run the ingestion pipeline: load data, split, embed, and persist FAISS index."""
    # Load environment variables (expects OPENAI_API_KEY in .env)
    load_dotenv()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Source document not found at {DATA_PATH}. Please ensure the file exists."
        )

    # 1) Load the raw document
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # 2) Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3) Create embeddings
    embeddings = OpenAIEmbeddings()

    # 4) Build FAISS vector store from chunks and embeddings
    db = FAISS.from_documents(chunks, embeddings)

    # Ensure target directory exists and persist locally
    Path("vector_store").mkdir(parents=True, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print(f"Ingestion complete. FAISS index saved to {DB_FAISS_PATH}.")


if __name__ == "__main__":
    run_ingestion()
