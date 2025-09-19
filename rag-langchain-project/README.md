# Full-Fledged RAG Project using LangChain and OpenAI

This project is a command-line application that uses the Retrieval-Augmented Generation (RAG) technique to answer questions based on a provided text document.

## Core Technologies
- Python 3.9+
- LangChain
- OpenAI API (for LLMs and Embeddings)
- FAISS (for local vector storage)

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-langchain-project
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   - Open the `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY="sk-..."
     ```

## How to Run

The application is a two-step process:

### Step 1: Ingest the Data
First, you need to process the source document (`data/knowledge_base.txt`) and create a vector store. Run the ingestion script:
```bash
python ingest.py
```
This will create a `vector_store/faiss_index` directory containing the database. You only need to run this once, or whenever you update the `knowledge_base.txt` file.

### Step 2: Run the Q&A Application
Now you can start the application and ask questions:
```bash
python app.py
```
The application will load the vector store and prompt you to ask questions. To quit the application, type `exit`.
