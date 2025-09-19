import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()

DB_FAISS_PATH = "vector_store/faiss_index"

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main() -> None:
    # Prepare embeddings and load local vector DB
    embeddings = OpenAIEmbeddings()
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        print("FAISS index not found. Please run 'python ingest.py' first to build the index.")
        return

    # Initialize the chat model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Configure retriever to fetch top-k relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # RAG prompt template
    prompt_template = """
You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the answer is not found in the context, respond with "I cannot find the answer in the provided knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # RAG chain: retrieve context, compose prompt, ask LLM, parse text output
    rag_chain = (
        {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Interactive Q&A loop
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        answer = rag_chain.invoke(user_input)
        print("\nAnswer:\n" + str(answer) + "\n")
