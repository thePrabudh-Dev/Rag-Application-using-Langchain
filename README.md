# Rag Application using Langchain
 This repository showcases a Retrieval Augmented Generation (RAG) application built using LangChain and OpenAI. It demonstrates how to leverage large language models (LLMs) to create powerful AI applications that can access and understand information from a variety of sources. The application ingests a dataset of documents, creates vector embeddings for each document, and stores them in a vector database. When a user query is received, the application retrieves the most relevant documents from the database, feeds them to the LLM, and generates a comprehensive response. This approach enables the creation of intelligent chatbots, knowledge bases, and other AI-powered tools that can provide informative and engaging interactions.

# Requirements
```python
pypdf
langchain
chromadb # Vector storage
pytest
boto3```