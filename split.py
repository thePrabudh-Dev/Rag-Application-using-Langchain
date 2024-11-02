from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.schema.document import Document
DATA_PATH="data\gemicitabine"
#-----------------------------------------------
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

#-----------------------------------------------
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_splitter.split_documents(documents)



documents =  load_documents()
chunk = split_documents(documents)
print(chunk[0])