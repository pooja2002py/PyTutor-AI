from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

loader=DirectoryLoader(
    ".Data/python_ebook.pdf",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents=loader.load()

print(f"total number of documents: {len(documents)}")

text_spliter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=text_spliter.split_documents(documents)

print(f"total number of chunk: {len(chunks)}")

embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Documents store successfully")