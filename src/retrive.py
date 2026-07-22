from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os 
from langchain_groq import ChatGroq

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db=Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever=db.as_retriever(search_type="similarity")

query="what is python"

docs=retriever.invoke(query)

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(f"API key not found please check your .env file")
print("API key loaded successfully")

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.2,
    max_tokens=1024
)
for i, doc in enumerate(docs, 1):
    print(f"Chunk {i}")
    print(doc.page_content)
    print("-" * 50)
context="\n\n".join(doc.page_content for doc in docs)

prompt = f"""
You are an expert Python tutor.

Your task is to answer questions using ONLY the provided document context.

Guidelines:
- Answer in a clear, beginner-friendly manner.
- Organize the answer into short paragraphs or bullet points when appropriate.
- If the context contains definitions, explain them simply.
- If the context includes examples, use or summarize them.
- Do not use any information that is not explicitly present in the context.
- If the context does not contain enough information to answer the question, reply exactly:
  "I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""
response=llm.invoke(prompt)
print(response.content)