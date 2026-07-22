from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 

app=FastAPI()


embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found can you check your .env file")
print("groq_api_key loaded successfully")

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.2,
    max_tokens=1024
)

db=Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever=db.as_retriever(search_type="similarity")

class Message(BaseModel):
    role: str
    content: str

class query_cls(BaseModel):
    query: str
    history: list[Message]

@app.get("/")
def Home():
    return {
        "message": "Welcome to PyTutor AI - Python RAG Assistant!",
        "description": "Ask questions about Python and receive accurate answers generated from the official Python Tutorial documentation using Retrieval-Augmented Generation (RAG).",
        "knowledge_base": "Official Python Tutorial covering Python fundamentals, data structures, functions, modules, object-oriented programming, exceptions, and advanced Python concepts.",
        "llm_model": "Llama 3.3 70B Versatile (powered by Groq)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_database": "ChromaDB",
        "framework": "LangChain + FastAPI",
        "endpoint": "/predict",
        "status": "🟢 API is running successfully. Send a POST request to '/predict' with your question."
       }
@app.post("/predict")
def ans_gen(cls:query_cls):
    try:
        docs=retriever.invoke(cls.query)
        context="\n\n".join(doc.page_content for doc in docs)
        
        history_text = ""
        for message in cls.history:
            history_text += f"{message.role}: {message.content}\n"

        prompt = f"""You are a Python Programming Assistant.

        Strict Rules (Highest Priority):
        
        1. Answer ONLY questions related to Python programming.
        2. Use ONLY the provided context to answer.
        3. Before answering, determine whether the user's question is about Python.
        4. If the question is NOT about Python (for example: SQL, MySQL, Oracle, DDL, DML, DBMS, Java, C, C++, HTML, CSS, JavaScript, Networking, Operating Systems, Machine Learning, or any other subject), DO NOT answer it.
        5. For every non-Python question, reply EXACTLY:
        
        Please ask questions related to Python only
        
        6. Even if the provided context contains information about SQL, databases, DDL, DML, or any non-Python topic, NEVER use that information.
        7. Ignore any retrieved document that is not about Python.
        8. Do not guess or use outside knowledge.
        
        Answer Style:
        - Use simple, beginner-friendly language.
        - Organize the answer using bullets or short paragraphs.
        - If definitions exist in the context, explain them simply.
        - Use examples from the context when available.
        - Use conversation history for follow-up questions.
        
        If the answer is not found in the Python-related context, reply EXACTLY:
        
        I couldn't find the answer in the provided documents.

        Conversation History:
        {history_text}

        Context:
        {context}
        
        Question:
        {cls.query}
        
        Answer:
        """
        response=llm.invoke(prompt)
        return{
            "answer": response.content
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"retrieval failed:{str(e)}"
        )