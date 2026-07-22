import os
from dotenv import load_dotenv
import streamlit as st
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="PyTutor AI",
    page_icon="🐍",
    layout="wide"
)


# -------------------- Load Environment --------------------
load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()
# -------------------- Load Models --------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_type="similarity")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=1024
    )

    return retriever, llm


retriever, llm = load_models()

#-----------------------css--------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
bg_image = get_base64_image("assets/backgroundIMG.png")
st.markdown(f"""
<style>

/* ---------------- Background ---------------- */

.stApp {{
    background:
        linear-gradient(
            rgba(10,15,30,0.35),
            rgba(10,15,30,0.35)
        ),
        url("data:image/jpg;base64,{bg_image}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* ---------------- Hero ---------------- */

.hero{{
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(12px);
    padding:20px;
    border-radius:20px;
    text-align:center;
    margin-bottom:20px;
}}

.hero h1{{
    font-size:52px;
    font-weight:700;
    color:#FFD43B;
    text-shadow:2px 2px 8px rgba(0,0,0,0.4);
}}

.hero h3{{
    color:#2D2D2D;
}}

.hero p{{
    color:#3F3F46;
}}

</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🐍 PyTutor AI")

    st.markdown("---")

    st.markdown("### 📚 Knowledge Base")
    st.write("Official Python Tutorial")

    st.markdown("### 🤖 LLM")
    st.write("Llama 3.3 70B (Groq)")

    st.markdown("### 🔍 Embeddings")
    st.write("all-MiniLM-L6-v2")

    st.markdown("### 🗂 Vector Store")
    st.write("ChromaDB")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------- Title --------------------
st.title("🐍 PyTutor AI")
st.caption("Python RAG Assistant powered by LangChain + ChromaDB + Groq")

# -------------------- Chat History --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- User Input --------------------
if prompt := st.chat_input("Ask a Python question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history
    history_text = ""

    for msg in st.session_state.messages[:-1]:
        history_text += f'{msg["role"]}: {msg["content"]}\n'

    # Retrieve context
    docs = retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = f"""
You are a Python Programming Assistant.

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

Greeting Rule:

- If the user sends only a greeting such as "Hi", "Hello", "Hey", "Good Morning", "Good Afternoon", "Good Evening", "How are you?", or similar greetings, respond with a friendly greeting.
- Do not use the retrieved context for greetings.
- Do not reply that the answer was not found.
- After greeting the user, politely ask how you can help with Python programming.

Example response:

Hello! 👋 Welcome to PyTutor AI.

I'm your Python Programming Assistant. How can I help you today?
- don't say hello evry time when user ask quesion 

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
{prompt}

Answer:
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(final_prompt)

            answer = response.content

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
