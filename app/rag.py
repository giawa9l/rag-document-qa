import os
import chromadb
import pdfplumber
import io
from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("documents")

def extract_text_from_pdf(file_bytes: bytes) -> list:
    chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                chunks.extend(paragraphs)
    return chunks

def get_embedding(text: str) -> list:
    response = gemini_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def index_document(file_bytes: bytes, filename: str) -> int:
    chunks = extract_text_from_pdf(file_bytes)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return len(chunks)

def ask_question(question: str) -> dict:
    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])
    prompt = f"""Answer using ONLY this context. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": results["documents"][0]
    }
