from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.rag import index_document, ask_question
from pydantic import BaseModel

app = FastAPI(title="RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Document Q&A API is running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    num_chunks = index_document(content, file.filename)
    return {
        "message": f"Successfully indexed {file.filename}",
        "chunks_indexed": num_chunks
    }

@app.post("/ask")
async def ask(body: Question):
    result = ask_question(body.question)
    return result
