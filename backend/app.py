# backend/app.py
import os
import sqlite3
import shutil
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from typing import Optional

# === Config ===
UPLOAD_DIR = "uploaded"
INDEX_DIR = "faiss_indexes"
DB_PATH = "resources.db"
MODEL = "qwen:1.8b"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Helpers ===
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def create_faiss_vector_store(text: str, index_path: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(index_path)


def load_faiss_vector_store(index_path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    return None


def build_qa_chain(index_path: str):
    vector_store = load_faiss_vector_store(index_path)
    if not vector_store:
        return None
    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model=MODEL)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    return RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)


def search_resource_multi(question: str):
    words = question.lower().split()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for word in words:
        cur.execute(
            "SELECT keyword, document_link, video_link FROM resources WHERE LOWER(keyword) LIKE ?",
            (f"%{word}%",),
        )
        results.extend(cur.fetchall())
    conn.close()
    return list({(d[0], d[1], d[2]) for d in results})


# === API Models ===
class Question(BaseModel):
    question: str
    file_name: Optional[str] = None


# === Routes ===
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create FAISS index
    index_path = os.path.join(INDEX_DIR, f"{file.filename}.index")
    text = extract_text_from_pdf(file_path)
    create_faiss_vector_store(text, index_path)

    return {
        "message": f"Uploaded and indexed {file.filename}",
        "file_name": file.filename,
    }


@app.post("/ask")
async def ask_question(q: Question):
    try:
        llm = OllamaLLM(model=MODEL)

        if q.file_name:
            index_path = os.path.join(INDEX_DIR, f"{q.file_name}.index")
            qa_chain = build_qa_chain(index_path)
            if not qa_chain:
                return {"answer": "No index found. Please re-upload the document."}

            if "summarize" in q.question.lower():
                vector_store = load_faiss_vector_store(index_path)
                docs = vector_store.as_retriever().get_relevant_documents("")
                full_text = "\n\n".join([d.page_content for d in docs])
                prompt = f"Summarize the following document:\n\n{full_text}"
                answer = llm(prompt=prompt)
                return {"answer": answer}
            else:
                answer = qa_chain.run(q.question)
                return {"answer": answer}
        else:
            # Search DB
            db_results = search_resource_multi(q.question)
            if db_results:
                return {
                    "resources": [
                        {"keyword": d[0], "doc_path": d[1], "video_link": d[2]}
                        for d in db_results
                    ]
                }
            else:
                answer = llm(prompt=q.question)
                return {"answer": answer}

    except Exception as e:
        # Instead of "internal error", return JSON
        return JSONResponse(
            status_code=500, content={"error": f"Internal server error: {str(e)}"}
        )


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=file_name)
    return JSONResponse(status_code=404, content={"error": "File not found"})
