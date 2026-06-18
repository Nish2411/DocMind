import os
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from groq import Groq

from pipeline.loader import extract_text
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_texts, embed_query
from pipeline.store import (
    save_document_and_chunks,
    retrieve_similar_chunks,
    list_all_documents,
    delete_document_by_id,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEN_MODEL = "gemini-2.5-flash"

def get_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── Routes ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    filename = file.filename
    logger.info(f"Received: {filename} ({len(data)} bytes)")

    # 1. Extract text
    try:
        text = extract_text(filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted.")

    # 2. Chunk
    chunks = chunk_text(text)
    logger.info(f"  {len(chunks)} chunks created")

    # 3. Embed
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 4. Store
    ext = Path(filename).suffix.lower().lstrip(".")
    doc_id = save_document_and_chunks(filename, ext, chunks, embeddings)
    logger.info(f"  Stored as {doc_id}")

    return {
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "characters": len(text),
    }


@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Embed question
    try:
        q_embedding = embed_query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {e}")

    # 2. Retrieve relevant chunks
    chunks = retrieve_similar_chunks(q_embedding)

    if not chunks:
        return {
            "answer": "No documents found. Please upload some documents first.",
            "sources": []
        }

    # 3. Build context
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}: {c['filename']}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # 4. Generate answer
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document excerpts.
Only use information from the sources below. If the answer isn't there, say so clearly.
Cite which source(s) you used.

DOCUMENT EXCERPTS:
{context}

QUESTION: {req.question}

ANSWER:"""

    try:
        client = get_client()
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {
        "answer": answer,
        "sources": [
            {
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "similarity": round(float(c["similarity"]), 4),
                "snippet": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
            }
            for c in chunks
        ],
    }


@app.get("/documents")
async def documents():
    return {"documents": list_all_documents()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    found = delete_document_by_id(doc_id)
    if not found:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True, "deleted": doc_id}