#!/usr/bin/env python3
"""FastAPI untuk inference LLM dengan RAG (Memori Tanpa Batas)"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Import RAG pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.pipeline import query_rag

app = FastAPI(
    title="AI Clone API (Dengan Memori Tanpa Batas)",
    description="API untuk inference LLM menggunakan RAG agar memiliki memori dokumen tak terbatas."
)

# --- Konfigurasi Model ---
MODEL_PATH = os.path.abspath("./models/finetuned/")
if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    MODEL_PATH = "mistralai/Mistral-7B-v0.1"

# Muat tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Inisialisasi inference pipeline
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95
)

# --- Model Request & Response ---
class QueryRequest(BaseModel):
    prompt: str
    use_rag: bool = True
    system_prompt: str = "Anda adalah ahli AI yang sangat bermanfaat."

class QueryResponse(BaseModel):
    hasil: str
    konteks_dibutuhkan: str
    source_dokumen: str = ""

# --- Endpoint API ---
@app.post("/query", response_model=QueryResponse)
async def generate_text(data: QueryRequest):
    # Step 1: Cari konteks relevan jika RAG diaktifkan
    context, source_file = "", ""
    prompt_final = data.prompt

    if data.use_rag:
        try:
            context = query_rag(data.prompt)
            prompt_final = f"{data.system_prompt}\nBerdasarkan konteks yang ditemukan:\n{context}\n\nPertanyaan: {data.prompt}"
        except Exception as e:
            context = f"[ERROR MENCARI KONTEKS] {str(e)}"

    # Step 2: Generate jawaban
    try:
        hasil = llm(prompt_final, do_sample=True)[0]["generated_text"]
        # Hilangkan prompt duplikat
        jawaban = hasil.replace(prompt_final, "").strip()
        jawaban = jawaban.replace(data.prompt, "", 1).strip()
    except Exception as e:
        jawaban = f"[ERROR GENERATE] {str(e)}"

    return QueryResponse(
        hasil=jawaban,
        konteks_dibutuhkan=context[:500] if context else "Tidak ada",
        source_dokumen=source_file
    )

@app.get("/")
def root():
    return {
        "pesan": "AI Clone API aktif! Pakai /docs untuk API.",
        "fitur": ["RAG Memori Tanpa Batas", "FastAPI", "LLM Inference"],
        "endpoint": "/query",
        "curl_example": 'curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d \'{"prompt": "Apa itu RAG?", "use_rag": true}\''
    }

@app.get("/health")
def health():
    return {"status": "OK", "model_loaded": MODEL_PATH}