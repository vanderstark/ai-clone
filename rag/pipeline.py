import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Setup Embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Simpan ke Vector DB (Memori Tanpa Batas)
    db = Chroma.from_documents(docs, embeddings, persist_directory="./rag/chroma_db")
    db.persist()
    print(f"✅ Dokumen {file_path} berhasil di-ingest ke memori AI.")

def query_rag(question):
    db = Chroma(persist_directory="./rag/chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return context
