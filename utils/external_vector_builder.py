import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import json
import torch
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_external_vector_db(domain:str,file_path, model_name="BAAI/bge-m3"):
    """
    Reads the TXT file, creates embeddings, and saves a FAISS index.
    """
    # 1. Load the text
    with open(file_path, "r") as f:
        raw_text = f.read()

    # 2. Split into Steps (We split by 'STEP_' so each step is its own search result)
    # This preserves the procedural integrity
    steps = raw_text.split("STEP_")
    documents = [Document(page_content="STEP_" + s.strip()) for s in steps if s.strip()]

    # 3. Initialize Embedding Model
    print(f"Loading embedding model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create Vector Store
    print("Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 5. Save Locally
    index_dir = os.path.join("domains", domain, "faiss_external_backlight")
    os.makedirs(index_dir, exist_ok=True)
    index_name = "faiss_external_backlight"
    vectorstore.save_local(index_dir)
    
    print(f"Success! Vector index saved as: {index_name}")
    return index_name

# Execution
if __name__ == "__main__":
    build_external_vector_db("backlight_external_kb.txt")