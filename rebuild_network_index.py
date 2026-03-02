from utils.vector_builder import build_faiss_index
from utils.external_vector_builder import build_external_vector_db
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    domain = "backlight"
    kb_path = f"domains/{domain}/knowledge_base.txt"
    kb_path_external = f"domains/{domain}/knowledge_base_external.txt"
    print(f"Rebuilding FAISS index for domain: {domain}")
    build_faiss_index(domain, kb_path)
    build_external_vector_db(domain, kb_path_external)
    print("Index rebuilding complete.")
