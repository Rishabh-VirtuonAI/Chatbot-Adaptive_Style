import os
import streamlit as st
from sympy import false
from PyPDF2 import PdfReader
# removed unused RecursiveCharacterTextSplitter import (we now do custom splitting)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LRScheduler
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
# from langchain.prompts import PromptTemplate  # already imported above

# Try to import CrossEncoder for re-ranking (optional but recommended)
CROSS_ENCODER_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception as e:
    print("CrossEncoder import failed (re-ranking unavailable):", e)
    CROSS_ENCODER_AVAILABLE = False


def init_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None


def process_pdf(pdf_file):
    """
    Symptom-grouping chunking:
    - For every occurrence of "symptom": capture the block including related keys
      (possible_causes, tests_diagnostics, solutions, knowledge_facts) so one chunk
      contains whole troubleshooting case.
    - Fallback: block-based splitting if no explicit "symptom" found.
    - Build embeddings and FAISS vectorstore.
    - If the context fetched does not matches with the question then return "I dont know about this field."
    """
    import re
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    # Read PDF
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # normalize whitespace
    txt = re.sub(r'\r\n|\r', '\n', text)

    # Find all "symptom" positions
    symptoms_idx = [m.start() for m in re.finditer(r'"symptom"\s*:', txt)]

    chunks = []
    if symptoms_idx:
        for i, start in enumerate(symptoms_idx):
            # default end = next symptom start or end of text
            end = symptoms_idx[i + 1] if (i + 1) < len(symptoms_idx) else len(txt)

            # extend a lookahead window beyond the next symptom to capture trailing keys like "solutions"
            lookahead_end = min(len(txt), end + 3000)  # adjust window if needed
            extended = txt[start:lookahead_end]

            # try to include full "solutions" array if present
            sol_match = re.search(r'"solutions"\s*:\s*\[.*?\]', extended, flags=re.S)
            if sol_match:
                sol_end = sol_match.end()
                chunk_text = extended[:sol_end]
            else:
                # attempt to include up to the next top-level closing brace '}' if present
                brace_match = re.search(r'\}\s*(,|\n)', extended)
                if brace_match:
                    chunk_text = extended[:brace_match.end()]
                else:
                    # fallback: paragraph break or small snippet
                    para_match = re.search(r'\n\s*\n', extended)
                    chunk_text = extended[:para_match.start()] if para_match else extended

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
    else:
        # fallback: generic key-based block splitting
        raw_blocks = re.split(r'(?=("?[A-Za-z0-9_ ]+"?\s*:))', txt)
        buffer = ""
        for part in raw_blocks:
            if re.match(r'^"?[A-Za-z0-9_ ]+"?\s*:', part.strip()):
                if buffer:
                    chunks.append(buffer.strip())
                buffer = part
            else:
                buffer += part
        if buffer:
            chunks.append(buffer.strip())

    # dedupe similar chunks (quick heuristic)
    unique_chunks = []
    seen = set()
    for c in chunks:
        key = c[:180].strip()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # if still empty, save whole text as single chunk
    if not unique_chunks:
        unique_chunks = [txt]

    # build embeddings + FAISS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # create vectorstore
    st.session_state.vectorstore = FAISS.from_texts(unique_chunks, embeddings)
    return f"PDF processed successfully with {len(unique_chunks)} symptom+solution chunks!"


def get_context_with_rerank(question, top_k=10, rerank_top=1):
    """
    1) FAISS semantic search (top_k candidates)
    2) Re-rank candidates with CrossEncoder (if available)
    3) Return top `rerank_top` chunks concatenated as context string
    """
    import re

    if st.session_state.vectorstore is None:
        return ""

    # semantic search
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)
    candidates = [d.page_content for d in docs]

    if not candidates:
        return ""

    # re-rank using CrossEncoder if available
    if CROSS_ENCODER_AVAILABLE:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            pairs = [[question, c] for c in candidates]
            scores = cross.predict(pairs)  # higher = better
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top_chunks = [r[0] for r in ranked[:rerank_top]]
            return "\n\n".join(top_chunks)
        except Exception as e:
            print("Cross-encoder re-rank failed:", e)
            # fallback to FAISS ordering

    # fallback: use FAISS order
    return "\n\n".join(candidates[:rerank_top])


def clean_llm_output(text):
    """
    Post-process LLM output: remove labels like 'ANSWER:' or 'Final response:' and tidy lines.
    """
    import re
    if not text:
        return text
    # remove typical labels
    text = re.sub(r'(?i)\bfinal response:\s*', '', text)
    text = re.sub(r'(?i)\banswer:\s*', '', text)
    # strip and split into lines; keep non-empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # if lines look like a single paragraph, attempt to split by sentence-ending punctuation or semicolons
    if len(lines) == 1 and ('.' in lines[0] or ';' in lines[0]):
        # split on dot followed by space OR semicolon
        pieces = re.split(r'\.\s+|;\s+|\n', lines[0])
        pieces = [p.strip() for p in pieces if p.strip()]
        if len(pieces) > 1:
            lines = pieces
    # ensure each line is a clean bullet-like item
    return "\n".join(lines)


def get_context(question):
    # wrapper to keep compatibility with your existing calls
    return get_context_with_rerank(question, top_k=10, rerank_top=1)


def get_response_chain(question):
    try:
        # Get the context first (re-ranked)
        context = get_context(question)

        # Download and initialize the model
        model_path = hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q4_K_M.gguf",
            local_dir="./models",
            local_dir_use_symlinks=False
        )

        # Initialize the model with LlamaCpp
        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,  # Increased context window size
            n_batch=512,
            n_threads=4,
            temperature=0.0,
            verbose=False
        )

      

        prompt = f"""
            You are a mobile network repair expert and outstanding technician.

            RULES:
            - Understand the user's question first and then generate your answer from the context.
            - Use ONLY the information given in CONTEXT.
            - Do NOT mention symptoms, causes, or diagnostics in the answer.
            - Do NOT create questions, summaries, or explanations.
            - Do NOT invent or assume anything outside the CONTEXT.
            - Read the context keenly and then provide your answer according to the question.
            
            CONTEXT:
            {context}

            QUESTION: {question}

            """

        # Generate response with stricter parameters
        response = llm(
            prompt=prompt,
            temperature=0.0,
            top_p=0.9,
            frequency_penalty=0.8,
            presence_penalty=0.8,
        )

        # Post-process the model output to remove unwanted labels
        cleaned = clean_llm_output(response)
        # For extra safety, also remove any accidental leading "ANSWER:" left
        cleaned = cleaned.lstrip(': ').replace("ANSWER:", "").strip()

        print("question: ", question)
        print("context: ", context)
        print("Final raw response:", response)
        print("Final cleaned response:", cleaned)
        return cleaned if cleaned else "I couldn't generate a response. Please try again."

    except Exception as e:
        import traceback
        error_msg = f"Error in get_response_chain: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return "An error occurred while generating the response."


# main UI (unchanged except function wiring)
def main():
    st.set_page_config(page_title="chatbot", page_icon="🤖")
    st.title("Chatbot - Mobile Networks")
    st.write("Upload a PDF and ask questions about its content.")

    init_session_state()

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            status = process_pdf(uploaded_file)
            st.success(status)

    # Chat interface
    if st.session_state.vectorstore is not None:
        user_question = st.text_input("Ask a question about the document:")

        if user_question:
            with st.spinner("Generating response..."):
                response = get_response_chain(user_question)

                # Update conversation history
                st.session_state.conversation_history.append({
                    "user": user_question,
                    "assistant": response
                })

                # Display conversation
                for chat in st.session_state.conversation_history:
                    with st.chat_message("user"):
                        st.write(chat["user"])
                    with st.chat_message("assistant"):
                        st.write(chat["assistant"])

    # Display system info
    st.sidebar.markdown("### System Information")
    st.sidebar.write(f"Using device: {'GPU 🔥' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
