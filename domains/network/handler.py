from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableSequence

try:
    from sentence_transformers import CrossEncoder
    import torch
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

# Load once per process
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local("domains/network/faiss_index", 
            embeddings=embedding,
             allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}) # Fetch more candidates for re-ranking


# Initialize the CrossEncoder model for re-ranking if available
if CROSS_ENCODER_AVAILABLE:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
else:
    cross_encoder = None
# model="llama3.1:8b-instruct-q4_K_M",
# model="gemma3:12b-it-q4_K_M",
llm = Ollama(
    model="gemma3:12b-it-q4_K_M",
              temperature=0.5,
              top_p=0.5,
              num_ctx=4096,
                # num_ctx = 1024,
                # num_ctx = 2048,
                top_k=64,
                repeat_penalty=1.1)

# Load audio-specific prompt
with open("domains/network/prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

# prompt = PromptTemplate(
#     input_variables=["context", "question", "conversation_history"],
#     template=prompt_template
# )

# # Compose the custom chain
# chain = prompt | llm

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt}
# )


def get_context_with_rerank(question,conversation_history=None, top_k=10, rerank_top=3):
    """
    1) FAISS semantic search (top_k candidates)
    2) Re-rank candidates with CrossEncoder (if available)
    3) Return top `rerank_top` chunks concatenated as context string
    """
    import re

    if vectorstore is None:
        return ""
    
    if conversation_history:
        # Extract last 2-3 relevant exchanges
        history_text = "\n".join([f"User: {m['user_message']}\nBot: {m['bot_response']}" 
                                for m in conversation_history[-3:]])
        enhanced_query = f"{history_text}\nCurrent question: {question}"
    else:
        enhanced_query = question

    # semantic search
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(enhanced_query)
    candidates = [d.page_content for d in docs]

    if not candidates:
        return ""

    # re-rank using CrossEncoder if available
    if CROSS_ENCODER_AVAILABLE:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            cross = cross_encoder
            pairs = [[enhanced_query, c] for c in candidates]
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



def get_context(question, chat_history=None):
    # wrapper to keep compatibility with your existing calls
    return get_context_with_rerank(question, conversation_history=chat_history,top_k=10, rerank_top=1)


def rewrite_query(user_query: str) -> str:
    """
    Use LLM to rewrite Hinglish or messy queries into clean,
    professional English queries for better embedding search.
    """
    rewrite_prompt = f"""
You are a professional query rewriter for mobile hardware troubleshooting.

Your task:
Rewrite the given user query into a clean, professional English query while preserving the **exact technical meaning**.
Do not add, guess, or change the intent.

Guidelines:
1. Keep the device model and problem exactly as mentioned.
2. Expand Hinglish or slang into clear English.
3. Standardize voltage, IC, and PFO names (e.g., "Vreg 1.8v" → "VREG 1.8V").
4. Always keep IC numbers, part codes, and voltages exactly as they appear.
5. Do NOT introduce assumptions (e.g., band configuration, software issues) if the user didn’t mention them.
6. Keep it concise: one clear sentence.
7. Focus strictly on **hardware and network troubleshooting**.

Example:
Input: "Sir, Nokia 3.1 set ka network problem hy.. Radio on hy.. 4g pfo change krdu.. Or ek bat sir.. Pfo board me h 7215m-41..... Dusra pfo h 7219m-41 laga sekh ta hu"
Output: "The Nokia 3.1 is experiencing network issues even though the radio is on. Consider replacing the 4G PFO, as the current board has 7215m-41 and may be replaced with 7219m-41."

Now rewrite this user query faithfully:

User Query: "{user_query}"

Rewritten Query:

    """

    try:
        rewritten = llm.invoke(rewrite_prompt).strip()
        if rewritten:
            return rewritten
        else:
            return user_query  # fallback if rewrite fails
    except Exception as e:
        print("Query rewriting failed:", e)
        return user_query  # fallback

def chat_with_user(user_query, chat_history, username):

    # Step 1: Rewrite query first
    print("Original Query:", user_query)
    clean_query = rewrite_query(user_query)
    print("Rewritten Query:", clean_query)
    # Get the context first (re-ranked)
    context = get_context(clean_query, chat_history)
    # context = get_context(user_query, chat_history)

    # The llm is already initialized in the global scope, so no need to download or re-initialize.

    prompt = prompt_template.format(
        context=context,
        question=clean_query,
        conversation_history="\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in chat_history),
        username=username
    )

    # Generate response with stricter parameters
    response = llm.invoke(
        prompt
    )


    cleaned = clean_llm_output(response)
    # For extra safety, also remove any accidental leading "ANSWER:" left
    # cleaned = cleaned.lstrip(': ').replace("ANSWER:", "").strip()

    print("question: ", user_query)
    print("context: ", context)
    print("Final raw response:", response)
    print("Final cleaned response:", cleaned)
    return {"response": cleaned if cleaned else "I couldn't generate a response. Please try again."}


    # older code for getting the response
    # # Step 1: Initial retrieval
    # docs = retriever.get_relevant_documents(user_query)
    
    # # Step 2: Re-rank documents if CrossEncoder is available
    # if cross_encoder and docs:
    #     doc_contents = [doc.page_content for doc in docs]
    #     pairs = [[user_query, content] for content in doc_contents]
    #     scores = cross_encoder.predict(pairs)
        
    #     # Combine docs with their scores and sort
    #     scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
    #     # Select the top N documents after re-ranking
    #     docs = [doc for doc, score in scored_docs[:3]] # Keep top 3

    # # Reconstruct the context by joining the content of the top documents.
    # # The old grouping logic is no longer needed due to the improved chunking strategy.
    # context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # formatted_history = "\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in chat_history)
    
    # print(f"Context: {context}")
    # response = chain.invoke({
    #     "context": context,
    #     "question": user_query,
    #     "conversation_history": formatted_history,
    #     'username':username
    # })

    # return {"response": response}
