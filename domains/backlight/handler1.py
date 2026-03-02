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
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local("domains/backlight/faiss_index", 
            embeddings=embedding,
             allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}) # Fetch more candidates for re-ranking


# Initialize the CrossEncoder model for re-ranking if available
if CROSS_ENCODER_AVAILABLE:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    cross_encoder = CrossEncoder('BAAI/bge-reranker-large', device=device)
else:
    cross_encoder = None
# model="llama3.1:8b-instruct-q4_K_M",
# model="gemma3:12b-it-q4_K_M",
llm = Ollama(
    model="gemma3:12b-it-q4_K_M",
              temperature=0.3,
              top_p=0.5,
              num_ctx=4096,
                # num_ctx = 1024,
                # num_ctx = 2048,
                top_k=64,
                repeat_penalty=1.1)

# Load audio-specific prompt
with open("domains/backlight/prompt.txt", "r", encoding="utf-8") as f:
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


def get_context_with_rerank(question,conversation_history, top_k=10, rerank_top=1):
    """
    1) FAISS semantic search (top_k candidates)
    2) Re-rank candidates with CrossEncoder (if available)
    3) Return top `rerank_top` chunks concatenated as context string
    """
    import re

    if vectorstore is None:
        return ""
    
    # if conversation_history:
    #     # Extract last 2-3 relevant exchanges
    #     history_text = "\n".join([f"User: {m['user_message']}\nBot: {m['bot_response']}" 
    #                             for m in conversation_history[-3:]])
    #     enhanced_query = f"{history_text}\nCurrent question: {question}"
    # else:
    #     enhanced_query = question

    if conversation_history:
        # Extract last 2-3 relevant exchanges
        history_text = "\n".join([f"User: {m['user_message']}\nBot: {m['bot_response']}" 
                                for m in conversation_history[-1:]])
        enhanced_query = f"Current question: {question}\n History: {history_text}"
    else:
        enhanced_query = question

    # semantic search
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    print(f"the enhanced query is {enhanced_query}")
    docs = retriever.invoke(enhanced_query)
    candidates = [d.page_content for d in docs]
    print(f"the candidates are {candidates}")

    if not candidates:
        return ""

    # re-rank using CrossEncoder if available
    if CROSS_ENCODER_AVAILABLE:
        print("Cross-encoder re-rank is working")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
            cross = cross_encoder
            pairs = [[enhanced_query, c] for c in candidates]
            scores = cross.predict(pairs)  # higher = better
            print(f"scores: {scores}")
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            print(f"ranked: {ranked}")
            top_chunks = []
            for a in ranked[0:rerank_top]:
                if a[1]< 0.00:
                    top_chunks.append("there not enough data in the knowledge base")
                    break
                elif a[1]>=0.80:
                    # if the top chunk has a score greater than 0.80 , then include it 
                    top_chunks.append(a[0])
                else:
                    # if the top chunk has score lesser than 0.80 , then go foe fault normalization map logic 
                    normalize_fault = " "
                    top_chunks.append(normalize_fault)
            # top_chunks = [if p[1]> 0.80 for p in ranked[:rerank_top] then r[0] for r in ranked[:rerank_top] else "there not enough data in the knowledge base"]
            print(f"top_chunks: {top_chunks}")
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

# ---------------- FAULT NORMALIZATION ---------------- #

FAULT_NORMALIZATION_MAP = {
    "NO_BACKLIGHT": [
        "no backlight",
        "backlight nahi",
        "backlight nahi aa rahi",
        "display andhera",
        "screen dark",
        "backlight na aana",
        "no back light"
    ],
    "DIM_BACKLIGHT": [
        "backlight dim",
        "light kam",
        "brightness kam",
        "display dim"
    ],
    "BRIGHTNESS_FIXED": [
        "brightness fix",
        "brightness 100%",
        "light full",
        "backlight fix ho jana"
    ]
}

# ---------------- FAULT → PIN MAP ---------------- #

FAULT_PIN_MAP = {
    "AL65": {
        "NO_BACKLIGHT": [6, 3, 4, 5, 1],
        "DIM_BACKLIGHT": [4, 5, 1],
        "BRIGHTNESS_FIXED": [5, 1]
    }
}



def normalize_fault(user_query: str):
    q = user_query.lower()
    for norm_fault, variants in FAULT_NORMALIZATION_MAP.items():
        for v in variants:
            if v in q:
                return norm_fault
    return None

def detect_direct_pin(user_query: str, kb):
    q = user_query.lower()

    for pin in kb.get("pin_details", []):
        if str(pin["pin_number"]) in q:
            return pin["pin_number"]

        if pin["pin_name"].lower() in q:
            return pin["pin_number"]

    return None

def get_pin_specific_context(pin_number, top_k=5):
    query = f"pin number {pin_number}"
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs[:top_k])


def get_context(question, chat_history=None):
    # wrapper to keep compatibility with your existing calls
    return get_context_with_rerank(question, conversation_history=chat_history,top_k=15, rerank_top=2)

def rewrite_backlight_query(user_query: str, conversation_history, context) -> str:
    """
    Use LLM to rewrite Hinglish or messy queries into clean,
    professional English queries for better embedding search.
    """
    rewrite_prompt = f"""
        You are a highly specialized **Hinglish to Technical English Query Translator** for Mobile Hardware Troubleshooting.
        Your sole task is to convert the user's Romanized Hinglish troubleshooting query into a single, highly precise, grammatically correct English sentence.

        ---
        **TECHNICAL CONTEXT (FOR REFERENCE ONLY):**
        The following context has been retrieved from the KB. Use it ONLY to understand the technical terms and entities (IC codes, lines, voltages) in the User Query, but DO NOT translate this context or use it in the output.
        **{context}**
        ---

        **CONVERSATION CONTEXT (For Reference Only):**
        The following is the recent history of the ongoing troubleshooting session. Use this context to resolve any ambiguity or pronoun references (like 'woh' or 'us par') in the User Query.
        **{conversation_history}**
        ---
        
        ... (EXAMPLES section remains here) ...

        **STRICT RULES FOR OUTPUT:**
        1. **DO NOT** write any greetings, explanations, or additional text.
        2. **DO NOT** add any knowledge not present in the User Query.
        3. **DO NOT** modify the User Query words until most needed. Preserve the english words present in the user query.
        4. **PRESERVE** all technical codes, part numbers (like IC, OVP, SW, LX), and voltage values exactly as they appear.
        5. **OUTPUT** only the final, translated English query sentence, enclosed in triple backticks, and nothing else.
        ---

        **User Query:** {user_query}

        **Translated English Query:**
    """
    try:
        rewritten = llm.invoke(rewrite_prompt).strip()
        if rewritten:
            return rewritten
        else:
            return user_query  # fallback if rewrite fails
    except Exception as e:
        print("Backlight query rewriting failed:", e)
        return user_query  # fallback

def RewriteConvo(user_query,chat_history) ->str:
    rewrite_convo = f'''
    You are a highly specialized **Hinglish Conversational Query Rewriter (H-CQR)** for Mobile Hardware Troubleshooting.

    Your sole task is to analyze the conversation history and the current user query, and output a **single, standalone, and unambiguous Hinglish search query**. This rewritten query will be used by a search engine to retrieve relevant technical documents.

    ---
    **RULES FOR REWRITING:**
    1.  **CRITICAL:** The output MUST be a complete, self-contained Hinglish sentence or phrase that requires **NO** prior conversation history to be understood by a search engine.
    2.  **Pronoun Resolution:** Resolve all ambiguous references and pronouns (e.g., 'woh,' 'us par,' 'iske liye') by replacing them with the actual technical component name (e.g., 'LED_A line,' 'backlight IC,' 'Boost voltage') using the conversation history.
    3.  **No Change Rule:** If the conversation history is empty OR the current user query is already clear and standalone, output the **User Query AS IS**.
    4.  **Preserve Technical Terms:** DO NOT translate technical codes (e.g., K1/K2, VIN, OVP) into English or Hindi. Preserve them exactly as they appear.
    5.  **STRICT OUTPUT FORMAT:** Output ONLY the final, rewritten Hinglish query, enclosed in triple backticks, and nothing else.

    ---
    **CONVERSATION HISTORY (For Context Resolution):**
    {chat_history}

    **CURRENT AMBIGUOUS USER QUERY:**
    {user_query}

    **REWRITTEN STANDALONE HINGLISH SEARCH QUERY:**
    '''
    try:
        rewritten = llm.invoke(rewrite_convo).strip()
        if rewritten:
            return rewritten
        else:
            return user_query  # fallback if rewrite fails
    except Exception as e:
        print("Backlight query rewriting failed:", e)
        return user_query  # fallback
    

def chat_with_user(user_query, chat_history, username):

    IC_CODE = "AL65"   # for now hardcoded (as discussed)

    # ---------------- STEP 1: DIRECT PIN CHECK ---------------- #
    direct_pin = detect_direct_pin(user_query, KB_DATA)

    # Step 1: Rewrite query first
    print("Original Query:", user_query)
    
    # Get the context first (re-ranked)
    rewrittern_Convo = RewriteConvo(user_query,chat_history)
    print("Rewritten_convo Query:", rewrittern_Convo)
    context = get_context(user_query, chat_history)
    # context = get_context(user_query, chat_history)
    clean_query = rewrite_backlight_query(user_query, chat_history, context)
    print("Rewritten Query:", clean_query)
    # The llm is already initialized in the global scope, so no need to download or re-initialize.

    prompt = prompt_template.format(
        context=context,
        question=user_query,
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
