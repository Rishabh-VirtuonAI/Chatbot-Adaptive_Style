from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableSequence
import re
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


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
    cross_encoder = CrossEncoder('BAAI/bge-reranker-large', device=device)
else:
    cross_encoder = None

llm = Ollama(
    model="gemma3:12b-it-q4_K_M", 
              temperature=0.3,
              top_p=0.5,
              num_ctx=4096,

                top_k=64,
                repeat_penalty=1.1)

# Load audio-specific prompt
with open("domains/backlight/prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

def get_context_with_rerank(question,conversation_history, top_k=10, rerank_top=1):
    """
    1) FAISS semantic search (top_k candidates)
    2) Re-rank candidates with CrossEncoder (if available)
    3) Return top `rerank_top` chunks concatenated as context string
    """
    import re

    if vectorstore is None:
        return ""
    

    if conversation_history:
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



def extract_ic_diag_vocab(kb):
    """
    Build IC-specific diagnostic vocabulary directly from KB content.
    """
    vocab = set()

    for pin in kb.get("pin_details", []):
        for field in ["pin_name", "function", "uses", "work"]:
            text = pin.get(field, "")
            for w in text.lower().split():
                if len(w) > 2:
                    vocab.add(w)

        for fault in pin.get("faults", []):
            for field in ["possible_causes", "diagnostic_procedure", "diagnostic_tests"]:
                for sentence in fault.get(field, []):
                    for w in sentence.lower().split():
                        if len(w) > 2:
                            vocab.add(w)

    return vocab


# cache per IC (do this once)
# IC_DIAG_VOCAB = {
#     "AL65": extract_ic_diag_vocab(KB_DATA)
# }



def clean_llm_output(text):
    """
    Post-process LLM output: remove labels like 'ANSWER:' or 'Final response:' and tidy lines.
    """
    import re
    if not text:
        return text
    text = re.sub(r'(?i)\bfinal response:\s*', '', text)
    text = re.sub(r'(?i)\banswer:\s*', '', text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and ('.' in lines[0] or ';' in lines[0]):
        pieces = re.split(r'\.\s+|;\s+|\n', lines[0])
        pieces = [p.strip() for p in pieces if p.strip()]
        if len(pieces) > 1:
            lines = pieces
    return "\n".join(lines)


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


FAULT_PIN_MAP = {
    "AL65": {
        "NO_BACKLIGHT": [6, 3, 4, 5, 1],
        "DIM_BACKLIGHT": [4, 5, 1],
        "BRIGHTNESS_FIXED": [5, 1]
    }
}

FAULT_ONTOLOGY = {
    "NO_BACKLIGHT": {
        "meaning": "Screen has no light at all. Display is dark or black.",
        "signals": [
            "no backlight",
            "screen dark",
            "andhera",
            "no light"
        ]
    },
    "DIM_BACKLIGHT": {
        "meaning": "Backlight is present but weak or low brightness.",
        "signals": [
            "dim",
            "light kam",
            "brightness kam"
        ]
    },
    "BRIGHTNESS_FIXED": {
        "meaning": "Brightness is stuck or always at full level.",
        "signals": [
            "full light",
            "100%",
            "brightness fix"
        ]
    }
}



def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def normalize_user_query_strict(user_query: str, threshold: float = 0.75) -> str | None:
    query = clean_text(user_query)

    for fault, patterns in FAULT_NORMALIZATION_MAP.items():
        for phrase in patterns:
            phrase_clean = clean_text(phrase)

            # 1️⃣ Exact substring
            if phrase_clean in query:
                print("Exact match found:", phrase_clean, "in", query)
                return fault

            # 2️⃣ Fuzzy similarity
            if similarity(phrase_clean, query) >= threshold:
                print("Fuzzy match found:", phrase_clean, "vs", query)
                return fault

            # 3️⃣ Token overlap (VERY important for Hinglish)
            # phrase_tokens = set(phrase_clean.split())
            # query_tokens = set(query.split())

            # overlap = phrase_tokens & query_tokens
            # if len(overlap) >= max(1, len(phrase_tokens) - 1):
            #     print("Token overlap match found:", phrase_clean, "vs", query)
            #     return fault

    return None

FAULT_SEMANTIC_ANCHORS = {
    "NO_BACKLIGHT": "screen dark black display no backlight no light",
    "DIM_BACKLIGHT": "display dim low brightness light kam brightness kam",
    "BRIGHTNESS_FIXED": "brightness fixed full 100 percent cannot change brightness"
}

def semantic_fault_match(user_query: str, threshold: float = 0.80) -> str | None:
    """
    Uses embeddings to semantically match user query to a fault.
    Returns fault if confidence >= threshold, else None.
    """

    if not user_query.strip():
        return None

    # Embed user query
    query_embedding = embedding.embed_query(user_query)

    best_fault = None
    best_score = 0.0

    for fault, anchor_text in FAULT_SEMANTIC_ANCHORS.items():
        anchor_embedding = embedding.embed_query(anchor_text)

        score = cosine_similarity(
            [query_embedding],
            [anchor_embedding]
        )[0][0]

        if score > best_score:
            best_score = score
            best_fault = fault

    # Confidence gate (VERY IMPORTANT)
    if best_score >= threshold:
        return best_fault

    return None

FAULT_CLASSIFIER_PROMPT = """
You are a fault classification system.

Classify the user's issue into EXACTLY ONE of the following faults:

- NO_BACKLIGHT
- DIM_BACKLIGHT
- BRIGHTNESS_FIXED

User query:
"{user_query}"

Rules:
- Return ONLY the fault name.
- If the query does not clearly match any fault, return UNKNOWN.
- Do not explain anything.
"""
def llm_fault_classify(user_query: str, fault_ontology: dict) -> str | None:
    fault_descriptions = "\n".join([
        f"""
    Fault: {fault}
    Meaning: {data['meaning']}
    Common user phrases: {', '.join(data['signals'])}
    """ for fault, data in fault_ontology.items()
        ])

    prompt = f"""
    You are a domain expert technician.

    Below are possible fault categories and their meanings.

    {fault_descriptions}

    User query:
    "{user_query}"

    Task:
    - Understand the user's intent and symptom.
    - Match it to EXACTLY ONE fault based on meaning.
    - If no fault clearly matches, return UNKNOWN.

    Rules:
    - Return ONLY the fault name.
    - Do NOT explain.
    """

    try:
        response = llm.invoke(prompt)
        fault = response.strip().upper()

        if fault in fault_ontology:
            return fault

    except Exception as e:
        print(f"LLM classification failed: {e}")

    return None





def normalize_user_query(user_query: str) -> str | None:
    """
    Returns canonical fault name if matched, else None
    """
    fault = normalize_user_query_strict(user_query)
    if fault:
        print("Normalized via strict matching: ++++++++++++++ ----------- **********", fault)
        return fault

    # Level 3: Semantic embeddings
    fault = semantic_fault_match(user_query)  # your FAISS-based logic
    if fault:
        print("Normalized via semantic matching: ++++++++++++++ ----------- **********", fault)
        return fault

    # Level 4: LLM fallback (GUARANTEED)
    fault = llm_fault_classify(user_query,FAULT_ONTOLOGY)
    if fault in FAULT_NORMALIZATION_MAP:
        print("Normalized via LLM classification: ++++++++++++++ ----------- **********", fault)
        return fault

    return None

def get_pins_for_fault(board: dict, fault: str) -> list[int]:
    """
    Returns list of pins related to the fault for a given board
    """
    return FAULT_PIN_MAP.get(board, {}).get(fault, [])

def retrieve_kb_for_pins(pins: list[int], kb_chunks: list[dict]) -> list[str]:
    """
    Returns relevant KB texts only for required pins
    """
    context = []

    for chunk in kb_chunks:
        if chunk.get("pin") in pins:
            context.append(chunk["text"])

    return context




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
    


def build_pin_aware_query(user_query: str, fault: str, pins: list[int]) -> str:
    """
    Injects fault + pin constraints into the semantic search query
    """
    pin_text = ", ".join([f"pin {p}" for p in pins])

    enhanced_query = f"""
        User issue: {user_query}
        Detected fault: {fault}
        Relevant circuit pins: {pin_text}

        Retrieve technical information specifically related to these pins.
        """.strip()

    return enhanced_query





def get_pin_scoped_context(
    user_query: str,
    fault: str,
    pins: list[int],
    conversation_history=None,
    top_k=10,
    rerank_top=1
):
    

    if vectorstore is None:
        return ""

    # Step 1: Build pin-aware query
    pin_aware_query = build_pin_aware_query(user_query, fault, pins)

    # Step 2: Add minimal conversation history (as you already do)
    if conversation_history:
        history_text = "\n".join(
            [f"User: {m['user_message']}\nBot: {m['bot_response']}"
             for m in conversation_history[-1:]]
        )
        enhanced_query = f"{pin_aware_query}\nConversation History:\n{history_text}"
    else:
        enhanced_query = pin_aware_query

    # Step 3: FAISS semantic retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(enhanced_query)
    candidates = [d.page_content for d in docs]

    if not candidates:
        return ""

    # Step 4: Cross-encoder re-ranking (your logic)
    if CROSS_ENCODER_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cross = cross_encoder  # already loaded in your app

            pairs = [[enhanced_query, c] for c in candidates]
            scores = cross.predict(pairs)

            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_chunks = [c for c, _ in ranked[:rerank_top]]
            return "\n\n".join(top_chunks)

        except Exception as e:
            print(f"Cross-encoder failed: {e}")

    # fallback (no rerank)
    return "\n\n".join(candidates[:rerank_top])

def update_diagnostic_history(chat_history, user_query, bot_response):
    """
    Summarizes ONLY the facts and findings from the conversation.
    This acts as the 'Service Record' for the phone.
    """
    
    state_extraction_prompt = f"""
    ### ROLE
    You are a Technical Data Extractor. Your only job is to update the 'Diagnostic History String' based on the latest conversation.
    ### INPUT DATA
    - PREVIOUS HISTORY: {chat_history}
    - LATEST INTERACTION:
        User: {user_query}
        Deepak (Bot): {bot_response}

    ### INSTRUCTIONS
    1. **RECORD FACTS:** Extract every component name, Pin number, Voltage, GR value (Diode value), and physical action (Resoldered/Replaced).
    2. **CURRENT FAULT:** Explicitly state the problem currently being solved (e.g., No Backlight).
    3. **BE CONCISE:** Use a list format. (e.g., "Coil: 3.7V OK", "Pin 1: 0.350 GR").
    4. **RESOLVE CONFLICTS:** If the user provides a new value for a component already in the history, update it to the latest value.
    4. **NO DUPLICATION:** If a pin was already tested and the value hasn't changed, do not list it twice. Update it only if the value changed.
    5. **CLEANLINESS:** Do not include conversation, advice, or "Next Steps". Only findings.

    ### OUTPUT FORMAT (STRICT)
    Device: [Model] | IC: [Name] | Fault: [Current Problem]
    Findings: [Step-by-step list of verified facts]
    """

    # Call Gemma 3:12B-it
    response = llm.invoke(state_extraction_prompt).strip()
    
    return response.strip()



def chat_with_user(user_query, chat_history, username):

    IC_CODE = "AL65"   

    # direct_pin = detect_direct_pin(user_query, KB_DATA)

    # Step 1: Normalize fault
    fault = normalize_user_query(user_query)
    if not fault:
        context = get_context(user_query, chat_history)
        # return "Unable to identify fault from the query."

    # Step 2: Get relevant pins
    print("Normalized Fault i am getting issssss :", fault)
    if fault:
        pins = get_pins_for_fault("AL65", fault)
        if not pins:
            context = get_context(user_query, chat_history)
            # return f"No pin mapping found for fault: {fault}"
    
        print("Relevant Pins aressssssssssssssssssss       =====================        :", pins)
        
        # Step 3: Pin-aware embedding retrieval
        context = get_pin_scoped_context(
            user_query=user_query,
            fault=fault,
            pins=pins,
            conversation_history="\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in chat_history),
            top_k=10,
            rerank_top=len(pins)
        )

        print("Pin-aware Context:", context)
    

    # Step 1: Rewrite query first
    print("Original Query:", user_query)
    
    rewrittern_Convo = RewriteConvo(user_query,chat_history)
    print("Rewritten_convo Query:", rewrittern_Convo)
    # context = get_context(user_query, chat_history)
    
    # clean_query = rewrite_backlight_query(user_query, chat_history, context)
    # print("Rewritten Query:", clean_query)
    
    prompt = prompt_template.format(
        context=context,
        question=user_query,
        conversation_history="\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in chat_history),
        username=username
    )

    response = llm.invoke(
        prompt
    )
    
    diagnostic_state = update_diagnostic_history(chat_history, user_query, response)

    print("Raw LLM response:", diagnostic_state)

    cleaned = clean_llm_output(response)

    print("question: ", user_query)
    print("context: ", context)
    print("Final raw response:", response)
    print("Final cleaned response:", cleaned)

    return {"response": cleaned if cleaned else "I couldn't generate a response. Please try again."}

def load_external_manual(file_path="backlight_external_kb.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "No external diagnostic data available."
    
def run_chain_1_intake(client_data):
    # Constructing the Expert Intake Prompt
    # We pass the history so it remembers what was discussed in previous turns
    system_prompt = f"""
    You are a Senior Mobile Technician (Ustaad). Your goal is to identify the IC and the BACKLIGHT PROBLEM.
    
    CURRENT DATA:
    - IC: {client_data["payload"]['state']['detected_ic'] or 'Unknown'}
    - Problem: {client_data["payload"]['state']['detected_problem'] or 'Unknown'}
    - User Query: {client_data["payload"]["user_query"]}
    - DIAGNOSTIC HISTORY: {client_data["payload"]["history_summary"]}
    
    INSTRUCTIONS:
    1. If information is missing, ask for it in Hinglish.
    2. If the user provides a mobile model (e.g., Redmi 6A), suggest the IC (AL65).
    3. Check the DIAGNOSTIC HISTORY. If the IC or Problem is already listed there, do not ask the user for them again. Instead, use that data to ask for a final confirmation
    3. If both IC and Problem are identified, you MUST ask for confirmation: "Kya hum [IC] aur [Problem] par kaam shuru karein?"
    4. ONLY set USER_CONFIRMED: Yes when the user explicitly agrees (e.g., "Haan", "Ok", "Shuru karo").

    OUTPUT FORMAT:
    Response: [Hinglish Text]
    ---
    DETECTED_IC: [Name]
    DETECTED_PROBLEM: [Problem]
    USER_CONFIRMED: [Yes/No]
    """

    # Call your local Gemma 3 model here
    # llm_response = gemma_model.generate(system_prompt, client_data['user_query'])
    llm_response= llm.invoke(
        system_prompt
    )

    updated_history_string = update_diagnostic_history(client_data["payload"]["conversation_history"], client_data["payload"]["user_query"], llm_response)

    # Extract metadata using regex
    import re
    meta_ic = re.search(r"DETECTED_IC:\s*(.*)", llm_response)
    meta_prob = re.search(r"DETECTED_PROBLEM:\s*(.*)", llm_response)
    meta_conf = re.search(r"USER_CONFIRMED:\s*(.*)", llm_response)

    return {
        "user_id": client_data['user_id'],
        "bot_response": llm_response.split("---")[0].strip(),
        "ic_name": meta_ic.group(1).strip() if meta_ic else client_data["payload"]['state']['detected_ic'],
        "problem_type": meta_prob.group(1).strip() if meta_prob else client_data["payload"]['state']['detected_problem'],
        "is_confirmed": "Yes" in meta_conf.group(1) if meta_conf else False,
        "context": "", # Remains empty until the next step in chat_with_user
        "context_external": "", # Load external manual content for Chain 2
        "history_summary": updated_history_string # Pass through updated history summary
    }



def run_chain_2_expert_diagnostic(client_data, gemma_model, vectorstore_functions):
    """
    Final Chain 2 Handler with Human-in-the-loop Confirmation.
    """
    # 1. Prepare Inputs
    # Splitting context if your retrieval provides it combined, 
    # or passing it as is if formatted in your RAG logic.
    context = client_data.get('context', "No data found")
    
    prompt = prompt_template.format(
        context_external=client_data["payload"]["context_external"],
        context_internal=client_data["payload"]["context"],
        user_query=client_data["payload"]["user_query"],
        history_summary=client_data["payload"]["history_summary"],
        username=client_data["username"]
    )

    # 2. Call local Gemma 3:12B
    llm_output= llm.invoke(
        prompt
    )
    

    
    
    # 3. Parse Response and Metadata
    parts = llm_output.split("---")
    bot_response = parts[0].strip()
    metadata = parts[1] if len(parts) > 1 else ""

    # Extract Metadata using Regex
    history_match = re.search(r"UPDATED_HISTORY:\s*(.*)", metadata)
    new_history_facts = history_match.group(1).strip() if history_match else ""
    
    switch_match = re.search(r"\[\[SWITCH_CONFIRMED:\s*(.*)\]\]", metadata)
    confirmed_new_problem = switch_match.group(1).strip() if switch_match else "None"

    # 4. Handle Context Switching (Confirmation-based)
    if confirmed_new_problem != "None":
        print(f"Switching Context to: {confirmed_new_problem}")
        
        # Call your existing normalization and RAG logic
        normalized_fault = normalize_user_query(client_data["payload"]["user_query"])
        
        # normalized_fault = vectorstore_functions['normalize'](confirmed_new_problem)
        if normalized_fault:
            client_data["payload"]['state']['detected_problem'] = normalized_fault
            pins = get_pins_for_fault(client_data["payload"]['state']['detected_ic'], normalized_fault)
            
            # Refresh context for the next turn
            client_data["payload"]['context'] = get_pin_scoped_context(
            user_query=client_data["payload"]["user_query"],
            fault=normalized_fault,
            pins=pins,
            conversation_history="\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in client_data["payload"]["conversation_history"]),
            top_k=10,
            rerank_top=len(pins)
        )

    # 5. Update History Summary
    # (In a real app, you might use your Historian Chain here instead of simple string addition)
    updated_history_string = update_diagnostic_history(client_data["payload"]["conversation_history"], client_data["payload"]["user_query"], bot_response)
    

    return {
        "user_id": client_data['user_id'],
        "bot_response": bot_response,
        "history_summary": updated_history_string,
        "detected_ic": client_data["payload"]['state']['detected_ic'],
        "problem_type": client_data["payload"]['state']['detected_problem'],
        "is_confirmed": True,
        "context": client_data["payload"]['context'],
        "context_external": client_data["payload"]["context_external"]  
    }


def chat_with_userr(client_data):
    """
    Main entry point. 
    client_data contains: user_query, chat_history, ic_name, problem_type, is_confirmed, context

    {
        "user_id": "unique_id_123", -c 
        "username": "Deepak", -c 
        "payload": {
            "user_query": "Bhai line dead hai value nahi aa rahi", -c
            "conversation_history": [           -c
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "state": {
                "detected_ic": "AL65",
                "detected_problem": "No_Backlight",
                "repair_phase": "ic_deep_dive",
                "is_confirmed": true
            },
            "history_summary": "VPH checked (3.7V), Coil inspected (No burn), Pin 4 (SW) reading 0.350",
            "context": "[[ONLY FILLED BY BACKEND AFTER IC IS CONFIRMED]]"
            "context_external": "Context retrieved from vectorstore based on the user query and pins related to the detected fault.",
        }
    }
    """
    user_query = client_data['payload']['user_query']
    
    # CASE A: Diagnostic Phase (Expert is already working)
    if client_data['payload']['state']['is_confirmed']:
        return run_chain_2_expert_diagnostic(client_data)

    # CASE B: Intake Phase (Finding IC and Problem)
    else:
        # Run Chain 1 (The LLM logic to identify IC/Problem)
        intake_result = run_chain_1_intake(client_data)
        
        # Check if Chain 1 just achieved confirmation
        if intake_result['is_confirmed']:
            # 1. Normalize the fault using your existing logic
            normalized_fault = normalize_user_query(intake_result['problem_type'])
            
            # 2. Fetch the Scoped Context using your RAG pipeline
            if normalized_fault and intake_result['ic_name']:
                related_pins = get_pins_for_fault(intake_result['ic_name'], normalized_fault)
                
                # Fetching context based on your defined process
                new_context = get_pin_scoped_context(
                    user_query=user_query,
                    fault=normalized_fault,
                    pins=related_pins,
                    conversation_history=client_data['payload']["conversation_history"],
                )
                
                # load external manual content and append to context
                context_external = load_external_manual()

                intake_result['context_external'] = context_external
                intake_result['context'] = new_context
                intake_result['problem_type'] = normalized_fault # Update to canonical name
            
        return intake_result