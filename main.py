from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import importlib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional , Dict , Any
import os
import mysql.connector
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from utils.db_fetcher import get_raw_text_from_db
from utils.formatter import format_text_for_knowledge_base
from utils.vector_builder import build_faiss_index
from fastapi.middleware.cors import CORSMiddleware
from pydantic.generics import GenericModel
from typing import List , Optional,Generic, TypeVar
from concurrent.futures import ThreadPoolExecutor
import asyncio




app = FastAPI(root_path="/chatbot_llm")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for concurrency (CPU-safe)
executor = ThreadPoolExecutor(max_workers=2)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  
    "http://65.0.34.207",     
    "http://example.com"      
]

T = TypeVar("T")
class APIResponse(GenericModel, Generic[T]):
    success: bool
    message: str
    data: Optional[T]

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cache loaded categories handlers
loaded_categories = {}

# Define your allowed categories
ALLOWED_CATEGORIES = [
    "audio", "backlight", "bluetooth_wifi", "camera", "charging", "dead", "graphics",
    "hardware_others", "network", "reprogramming_changing", "sim_sdcard",
    "software_unlock", "touch_fingerprint", "ufs_reprogramming_changing", "usb_flashing"
]


class ChatState(BaseModel):
    detected_ic: Optional[str] = None
    detected_problem: Optional[str] = None
    is_confirmed: bool = False

class ChatPayload(BaseModel):
    user_query: str
    conversation_history: List[Dict[str, str]] = []  # format: {"role": "user", "content": "..."}
    state: ChatState
    history_summary: str = ""
    context_external: Optional[str] = None
    context_internal: Optional[str] = None # For IC Pinout data or any other internal context
class ChatRequest(BaseModel):
    user_id: str
    username: str
    payload: ChatPayload
    
def load_category_handler(category: str):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=404, detail="Invalid Category")

    if category in loaded_categories:
        return loaded_categories[category]

    try:
        module = importlib.import_module(f"domains.{category}.handler")
        loaded_categories[category] = module
        return module
    except ModuleNotFoundError:
        raise HTTPException(status_code=500, detail="Handler not found for {category}")

# API FOR CHAT MODUL
@app.post("/chat/{category}",
        response_model=APIResponse,
        summary="Chat with the AI Model for a category, using User Response, Chat History and Username.",
        description="""
          Get response from AI Model for any particular category.

          - `category`: The category for which user wants to chat  (e.g., 'audio','camera','charging').
          - JSON Body should include:
          - `user_query`: user query text
          - `chat_history`: user chat history (format:  [{"user_message": "user message .....","bot_response": " model response ..."},]  ),
          - `username` : name of the user
          - `example`: for api chat/sim_sdcard
              {
                "user_query": "Mere phone mein SIM detect nahi ho raha hai",
                "chat_history": [
                    {
                    "user_message": "SIM insert kiya par signal nahi aa raha",
                    "bot_response": "Kya aapne SIM ko kisi aur phone mein try kiya?"
                    },
                    {
                    "user_message": "Haan, dusre phone mein SIM chal raha hai",
                    "bot_response": "Toh ho sakta hai aapke phone ka SIM slot ya NFC IC mein issue ho."
                    }
                ],
                "username": "Deepak"
             }
          """)
async def chat(category: str, request: ChatRequest):
    handler = load_category_handler(category)
    loop = asyncio.get_event_loop()
    response =  await loop.run_in_executor(
        executor,
        # lambda: handler.chat_with_user(request.user_query, request.chat_history, request.username)
        lambda: handler.chat_with_userr(request)
    )
    return APIResponse(
        success= True,
        message=f'Response from the category {category}',
        data = response
    )
# data = await handler.chat_with_user(request.user_query, request.chat_history,request.username)




# knowledge base updation part

# API FOR KnowledgeBase Updation Part
@app.post("/update_knowledgeBase/data_{category}",
          
          summary="Update the Knowledge base of AI Model for a perticular category.",
          description="""
            Update the knowledge base of AI Model for a perticular domain.
            No body needed. 
            """)
async def update_kb(category: str):
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Invalid Category: {category}")

    logger.info(f"Updating knowledge base for category: {category}")

    # Step 1: Fetch raw text from DB
    raw_text = get_raw_text_from_db(category)

    # Step 2: Format using LLM
    formatted_text = format_text_for_knowledge_base(raw_text)

    # Step 3: Save formatted KB to file
    kb_path = f"domains/{category}/knowledge_base.txt"
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    with open(kb_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    # Step 4: Build FAISS index
    build_faiss_index(category, kb_path)

    return {"status": "success", "message": f"Knowledge base updated for category '{category}'"}