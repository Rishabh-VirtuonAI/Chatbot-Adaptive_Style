from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableSequence

# Load once per process
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local("domains/camera/faiss_index", 
            embeddings=embedding,
             allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 6})

llm = Ollama(model="llama3.1:8b-instruct-q4_K_M",
              temperature=1,
                top_p=0.7,
                num_ctx=4096,
                # num_ctx = 1024,
                # num_ctx = 2048,
                top_k=20,
                repeat_penalty=1.1)

# Load audio-specific prompt
with open("domains/camera/prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

prompt = PromptTemplate(
    input_variables=["context", "question", "conversation_history"],
    template=prompt_template
)

# Compose the custom chain
chain = prompt | llm

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

def chat_with_user(user_query, chat_history, username):
    docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in docs])
    formatted_history = "\n".join(f"User: {m['user_message']}\nBot: {m['bot_response']}" for m in chat_history)
    print(f"the user name i am getting is {username}")
    print(f"the context i am getting is ->{context}")

    response = chain.invoke({
        "context": context,
        "question": user_query,
        "conversation_history": formatted_history,
        'username':username
    })

    return {"response": response}
