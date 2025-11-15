import streamlit as st
import openai
import os
import shutil
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
import chromadb

# NEW: Added missing embedding model imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# --- PAGE CONFIG ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Chat With Your Docs",
    page_icon="📄",
    layout="wide",
)

# --- 1. SET UP ---
# (API key setup, global paths)
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass # Pass silently if key is not found

CHROMA_PATH = "./chroma_db"
TEMP_DOC_PATH = "./temp_docs"


# --- 2. CORE LOGIC (FUNCTION DEFINITIONS) ---

@st.cache_resource(show_spinner="Loading RAG index...")
def get_index():
    """
    Load the index from the ChromaDB persistent storage, 
    or create it if it doesn't exist.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection("main_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    try:
        index = load_index_from_storage(storage_context)
        print("Loaded existing index.")
    except Exception as e:
        print("Index not found, creating a new one...")
        # Note: The index will be created *using the LLM/Embed Model from Settings*
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        index.storage_context.persist(persist_dir=CHROMA_PATH)
    
    return index

# --- 3. HELPER FUNCTION (FUNCTION DEFINITION) ---

def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory for processing.
    """
    # Ensure the temporary directory exists
    if not os.path.exists(TEMP_DOC_PATH):
        os.makedirs(TEMP_DOC_PATH)
    else:
        # Clear old files if the directory already exists
        shutil.rmtree(TEMP_DOC_PATH)
        os.makedirs(TEMP_DOC_PATH)

    # Save each file
    for file in uploaded_files:
        # FIX: Changed 'mb' to 'wb' (write binary)
        with open(os.path.join(TEMP_DOC_PATH, file.name), "wb") as f:
            f.write(file.getbuffer())
    
    return TEMP_DOC_PATH

# --- 4. STREAMLIT UI (SIDEBAR) ---
# This section defines the UI and SETS the models in Settings
with st.sidebar:
    st.header("1. Upload Documents")
    st.write("Upload PDF documents here. Each upload will add to the knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents... This may take a moment."):
            try:
                # This call will now work, as the function is defined above
                doc_path = save_uploaded_files(uploaded_files) 
                
                documents = SimpleDirectoryReader(doc_path).load_data()
                
                if not documents:
                    st.error("No documents found or failed to load.")
                else:
                    index = get_index() # Get index *after* models are set
                    for doc in documents:
                        index.insert(doc)
                    index.storage_context.persist(persist_dir=CHROMA_PATH)
                    st.success("Documents processed and added to the knowledge base!")
                    # Refresh the index resource to reflect new data
                    st.cache_resource.clear()
                
                shutil.rmtree(TEMP_DOC_PATH)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                if os.path.exists(TEMP_DOC_PATH):
                    shutil.rmtree(TEMP_DOC_PATH)

    # --- LLM & Embedding Selection ---
    st.divider()
    st.header("2. Select Models")
    use_local_llm = st.toggle("Use Local Models (Ollama)", value=True)
    
    if use_local_llm:
        st.info("Using local Ollama (llama3.1:8b). Make sure Ollama is running!")
        Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
        
        st.info("Using local embedding model (BGE Small).")
        # API KEY FIX: Set embed model to local
        # This will download the model (380MB) on first run
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5" 
    
    else:
        # Check if API key is available
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            st.info("Using API (OpenAI gpt-3.5-turbo).")
            Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
            st.info("Using OpenAI embedding model.")
            # API KEY FIX: Set embed model to OpenAI
            Settings.embed_model = OpenAIEmbedding() 
        else:
            st.warning("OpenAI API Key not found. Please switch to local models.")
            # Default to local if key is missing
            st.info("Using local Ollama (llama3.1:8b). Make sure Ollama is running!")
            Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
            
            st.info("Using local embedding model (BGE Small).")
            Settings.embed_model = "local:BAAI/bge-small-en-v1.5"


# --- 5. STREAMLIT UI (MAIN CHAT) ---
# NOW that the models are set by the sidebar, we can
# create the index and query engine.

st.title("Chat With Your Company's Knowledge Base 📄")
st.info("Upload your documents (PDFs) on the left, then ask questions about them.")

# Create the index and query engine *after* the sidebar logic
index = get_index()
query_engine = index.as_query_engine(streaming=True)

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Select your LLM, upload documents, and ask me anything."}
    ]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box
if prompt := st.chat_input("What do you want to know?"):
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user's message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # Placeholder for the streaming response
        full_response = ""
        
        try:
            # Query the RAG engine
            streaming_response = query_engine.query(prompt)
            
            # Stream the response token by token
            for token in streaming_response.response_gen:
                full_response += token
                message_placeholder.markdown(full_response + "▌") # Add a cursor effect
            
            message_placeholder.markdown(full_response) # Final response
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error. Please try again."
            message_placeholder.markdown(full_response)

    # Add assistant's final response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})