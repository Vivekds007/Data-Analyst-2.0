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
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# --- CONFIGURATION ---
# Silence the tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Nvidia Investor AI",
    page_icon="📈",
    layout="wide",
)

# --- 1. SET UP AI MODELS (OpenAI Only) ---

try:
    # Check for API Key
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # Use GPT-4o-mini (Fast & Smart)
        Settings.llm = OpenAI(
            model="gpt-4o-mini", 
            temperature=0.2,
            max_tokens=4096
        )
        
        # Use optimized OpenAI Embeddings
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )
    else:
        st.error("🚨 OPENAI_API_KEY not found. Please add it to .streamlit/secrets.toml")
        st.stop()

except Exception as e:
    st.error(f"Error setting up models: {e}")

CHROMA_PATH = "./chroma_db"
TEMP_DOC_PATH = "./temp_docs"


# --- 2. CORE LOGIC ---

@st.cache_resource(show_spinner=False)
def get_index():
    """
    Load the index from ChromaDB or create a new one.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    # We use a specific collection name to avoid conflicts
    chroma_collection = db.get_or_create_collection("nvidia_investor_data")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    try:
        index = load_index_from_storage(storage_context)
    except Exception:
        # If index doesn't exist, create an empty one
        index = VectorStoreIndex.from_documents([], storage_context=storage_context)
        index.storage_context.persist(persist_dir=CHROMA_PATH)
    
    return index

def save_uploaded_files(uploaded_files):
    if not os.path.exists(TEMP_DOC_PATH):
        os.makedirs(TEMP_DOC_PATH)
    else:
        shutil.rmtree(TEMP_DOC_PATH)
        os.makedirs(TEMP_DOC_PATH)

    for file in uploaded_files:
        with open(os.path.join(TEMP_DOC_PATH, file.name), "wb") as f:
            f.write(file.getbuffer())
    
    return TEMP_DOC_PATH

# --- 3. UI LAYOUT ---

st.title("📈 Nvidia Investor AI Analyst")
st.caption("Powered by GPT-4o-mini & RAG")

with st.sidebar:
    st.header("Upload Financial Reports")
    uploaded_files = st.file_uploader(
        "Upload PDF Files (10-K, Whitepapers)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Analyze Documents 🚀") and uploaded_files:
        with st.spinner("Ingesting data... This may take a moment."):
            try:
                # 1. Save files
                doc_path = save_uploaded_files(uploaded_files) 
                
                # 2. Load data
                documents = SimpleDirectoryReader(doc_path).load_data()
                
                # 3. Update Index
                index = get_index()
                for doc in documents:
                    index.insert(doc)
                
                # 4. Save to Disk
                index.storage_context.persist(persist_dir=CHROMA_PATH)
                
                # 5. Clear Cache & Temp files
                st.cache_resource.clear() # IMPORTANT: Force reload of new data
                shutil.rmtree(TEMP_DOC_PATH)
                
                st.success("✅ Documents processed! You can now chat.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.divider()
    if st.button("Clear Database 🗑️"):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            st.cache_resource.clear()
            st.success("Database cleared. Please reload page.")

# --- 4. CHAT INTERFACE ---

# Load the index immediately
index = get_index()
query_engine = index.as_query_engine(streaming=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am ready. Upload the Nvidia Annual Report and ask me for a summary."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the stocks, revenue, or risks..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            streaming_response = query_engine.query(prompt)
            for token in streaming_response.response_gen:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error: {e}")
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})