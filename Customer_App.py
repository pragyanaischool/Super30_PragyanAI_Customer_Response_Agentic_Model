import streamlit as st
from langchain_groq import ChatGroq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.langchain import LangchainLLM
import os
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables for local development
load_dotenv()

# Use Streamlit's secrets management for deployment.
# This will try to get the secret from Streamlit Cloud first,
# and fall back to a local .env file if not found.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_PATH = "data"

# --- Streamlit Page Setup ---

st.set_page_config(
    page_title="PragyanAI Chat Assistant",
    page_icon="🤖",
    layout="centered"
)
st.image("PragyanAI_Transperent.png")
st.title("🤖 PragyanAI Chat Assistant")
st.caption("Welcome! I'm an AI assistant powered by Groq and LlamaIndex.")

# --- Caching the RAG Pipeline ---
# This is crucial for performance. We cache the index and query engine
# so they don't get re-created on every user interaction.
@st.cache_resource(show_spinner="Loading knowledge base... Please wait.")
def load_query_engine():
    """Loads documents, builds the index, and returns a query engine."""
    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    
    # Initialize the Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192", 
        temperature=0.7
    )
    
    # Configure LlamaIndex settings to use our LLM
    Settings.llm = LangchainLLM(llm=llm)
    
    # Create the vector store index
    index = VectorStoreIndex.from_documents(documents)
    
    # Return the query engine with streaming enabled for a better UX
    return index.as_query_engine(streaming=True)

# --- Main Application Logic ---
# Check if the API key is available before proceeding
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set! Please add it to your Streamlit secrets or a local .env file.")
else:
    # Check for a unique session ID from the URL query parameters.
    # This ensures the chat is accessed via the unique link from the email.
    if 'session_id' not in st.query_params:
        st.warning("Please access this chat through the unique link provided in your email.")
    else:
        # Load the query engine (will be cached after the first run)
        query_engine = load_query_engine()

        # Initialize chat history in Streamlit's session state if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today based on our knowledge base?"}]

        # Display past messages from the chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input from the chat interface
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to session state and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant's response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Query the engine with the user's prompt
                    streaming_response = query_engine.query(prompt)
                    
                    # Stream the response to the UI for a real-time effect
                    response_container = st.empty()
                    full_response = ""
                    for text in streaming_response.response_gen:
                        full_response += text
                        response_container.markdown(full_response)
                    
            # Add the complete assistant response to the session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
