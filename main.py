import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from generate import generate_response
from utils import getDocEmbeds, conversational_chat
from streamlit_chat import message
from prompt_engineering import get_system_prompt
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_icon="💬", page_title="MarketingBot")

# Load environment variables
load_dotenv()

# Validate API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("API Key untuk OpenAI tidak ditemukan. Pastikan telah diset di file .env.")
    st.stop()

# Set CSV file paths
CSV_FILE_PATHS = {
    "product_catalog": "data/product_catalog.csv",
    "komentar_instagram": "data/komentar_instagram.csv",
    "info_produk": "data/info_produk.csv",
    "amazon_review": "data/amazon_review_combined3.csv",
    "basic_info_instagram": "data/basic_info_instagram.csv",
}

# Function to check and load CSV files
def load_and_process_csv_files():
    data = {}
    for key, path in CSV_FILE_PATHS.items():
        if not os.path.exists(path):
            st.error(f"File tidak ditemukan: {path}")
            st.stop()
        data[key] = pd.read_csv(path)
    return data

# Function to initialize chatbot pipeline
def initialize_pipeline():
    try:
        # Load and preprocess CSV files
        data = load_and_process_csv_files()

        # Load document embeddings
        doc_path = CSV_FILE_PATHS["product_catalog"]
        with open(doc_path, "rb") as uploaded_file:
            file = uploaded_file.read()
        vectors = getDocEmbeds(file, doc_path)

        # Load system prompt
        system_prompt = get_system_prompt()

        # Initialize memory for conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_memory=10
        )

        # Initialize conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini"),
            retriever=vectors.as_retriever(),
            memory=memory,
        )

        return chain
    except Exception as e:
        st.error(f"Error saat inisialisasi chatbot: {str(e)}")
        st.stop()

# Main function
def main():
    # Session state initialization
    if "chain" not in st.session_state:
        st.session_state["chain"] = initialize_pipeline()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Chat interface
    st.title("Marketing Insight Bot 💬")
    response_container = st.container()
    input_container = st.container()

    with input_container:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Tanyakan sesuatu:",
                placeholder="Ketik pertanyaan Anda di sini...",
                key="input"
            )
            submit_button = st.form_submit_button(label="Kirim")

            if submit_button and user_input:
                chain = st.session_state["chain"]

                # Generate response
                try:
                    output = conversational_chat(user_input, chain)
                except Exception as e:
                    output = f"Maaf, terjadi kesalahan: {e}"

                # Update chat history
                st.session_state["history"].append({"user": user_input, "bot": output})

    # Display conversation history
    with response_container:
        if st.session_state["history"]:
            for i, chat in enumerate(st.session_state["history"]):
                message(chat["user"], is_user=True, key=f"user_{i}")
                message(chat["bot"], key=f"bot_{i}")

# Run main
if __name__ == "__main__":
    main()
