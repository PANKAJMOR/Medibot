import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

# Define paths
TEMP_DIR = "data1/"
DB_FAISS_PATH = "vectorstore/db_faiss1"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

# Load PDF files from data directory
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

# Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Get embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Set custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load LLM
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
    )
    return llm

def main():
    st.title("PDF Chat Assistant")
    
    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Add file uploader in sidebar
    with st.sidebar:
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        process_button = st.button("Process PDFs")
        
        if uploaded_files and process_button:
            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(TEMP_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            with st.spinner("Processing PDFs..."):
                # Process PDFs
                documents = load_pdf_files(data=TEMP_DIR)
                st.write(f"Number of PDF pages: {len(documents)}")
                
                # Create chunks
                text_chunks = create_chunks(extracted_data=documents)
                st.write(f"Number of text chunks: {len(text_chunks)}")
                
                # Create embeddings and store in FAISS
                embedding_model = get_embedding_model()
                db = FAISS.from_documents(text_chunks, embedding_model)
                db.save_local(DB_FAISS_PATH)
                st.success("PDFs processed successfully!")
    
    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input("Ask a question about your PDFs")
    
    if prompt:
        # Add user message to chat
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        # Define prompt template
        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer.
            
            Dont provide anything out of the given context
            
            Context: {context}
            Question: {question}
            
            Start the answer directly. No small talk please.
        """
        
        HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        # Get vector store and create QA chain
        vectorstore = get_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )
        
        # Get response
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({'query':prompt})
        
        result = response["result"]
        # Display only the result without source documents
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role':'assistant', 'content': result})

if __name__ == "__main__":
    main()