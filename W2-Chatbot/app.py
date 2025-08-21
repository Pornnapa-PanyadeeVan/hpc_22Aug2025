import os
import sys
import torch
import textwrap
import re
import hashlib
from pathlib import Path
from typing import List

# --- LangChain and Document Processing ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain.docstore.document import Document

# --- Hugging Face Transformers ---
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# --- 1. CONSTANTS AND CONFIGURATION ---
# ==============================================================================
DATA_PATH = "data/"
PROCESSED_PATH = "processed_texts/" # Cache for processed text
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
LLM_MODEL = "Qwen/Qwen2-1.5B-Instruct"
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.doc', '.docx']

# --- Tokenization Parameters ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# k for retriever
RETRIEVER_K = 10  # Number of documents to retrieve for each query


# ==============================================================================
# --- 2. HELPER FUNCTIONS (File Hashing, I/O, etc.) ---
# ==============================================================================
def ensure_directories_exist():
    """Ensure required directories exist."""
    for directory in [DATA_PATH, PROCESSED_PATH, os.path.dirname(DB_FAISS_PATH)]:
        os.makedirs(directory, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Generate an MD5 hash for a file's content."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_processed_files_record() -> dict:
    """Load the record of processed files and their hashes."""
    record_file = os.path.join(PROCESSED_PATH, "processed_files.txt")
    processed = {}
    if os.path.exists(record_file):
        with open(record_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|', 1)
                    processed[parts[0]] = parts[1]
    return processed

def save_processed_files_record(processed: dict):
    """Save the record of processed files and their hashes."""
    record_file = os.path.join(PROCESSED_PATH, "processed_files.txt")
    with open(record_file, 'w', encoding='utf-8') as f:
        for file_path, file_hash in processed.items():
            f.write(f"{file_path}|{file_hash}\n")

def get_safe_input(prompt: str) -> str:
    """Reads input safely from the terminal, handling potential encoding issues."""
    print(prompt, end="", flush=True)
    buffer = sys.stdin.buffer
    line_bytes = buffer.readline()
    return line_bytes.decode('utf-8', errors='replace').strip()

# ==============================================================================
# --- 3. DOCUMENT PROCESSING & VECTOR STORE CREATION ---
# ==============================================================================
def process_file(file_path: str) -> List[Document]:
    """Dynamically process a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.doc': UnstructuredFileLoader,
        '.docx': UnstructuredFileLoader,
    }
    
    if ext not in loaders:
        return []
        
    try:
        # TextLoader needs an encoding argument
        if ext == '.txt':
            loader = loaders[ext](file_path, encoding='utf-8')
        else:
            loader = loaders[ext](file_path)
            
        documents = loader.load()
        print(f"  ✓ Processed: {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return []

def create_vector_db():
    """
    Loads documents, processes new/modified ones, splits them, creates embeddings,
    and builds/saves the FAISS vector store.
    """
    print("\n" + "="*50)
    print("BUILDING/UPDATING VECTOR DATABASE")
    print("="*50)

    all_documents = []
    processed_record = load_processed_files_record()
    new_processed_record = processed_record.copy()
    
    files = [p for ext in SUPPORTED_EXTENSIONS for p in Path(DATA_PATH).glob(f"**/*{ext}")]
    
    if not files:
        print(f"Warning: No supported files found in {DATA_PATH}. The chatbot will have no knowledge.")
        return

    print(f"Found {len(files)} files to check...")
    
    for file_path in files:
        file_str = str(file_path)
        current_hash = get_file_hash(file_str)
        
        if file_str in processed_record and processed_record[file_str] == current_hash:
            # print(f"  → Unchanged: {os.path.basename(file_str)}")
            continue # Skip unchanged files
            
        print(f"  → Processing new/modified: {os.path.basename(file_str)}")
        docs = process_file(file_str)
        if docs:
            all_documents.extend(docs)
            new_processed_record[file_str] = current_hash
            
    if not all_documents:
        print("No new or modified documents to process. Vector store is up to date.")
        print("="*50 + "\n")
        return

    print(f"\nSplitting {len(all_documents)} new/modified document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= CHUNK_SIZE, chunk_overlap= CHUNK_OVERLAP)
    texts = text_splitter.split_documents(all_documents)
    print(f"✓ Split into {len(texts)} chunks.")

    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})# edit device as needed(cpu/cuda)

    if os.path.exists(DB_FAISS_PATH):
        print("Updating existing FAISS vector store...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(texts)
    else:
        print("Building new FAISS vector store...")
        db = FAISS.from_documents(texts, embeddings)
    
    db.save_local(DB_FAISS_PATH)
    save_processed_files_record(new_processed_record)
    print(f"✓ Vector store saved to {DB_FAISS_PATH}")
    print("="*50 + "\n")

# ==============================================================================
# --- 4. LLM AND RETRIEVER LOADING ---
# ==============================================================================
def load_llm_and_tokenizer():
    """Loads the Qwen model and tokenizer."""
    print(f"⏳ Loading LLM and Tokenizer: {LLM_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    print("✅ LLM and Tokenizer are ready.")
    return model, tokenizer

def setup_retriever():
    """Loads the vector store and embedding model to create a retriever."""
    print("⏳ Loading Embedding Model and Vector Store...")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"FATAL: Vector store not found at {DB_FAISS_PATH}")
        print("Please ensure there are documents in the 'data' folder and run the script again to build it.")
        sys.exit(1)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector Store and Embeddings are ready.")
    return db.as_retriever(search_kwargs={'k': RETRIEVER_K}) 

def detect_language(text: str) -> str:
    """Detect dominant language (Thai or English) based on character ratio."""
    thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)  # Unicode block ภาษาไทย
    eng_chars = re.findall(r'[A-Za-z]', text)

    if len(eng_chars) == 0 and len(thai_chars) > 0:
        return "thai"
    if len(eng_chars) > 0 and (len(eng_chars) / max(1, (len(eng_chars)+len(thai_chars)))) >= 0.9:
        return "english"
    return "thai"   # fallback → ถือว่าเป็นไทย


# ==============================================================================
# --- 5. MAIN CHAT APPLICATION ---
# ==============================================================================
def main():
    # Setup Encoding for Terminal
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    
    # --- Step 1: Ensure directories exist and update knowledge base ---
    ensure_directories_exist()
    create_vector_db() # This will now build or update the DB as needed

    # --- Step 2: Load all necessary components for chat ---
    retriever = setup_retriever()
    model, tokenizer = load_llm_and_tokenizer()


    print("\n" + "="*50)
    print(f"💬 CHATBOT READY! (Model: {LLM_MODEL})")
    print("="*50)
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    print("="*50 + "\n")

    # --- Step 3: Main conversation loop ---
    while True:
        try:
            user_question = get_safe_input("🧑‍💻 Your Question: ")
            if user_question.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break
            if not user_question.strip():
                continue

            # Detect language
            lang = detect_language(user_question)

            if lang == "thai":
                system_prompt = (
                    "คุณคือผู้ช่วยที่เป็นมิตร ตอบคำถามด้วยภาษาไทยหรือภาษาอังกฤษเท่านั้นเท่านั้น ห้ามใช้ภาษาอื่น"
                    "ใช้ข้อมูลจาก Context ด้านล่าง ถ้าไม่พบข้อมูล ให้ตอบว่า "
                    "'ไม่พบข้อมูลในเอกสารที่ให้ไว้'"
                )
            else:
                system_prompt = (
                    "You are a helpful assistant. "
                    "Answer ONLY in English. "
                    "Use the provided context as your main source. "
                    "If the context does not contain the answer, say: "
                    "'No relevant information found in the provided documents.'"
                )

            print("\n🤖 Assistant is thinking...")

            # --- RAG Core Logic ---
            docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in docs])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # --- Display results ---
            print("\n✅ Answer:")
            print(textwrap.fill(response, width=100))
            
            print("\n📚 Sources:")
            for i, doc in enumerate(docs):
                print(f"  [{i+1}] Source: {doc.metadata.get('source', 'N/A')}")

            print("-" * 50)
        except Exception as e:
            print(f"\n❗️ An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
