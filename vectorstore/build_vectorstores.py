import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuration ---
# Input data files
HR_DOC = "data/NexaCorp HR Manual.docx"
IT_DOC = "data/NexaCorp IT Support Manual.docx"
PAYROLL_DOC = "data/NexaCorp Payroll Support Manual.docx"
TICKETS_XLSX = "data/nexacorp_tickets.xlsx"
TICKETS_CSV = "data/nexacorp_tickets.csv" # Intermediate CSV file

# Output directories for ChromaDB
HR_DIR = "data/chroma/hr"
IT_DIR = "data/chroma/it"
PAYROLL_DIR = "data/chroma/payroll"
TICKETS_DIR = "data/chroma/tickets"

# Embedding model configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def process_word_doc(doc_path: str, persist_dir: str, embeddings):
    """Loads a .docx file, splits it into chunks, and saves it to a vector store."""
    print(f"Processing Word document: {doc_path}...")
    loader = UnstructuredWordDocumentLoader(doc_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print(f"Successfully saved vector store to {persist_dir}")
    return vectorstore

def process_tickets_csv(csv_path: str, persist_dir: str, embeddings):
    """
    Loads ticket data from a CSV, structures it for semantic search,
    and saves it to a vector store.
    """
    print(f"Processing tickets CSV file: {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8')

    # Create a new column with structured text for better semantic meaning
    df['structured_text'] = df.apply(
        lambda row: f"Ticket ID: {row['Complaint ID']}. User Complaint: {row['Complaint']}. Final Resolution: {row['Resolution']}",
        axis=1
    )

    # Create LangChain Document objects from the structured text
    documents = [
        Document(
            page_content=row['structured_text'],
            metadata={
                "ticket_id": row['Complaint ID'],
                "employee": row['Employee Name'],
                "domain": row['Domain'],
                "status": row['Status']
            }
        ) for index, row in df.iterrows()
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print(f"Successfully saved structured ticket vector store to {persist_dir}")
    return vectorstore

def convert_excel_to_csv(xlsx_path: str, csv_path: str):
    """Converts an Excel file to a CSV file."""
    print(f"Converting {xlsx_path} to {csv_path}...")
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print("Conversion successful.")

def main():
    """Main function to build all vector stores."""
    # Initialize the embeddings model once
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create directories if they don't exist
    os.makedirs(HR_DIR, exist_ok=True)
    os.makedirs(IT_DIR, exist_ok=True)
    os.makedirs(PAYROLL_DIR, exist_ok=True)
    os.makedirs(TICKETS_DIR, exist_ok=True)

    # Process all document files
    process_word_doc(HR_DOC, HR_DIR, embeddings)
    process_word_doc(IT_DOC, IT_DIR, embeddings)
    process_word_doc(PAYROLL_DOC, PAYROLL_DIR, embeddings)
    
    # Process the tickets file
    convert_excel_to_csv(TICKETS_XLSX, TICKETS_CSV)
    process_tickets_csv(TICKETS_CSV, TICKETS_DIR, embeddings)

    print("\nAll vector stores have been built successfully.")

if __name__ == "__main__":
    main()
