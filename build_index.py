#Diese Datei liest deine PDFs, splittet sie in Chunks und speichert den Chroma-Index lokal in chroma_db/.
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "Data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdfs"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="gemma2:2b")

    # Optional: alten Index löschen und komplett neu aufbauen
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore

def main():
    print("Lade PDFs...")
    docs = load_documents()
    print(f"{len(docs)} Dokumentseiten geladen.")

    print("Splitte Dokumente in Chunks...")
    chunks = split_documents(docs)
    print(f"{len(chunks)} Chunks erzeugt.")

    print("Erzeuge Embeddings und speichere Chroma-Index...")
    build_vectorstore(chunks)

    print("Fertig. Der Index liegt jetzt in 'chroma_db/'.")

if __name__ == "__main__":
    main()