#Diese Datei liest PDFs, splittet sie in Chunks und speichert den Chroma-Index lokal in chroma_db/.
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "Data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdfs"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    return splitter.split_documents(docs)

# Embeddings erzeugen:
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Optional: alten Index löschen und komplett neu aufbauen
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


def main():
    print("Lade PDFs...")
    docs = load_documents()
    print(f"{len(docs)} Dokumentseiten geladen.")

    print("Splitte Dokumente in Chunks...")
    chunks = split_documents(docs)
    print(f"{len(chunks)} Chunks erzeugt.")

    print("Erzeuge Embeddings mit HuggingFace und speichere Chroma-Index...")
    build_vectorstore(chunks)

    print("Fertig. Der Index liegt jetzt in 'chroma_db/'.")


if __name__ == "__main__":
    main()
