from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# load the document
#docs = PyPDFLoader("Data/Lineare_Modelle.pdf").load()

# Lädt alle PDFs aus dem angegebenen Ordner
loader = DirectoryLoader('./Data', glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1500) #hier anpassen möglich
chunks = splitter.split_documents(docs)

# embed the chunks and store them in chroma db
embeddings = OllamaEmbeddings(model="gemma2:latest")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")


# retrieve the most relevant chukns from the vectorstore
def retrieve(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results]) #returnt die dokumente die wir haben


def generate_answer(query: str, context: str) -> str:
    response = chat(
        model="gemma2:latest", #modell angepasst
        messages=[
            {"role": "user", "content": f"Answer the question based on the context provided.\n\nContext: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response["message"]["content"]


st.title("Lineare Regression RAG with Ollama & Chroma")

user_query = st.chat_input("Stelle eine Frage über lineare Regression:")

if user_query:
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(user_query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(user_query, context)

    st.write("**Answer : ", answer)
