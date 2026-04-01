from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ollama import chat
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# load the document
docs = PyPDFLoader("Data/windows-whats-new.pdf").load()

# split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# embed the chunks and store them in chroma db
embeddings = OllamaEmbeddings(model="gemma2:2b")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")


# retrieve the most relevant chukns from the vectorstore
def retrieve(query: str) -> str:
    results = vectorstore.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])


def generate_answer(query: str, context: str) -> str:
    response = chat(
        model="gemma2:2b",
        messages=[

            {
                "role": "system",
                "content": (
                    "Du beantwortest Fragen ausschließlich anhand des bereitgestellten Kontexts. "
                    "Wenn die Antwort nicht eindeutig im Kontext steht, antworte exakt mit: "
                    "'Ich weiß es nicht'. Erfinde nichts und nutze kein Weltwissen."
                )
            },

            {"role": "user", "content": f"Answer the question based on the context provided.\n\nContext: {context}"},
            {"role": "user", "content": query}

        ],
        options={"temperature": 0} #temperatur auf 0, damit wenige kreative Ergänzungen
    )
    return response["message"]["content"]


st.title("Windows 11 RAG with Ollama & Chroma")

user_query = st.chat_input("Ask a question about Windows 11:")

if user_query:
    with st.spinner("Retrieving relevant information..."):
        context = retrieve(user_query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(user_query, context)

    st.write("**Answer : ", answer)
