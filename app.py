import streamlit as st
from ollama import chat
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdfs"
CHAT_MODEL = "gemma2:2b"

st.set_page_config(page_title="RAG fürs Lernen")
st.title("RAG fürs Lernen")

@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore

vectorstore = load_vectorstore()

def retrieve_with_sources(query: str):
    results = vectorstore.similarity_search(query, k=5)
    return  results

def format_context(results):
    parts = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unbekannt")
        page = doc.metadata.get("page", "?")
        text = doc.page_content
        parts.append(f"[Treffer {i} | Quelle: {source} | Seite: {page}]\n{text}")
    return "\n\n".join(parts)



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
            {
                "role": "user",
                "content": f"Beantworte die Frage nur mit Hilfe dieses Kontexts:\n\n{context}\n\nFrage: {query}"
            }
        ],
        options={"temperature": 0}
    )
    return response["message"]["content"]

user_query = st.chat_input("Stelle eine Frage zu deinen PDFs")

if user_query:
    with st.spinner("Suche relevante Stellen..."):
        docs = retrieve_with_sources(user_query)
        context = format_context(docs)

    with st.spinner("Generiere Antwort..."):
        answer = generate_answer(user_query, context)

    st.write("**Antwort:**", answer)

    with st.expander("Gefundene Textstellen anzeigen"):
        st.text(context)