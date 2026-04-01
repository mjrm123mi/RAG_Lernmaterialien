import os
import glob
from typing import List

import streamlit as st
from ollama import chat

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_pdfs_from_data_folder(data_dir: str) -> list:
    """Load all PDFs from a folder into LangChain Documents."""
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    all_docs = []
    for path in pdf_paths:
        try:
            docs = PyPDFLoader(path).load()
            # ensure a stable source in metadata (handy for citations)
            for d in docs:
                d.metadata = dict(d.metadata or {})
                d.metadata.setdefault("source", os.path.basename(path))
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Konnte PDF nicht laden: {path} ({e})")
    return all_docs


@st.cache_resource(show_spinner=False)
def get_vectorstore(
    data_dir: str,
    persist_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    reindex: bool,
) -> Chroma:
    """Build or load a persistent Chroma DB."""
    embeddings = OllamaEmbeddings(model=embedding_model)

    has_existing = os.path.isdir(persist_dir) and any(
        os.path.exists(os.path.join(persist_dir, name)) for name in ("chroma.sqlite3",)
    )

    if has_existing and not reindex:
        st.info("Lade bestehenden Vektor-Index aus `chroma_db`...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    st.info(f"Baue neuen Vektor-Index aus PDFs in `{data_dir}`...")
    docs = load_pdfs_from_data_folder(data_dir)
    if not docs:
        st.error(
            f"Keine PDFs gefunden/geladen in '{data_dir}'. Lege PDFs in den Ordner oder prüfe den Pfad."
        )
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    st.success(f"{len(docs)} Seiten aus {len(set(d.metadata.get('source') for d in docs))} PDFs in {len(chunks)} Chunks aufgeteilt.")

    # rebuild
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    st.success("Neuer Index wurde gebaut und in `chroma_db` gespeichert.")
    return vectorstore


def retrieve(vectorstore: Chroma, query: str, k: int) -> List:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def format_context(docs: List) -> str:
    return "\n\n".join([d.page_content for d in docs])


def format_sources(docs: List) -> str:
    sources = []
    for d in docs:
        src = (d.metadata or {}).get("source")
        page = (d.metadata or {}).get("page")
        if src is None:
            continue
        if page is not None:
            sources.append(f"{src} (Seite {page + 1})")
        else:
            sources.append(str(src))
    # unique, keep order
    seen = set()
    uniq = []
    for s in sources:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return "\n".join(f"- {s}" for s in uniq)


def generate_answer(chat_model: str, query: str, context: str) -> str:
    response = chat(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": "Du bist ein hilfreicher Assistent. Antworte nur basierend auf dem gegebenen Kontext. Wenn der Kontext nicht reicht, sag ehrlich, dass du es nicht weißt.",
            },
            {
                "role": "user",
                "content": f"Kontext:\n{context}",
            },
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]


st.title("RAG über PDFs im Data/-Ordner (Ollama + Chroma)")

with st.sidebar:
    st.header("Einstellungen")
    data_dir = st.text_input("Data-Ordner", value="Data")
    persist_dir = st.text_input("Chroma Persist Dir", value="chroma_db")

    chat_model = st.text_input("Chat-Modell (Ollama)", value="gemma2:latest")
    embedding_model = st.text_input("Embedding-Modell (Ollama)", value="gemma2:latest")

    k = st.slider("Top-k Treffer", min_value=1, max_value=10, value=3)
    chunk_size = st.slider("Chunk Size", min_value=300, max_value=4000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=2000, value=200, step=50)

    reindex = st.checkbox("Index neu bauen (Reindex)", value=False)
    show_debug = st.checkbox("Debug: Kontext anzeigen", value=True)


with st.spinner("Vectorstore wird geladen/gebaut ..."):
    vectorstore = get_vectorstore(
        data_dir=data_dir,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reindex=reindex,
    )

user_query = st.chat_input("Stelle eine Frage zu den PDFs in Data/ ...")

if user_query:
    if vectorstore is None:
        st.error("Index ist nicht verfügbar. Bitte prüfe die Einstellungen und PDF-Dateien.")
    else:
        with st.spinner("Retrieval läuft ..."):
            retrieved_docs = retrieve(vectorstore, user_query, k=k)
            context = format_context(retrieved_docs)

        if show_debug:
            st.subheader("Gefundener Kontext (Debug)")
            st.text_area("Kontext", value=context, height=250)
            sources_md = format_sources(retrieved_docs)
            if sources_md:
                st.subheader("Quellen")
                st.markdown(sources_md)

        with st.spinner("Antwort wird generiert ..."):
            answer = generate_answer(chat_model, user_query, context)

        st.subheader("Antwort")
        st.write(answer)
