Wie starten?

 (Alte Chroma löschen bei Aenderungen der pdfs,chunksize, overlap oder Embedding-Modell-wechsel):
rm -rf chroma_db 

Index bauen:
python build_index.py

starten:
streamlit run app.py

Automatisierte Evaluation mit Precision@k, Recall@k und Cosine Similarity starten:
python evaluation.py

-----------
Olama starten:
ollama run gemma2

Venv aktivieren:
source .venv/bin/activate

Über http://localhost:11434 erreichbar

Streamlit starten:
streamlit run RAG_demo.py
