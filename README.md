Startanleitung:
-----------------------------------------------
Vorbereitung:

Olama starten:
ollama run gemma2

Venv aktivieren:
source .venv/bin/activate
pip install -r requirements.txt
-----------------------------------------------

Bei Änderungenen:
(Chroma löschen bei Aenderungen der pdfs,chunksize, overlap oder Embedding-Modell-wechsel):
rm -rf chroma_db 

Index bauen:
python build_index.py

starten:
streamlit run app.py
Über http://localhost:11434 erreichbar
-----------------------------------------------
Automatisierte Evaluation mit Precision@k, Recall@k und Cosine Similarity starten:
python evaluation.py

-----------------------------------------------
Bei Fragen mir gerne eine Mail oder per Slack schreiben.

Maryam.Mirza@Student.HTW-Berlin.de