Wie starten?

Aktuell
 (alte Chroma löschen bei Aenderungen der pdfs,chunksize, overlap oder Embedding-Modell-wechsel):
rm -rf chroma_db 

Index bauen:
python build_index.py

starten:
streamlit run app.py



-----------
Olama starten:
ollama run gemma2

Weitere Infos:
/bye (fenster schließen)
ollama run codellama (ist für Prototypen zum programmieren)
ollama list (alle modelle zu sehen)
ollama ps (alle modelle die grad laufen)
ollama serve (kann über die ip mit dem bot der ki kommunizieren)
------------

Venv aktivieren:
source .venv/bin/activate

Über http://localhost:11434 erreichbar

Streamlit starten:
streamlit run RAG_demo.py
(Da dann Frage stellen, braucht etwas zum laden)


Quellen:
Olama Tutorial: https://www.youtube.com/watch?v=x5kDUgx-B8w
RAG Tutorial, Streamlit, Chrome, Olama: https://www.youtube.com/watch?v=nOvSQk9kWdE

