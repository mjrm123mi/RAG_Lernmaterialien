# RAG fürs Lernen – Startanleitung

Dieses Projekt stellt einen RAG (Retrieval-Augmented-Generation) fürs Lernen mit eigen Unterlagen bereit. 
Es können eigene PDFs (Vorlesungsfolien, Notizen usw.) in den Ordner `Data/` gelegt und dann Fragen dazu gestellt werden.
Zudem können mithilfe einer Frage passende Quellen und Seitenzahlen dazu gefunden werden.
---
## 1. Vorbereitung

### 1.1 Ollama starten
Über Ollama sich das Modell `gemma2` installieren. Das Modell kann wie folgt gestartet werden: 
```bash
ollama run gemma2
```

### 1.2 Virtuelle Umgebung aktivieren
Im Projektordner:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---
## 2. Index neu aufbauen

Immer wenn folgende Änderungen gemacht werden, sollte der Chroma-Index gelöscht und neu aufgebaut werden:

- PDFs im Ordner `Data/` wurden geändert, hinzugefügt oder gelöscht
- Chunking-Parameter (chunk size, overlap) geändert
- Embedding-Modell gewechselt

### 2.1 Alten Chroma-Index löschen
```bash
rm -rf chroma_db
```

### 2.2 Chroma-Index aus PDFs bauen
```bash
python build_index.py
```

---
## 3. RAG-App starten

Starte die Streamlit-App:
```bash
streamlit run app.py
```

Die App ist dann im Browser erreichbar (`http://localhost:8501`).

> Hinweis: Ollama muss parallel laufen, damit das LLM `gemma2:2b` Anfragen beantworten kann.

---
## 4. Automatisierte Evaluation

Für eine einfache Evaluation mit Precision@k, Recall@k und Cosine Similarity:
```bash
python evaluation.py
```


---
## 5. Kontakt

Bei Fragen kannst du dich gerne bei mir per Slack oder Mail melden:
`Maryam.Mirza@Student.HTW-Berlin.de`
