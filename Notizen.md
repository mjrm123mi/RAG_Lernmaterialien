#Notizen

Lernmaterialien

Sentence Transformers  macht aus Text Zahlen (Embeddings), damit wird Text vergleicbar


Vektordatenbank:
Milvus Datenbank, weil opensource, passend für python, und kann für dichte vektoren (Bild, Textverarbeitung verwendet werden) (aber überdimensioniert bei nur ein paar dokumenten)
besser Chroma (FAISS nicht recherchiert)

Welche Daten?
PDFs von Foliensätzen aus dem Studium (Frage muss ich Genehmigung holen?)
Mit Langchain einlesen (Pipeline)


Wie evaluieren?
Testdatensatz erstellen mit Antworten auf Fragen.

Accuracy, Recall, Precision F2 Score ausrechnen und selbst prüfen ob Antworten passend sind. Evtl ein BertScore Modell verwenden, aber muss noch schauen ob das passend wäre.

Ist unter den Chunks mindestens einer relevant?
Eigene manuelle Stichproben

LLM as judge eher nicht

1. Retrieval bewerten, ob richtiges Dokument gefunden wird
2. Prüfen ob Antwort richtig ist
3. Halluziniert es?
4. Mischung aus einfachen, komplexen und mehrdeutigen Fragen 
5. bilder können nicht beantwortet werden
