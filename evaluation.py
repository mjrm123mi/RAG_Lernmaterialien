import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

from app import retrieve_with_sources, format_context, generate_answer

K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def embed(text: str):
    return embeddings.embed_query(text)


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]


eval_data = [
    {
        "query": "Was ist ein Hyperparameter?",
        "expected_answer": "Das sind die Parameter die vor dem eigentlichen Training festgelegt werden",
        "relevant_docs": ["Pruefungsvorbereitung"]
    },
    {
        "query": "Wie kann ich den Einfluss von Hyperparametern messen?",
        "expected_answer": "Das Modell mit verschiedneen Hyperparametern trainieren und die Performance auf den Validierungsdaten vergleichen",
        "relevant_docs": ["Pruefungsvorbereitung"]
    },
    {
        "query": "Wie kann ich die optimalen Hyperparameter-Werte finden?",
        "expected_answer": "Cross Validation und Grid Search",
        "relevant_docs": ["Pruefungsvorbereitung"]
    },
    {
        "query": "Was ist Overfitting?",
        "expected_answer": "Überanpassung",
        "relevant_docs": ["Pruefungsvorbereitung"]
    },
    {
        "query": "Wofür wird Machine Learning verwendet?",
        "expected_answer": "Seit einigen Jahren ist Machine Learning (ML) zu einem Standardwerkzeug für fast alle Aufgaben geworden, bei denen Informationen aus Daten extrahiert werden müssen",
        "relevant_docs": ["Einführung in Machinelles Lernen"]
    },
{
        "query": "Was ist ein Domain Set?",
        "expected_answer": "Urbildmenge",
        "relevant_docs": ["Einführung in Machinelles Lernen"]
    },

{
        "query": "Wo kommt die IQR Regel her?",
        "expected_answer": "von John Tukey",
        "relevant_docs": ["Data Preprocessing"]
    },

]


def precision_at_k(retrieved_docs, relevant_docs, k=5):
    hits = 0
    for doc in retrieved_docs[:k]:
        source = doc.metadata.get("source", "")
        if any(rel.lower() in source.lower() for rel in relevant_docs):
            hits += 1
    return hits / k


def recall_at_k(retrieved_docs, relevant_docs, k=5):
    retrieved_sources = set()

    for doc in retrieved_docs[:k]:
        source = doc.metadata.get("source", "").lower()

        for rel in relevant_docs:
            if rel.lower() in source:
                retrieved_sources.add(rel.lower())

    return len(retrieved_sources) / len(relevant_docs) if relevant_docs else 0


def evaluate_single(query, expected_answer, relevant_docs):
    # Retrieval
    docs = retrieve_with_sources(query)

    # Context bauen
    context = format_context(docs)

    # Antwort generieren
    answer = generate_answer(query, context)

    # Precision@k
    p_at_k = precision_at_k(docs, relevant_docs, K)

    #Recall
    r_at_k = recall_at_k(docs, relevant_docs, K)

    # Cosine Similarity
    emb_pred = embed(answer)
    emb_gt = embed(expected_answer)
    cos_sim_score = cosine_sim(emb_pred, emb_gt)

    return {
        "query": query,
        "answer": answer,
        "expected": expected_answer,
        "precision@k": p_at_k,
        "recall@k": r_at_k,
        "cosine_similarity": cos_sim_score,
    }


def run_evaluation(eval_data):
    results = []

    for i, item in enumerate(eval_data, 1):
        print(f"Frage {i}: {item['query']}")

        res = evaluate_single(
            item["query"],
            item["expected_answer"],
            item["relevant_docs"]
        )

        results.append(res)

        print(f"Antwort: {res['answer']}")
        print(f"Precision@{K}: {res['precision@k']:.2f}")
        print(f"Recall@{K}: {res['recall@k']:.2f}")
        print(f"Cosine Similarity: {res['cosine_similarity']:.2f}")
        print("-" * 50)

    avg_precision = np.mean([r["precision@k"] for r in results])
    avg_recall = np.mean([r["recall@k"] for r in results])
    avg_cosine = np.mean([r["cosine_similarity"] for r in results])

    print("\n Ergebnisse:")
    print(f"Durchschnitt Precision@{K}: {avg_precision:.3f}")
    print(f"Durchschnitt Recall@{K}: {avg_recall:.3f}")
    print(f"Durchschnitt Cosine Similarity: {avg_cosine:.3f}")

    return results


if __name__ == "__main__":
    run_evaluation(eval_data)
