import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self):

        print("Chargement metadata...")
        with open("data/rag/vector/metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("Chargement FAISS...")
        self.index = faiss.read_index("data/rag/vector/faiss_index.bin")

        print("Chargement modèle embedding...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")

        print("Préparation corpus BM25...")
        corpus = [doc["text"] for doc in self.metadata]
        tokenized = [doc.split(" ") for doc in corpus]

        self.bm25 = BM25Okapi(tokenized)

        self.corpus = corpus

        print("Retriever prêt")

    # Recherche FAISS

    def search_embedding(self, query, k=5):

        q = "query: " + query

        embedding = self.model.encode(
            q,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, ids = self.index.search(
            np.array([embedding]),
            k
        )

        results = []

        for i, idx in enumerate(ids[0]):
            results.append({
                "text": self.metadata[idx]["text"],
                "score": float(scores[0][i]),
                "source": self.metadata[idx]["source_nom"]
            })

        return results

    # Recherche BM25

    def search_bm25(self, query, k=5):

        tokenized_query = query.split(" ")

        scores = self.bm25.get_scores(tokenized_query)

        top_k = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_k:
            results.append({
                "text": self.metadata[idx]["text"],
                "score": float(scores[idx]),
                "source": self.metadata[idx]["source_nom"]
            })

        return results

    # Hybrid search

    def search(self, query, k=5):

        emb_results = self.search_embedding(query, k)
        bm25_results = self.search_bm25(query, k)

        combined = {}

        for r in emb_results:
            combined[r["text"]] = r

        for r in bm25_results:
            if r["text"] not in combined:
                combined[r["text"]] = r

        results = list(combined.values())

        return results[:k]