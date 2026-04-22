import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths

CHUNKS_PATH = "data/rag/chunk/rag_chunks_meta.json"
VECTOR_DIR = "data/rag/vector"

os.makedirs(VECTOR_DIR, exist_ok=True)

FAISS_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_DIR, "metadata.json")

# Charger chunks

print("Chargement des chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks :", len(chunks))


# Charger modèle embedding

print("Chargement modèle embedding...")

model = SentenceTransformer("intfloat/multilingual-e5-large")

# Préparer textes

texts = []

for c in chunks:
    # format recommandé pour e5
    texts.append("passage: " + c["text"])

# Générer embeddings

print("Génération embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Shape embeddings :", embeddings.shape)

# Création index FAISS

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index size :", index.ntotal)

# Sauvegarder index

faiss.write_index(index, FAISS_PATH)

print("Index sauvegardé :", FAISS_PATH)

# Sauvegarder metadata

metadata = []

for i, c in enumerate(chunks):
    metadata.append({
        "chunk_id": c.get("chunk_id"),
        "doc_id": c.get("doc_id"),
        "source_nom": c.get("source_nom"),
        "source_type": c.get("source_type"),
        "categorie": c.get("categorie"),
        "page": c.get("page"),
        "text": c.get("text")
    })

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Metadata sauvegardé :", META_PATH)

print("Embedding terminé ✅")