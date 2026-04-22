import json, os, hashlib, logging
from datetime import datetime
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    logger.warning("faiss non installé → pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False
    logger.warning("sentence-transformers non installé")


# CONFIGURATION — même chemins que embed_chunks.py

CHUNKS_PATH = "data/rag/chunk/rag_chunks_meta.json"
FAISS_PATH  = "data/rag/vector/faiss_index.bin"
META_PATH   = "data/rag/vector/metadata.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

# Longueur min d'un snippet pour être ajouté
MIN_SNIPPET_LENGTH = 80

# Score min d'un résultat web pour être ajouté (0-1)
# Basé sur la longueur et la pertinence du snippet
MIN_QUALITY_SCORE  = 0.3


# CLASSE PRINCIPALE

class AutoLearning:
    """
    Ajoute les résultats web validés à la base RAG sans re-embedding complet.
    """

    def __init__(self):
        if not FAISS_OK:
            raise ImportError("pip install faiss-cpu")
        if not ST_OK:
            raise ImportError("pip install sentence-transformers")

        # Charger le modèle embedding (même que embed_chunks.py)
        logger.info(f"Chargement modèle embedding ({EMBED_MODEL})...")
        self.model = SentenceTransformer(EMBED_MODEL)
        logger.info("Modèle chargé ✔")

        # Charger l'index FAISS existant
        logger.info(f"Chargement index FAISS : {FAISS_PATH}")
        self.index = faiss.read_index(FAISS_PATH)
        logger.info(f"FAISS chargé ✔ ({self.index.ntotal} vecteurs)")

        # Charger les chunks existants
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Charger les métadonnées existantes
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Index rapide des chunk_ids existants (pour éviter les doublons)
        self.existing_ids = {c["chunk_id"] for c in self.chunks}

        logger.info(f"AutoLearning initialisé ✔ ({len(self.chunks)} chunks existants)")

    # ── CONVERSION WEB → CHUNK ────────────────────────────────────────────────

    def _web_result_to_chunk(self, result: dict, trouble: str,
                              age_group: str = "all") -> dict | None:
        """
        Convertit un résultat web en chunk compatible avec rag_chunks_meta.json.

        Args:
            result    : dict depuis web_search.py
                        { title, url, snippet, source, domain }
            trouble   : "TSA" | "RM" | "MIXTE"
            age_group : groupe d'âge cible

        Returns:
            dict chunk ou None si snippet trop court / doublon
        """
        snippet = result.get("snippet", "").strip()
        title   = result.get("title", "").strip()

        # Texte final = titre + snippet
        text = f"{title}. {snippet}" if title else snippet

        # Vérifier qualité minimale
        if len(text) < MIN_SNIPPET_LENGTH:
            logger.debug(f"Snippet trop court ({len(text)} chars) → ignoré")
            return None

        # Générer un chunk_id déterministe depuis l'URL
        chunk_id = hashlib.md5(result.get("url", text).encode()).hexdigest()[:8]
        doc_id   = hashlib.md5(result.get("domain", "web").encode()).hexdigest()[:8]

        # Vérifier doublon
        if chunk_id in self.existing_ids:
            logger.debug(f"Chunk déjà existant : {chunk_id} → ignoré")
            return None

        return {
            "chunk_id"   : chunk_id,
            "doc_id"     : doc_id,
            "source_nom" : result.get("domain", "web"),
            "source_type": "web_auto",        # marqueur auto-learning
            "trouble"    : trouble.upper(),
            "categorie"  : "web",
            "langue"     : "fr",
            "age_group"  : age_group,
            "sexe"       : "all",
            "page"       : None,
            "chunk_index": 0,
            "text"       : text,
            "word_count" : len(text.split()),
            "url"        : result.get("url", ""),
            "added_at"   : datetime.now().isoformat(),
        }

    # ── EMBEDDING DE NOUVEAUX CHUNKS ─────────────────────────────────────────

    def _embed_new_chunks(self, new_chunks: list) -> np.ndarray:
        """
        Génère les embeddings pour les nouveaux chunks uniquement.
        Même format que embed_chunks.py : "passage: " + text

        Returns:
            np.ndarray shape (n, dimension)
        """
        texts = [f"passage: {c['text']}" for c in new_chunks]

        logger.info(f"Embedding {len(texts)} nouveaux chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,  # ← cohérence avec embed_chunks.py
        )
        return embeddings.astype("float32")

    # ── SAUVEGARDE ────────────────────────────────────────────────────────────

    def _save(self):
        """Sauvegarde FAISS + chunks + metadata sur disque."""

        # Sauvegarder FAISS
        faiss.write_index(self.index, FAISS_PATH)
        logger.info(f"FAISS sauvegardé ✔ ({self.index.ntotal} vecteurs)")

        # Sauvegarder chunks
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunks sauvegardés ✔ ({len(self.chunks)} total)")

        # Sauvegarder metadata
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata sauvegardée ✔ ({len(self.metadata)} total)")

    # ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────

    def add_web_results(self, web_results: list, trouble: str = "TSA",
                        age_group: str = "all") -> int:
        """
        Ajoute les résultats web à la base RAG.

        Args:
            web_results : liste depuis web_search.py
            trouble     : "TSA" | "RM" | "MIXTE"
            age_group   : groupe d'âge cible (optionnel)

        Returns:
            Nombre de chunks effectivement ajoutés
        """
        if not web_results:
            logger.info("Aucun résultat web à ajouter")
            return 0

        # ── Étape 1 : Convertir en chunks ────────────────────────────────
        new_chunks = []
        for result in web_results:
            chunk = self._web_result_to_chunk(result, trouble, age_group)
            if chunk:
                new_chunks.append(chunk)

        if not new_chunks:
            logger.info("Aucun nouveau chunk valide après filtrage")
            return 0

        logger.info(f"{len(new_chunks)} nouveaux chunks valides")

        # ── Étape 2 : Générer embeddings ──────────────────────────────────
        embeddings = self._embed_new_chunks(new_chunks)

        # ── Étape 3 : Ajouter à FAISS ────────────────────────────────────
        self.index.add(embeddings)

        # ── Étape 4 : Mettre à jour chunks + metadata ─────────────────────
        for chunk in new_chunks:
            # Ajouter aux chunks
            self.chunks.append(chunk)
            self.existing_ids.add(chunk["chunk_id"])

            # Ajouter aux métadonnées FAISS (même format que embed_chunks.py)
            self.metadata.append({
                "chunk_id"  : chunk["chunk_id"],
                "doc_id"    : chunk["doc_id"],
                "source_nom": chunk["source_nom"],
                "source_type": chunk["source_type"],
                "categorie" : chunk["categorie"],
                "page"      : chunk["page"],
                "text"      : chunk["text"],
            })

        # ── Étape 5 : Sauvegarder ────────────────────────────────────────
        self._save()

        logger.info(f"Auto-learning ✔ : {len(new_chunks)} chunks ajoutés")
        return len(new_chunks)

    # ── STATISTIQUES ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Retourne les statistiques de la base RAG actuelle."""
        web_chunks = [c for c in self.chunks if c.get("source_type") == "web_auto"]
        return {
            "total_chunks"  : len(self.chunks),
            "faiss_vectors" : self.index.ntotal,
            "web_chunks"    : len(web_chunks),
            "original_chunks": len(self.chunks) - len(web_chunks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Simuler des résultats web
    fake_web_results = [
        {
            "title"  : "Gérer les crises chez l'enfant autiste",
            "url"    : "https://has-sante.fr/autisme-crises-gestion",
            "snippet": (
                "Lors d'une crise chez un enfant autiste, il est recommandé "
                "de maintenir un environnement calme et prévisible. Évitez les "
                "stimulations sensorielles excessives. Proposez un espace de "
                "retrait sécurisé. Utilisez un langage simple et rassurant. "
                "Restez calme et patient pour ne pas amplifier la crise."
            ),
            "source" : "has-sante.fr",
            "domain" : "has-sante.fr",
        },
        {
            "title"  : "Communication PECS pour enfants non verbaux",
            "url"    : "https://autismesociete.org/pecs-communication",
            "snippet": (
                "Le système PECS (Picture Exchange Communication System) "
                "permet aux enfants non verbaux atteints de TSA d'initier "
                "une communication fonctionnelle via des pictogrammes. "
                "Il se déroule en 6 phases progressives et a montré son "
                "efficacité dans de nombreuses études cliniques."
            ),
            "source" : "autismesociete.org",
            "domain" : "autismesociete.org",
        },
        {
            "title"  : "X",   # trop court → sera ignoré
            "url"    : "https://example.com/x",
            "snippet": "court",
            "source" : "example.com",
            "domain" : "example.com",
        },
    ]

    print("\n" + "="*60)
    print("  TEST — Auto-Learning")
    print("="*60)

    al = AutoLearning()

    print(f"\n  Avant : {al.stats()}")

    added = al.add_web_results(fake_web_results, trouble="TSA", age_group="4-6")

    print(f"\n  Après : {al.stats()}")
    print(f"  Chunks ajoutés : {added}")
    print("="*60)