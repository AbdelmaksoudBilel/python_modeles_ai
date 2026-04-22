import json, re, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE  = "data/rag/chunk/rag_chunks.json"
OUTPUT_FILE = "data/rag/chunk/rag_chunks_meta.json"

# ── Mots-clés pour age_group ──────────────────────────────────────────────────
AGE_KEYWORDS = {
    "0-3": [
        "nourrisson", "bébé", "nouveau-né", "0 à 3", "0-3", "18 mois", "24 mois",
        "tout-petit", "toddler", "q-chat", "qchat", "m-chat", "mchat",
        "premiers mois", "première année",
    ],
    "4-6": [
        "maternelle", "préscolaire", "4 ans", "5 ans", "6 ans", "4-6",
        "petite section", "moyenne section", "grande section",
        "entrée à l'école", "premier apprentissage",
    ],
    "7-12": [
        "primaire", "élémentaire", "scolaire", "7 ans", "8 ans", "9 ans",
        "10 ans", "11 ans", "12 ans", "7-12", "cp", "ce1", "ce2", "cm1", "cm2",
        "enfant d'âge scolaire", "école primaire",
    ],
    "13+": [
        "adolescent", "ado", "lycée", "collège", "13 ans", "14 ans", "15 ans",
        "16 ans", "17 ans", "18 ans", "13+", "puberte", "puberté",
        "transition vers l'âge adulte", "adulte", "vie adulte",
    ],
}

# ── Mots-clés pour sexe ───────────────────────────────────────────────────────
SEXE_KEYWORDS = {
    "M": [
        "garçon", "fils", "frère", "lui", "il est", "son fils",
        "chez les garçons", "les garçons", "masculin",
    ],
    "F": [
        "fille", "sœur", "elle est", "sa fille",
        "chez les filles", "les filles", "féminin",
    ],
}


def detect_age_group(text: str) -> str:
    """Détecte le groupe d'âge depuis le texte du chunk."""
    text_lower = text.lower()
    scores = {age: 0 for age in AGE_KEYWORDS}

    for age_group, keywords in AGE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[age_group] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "all"


def detect_sexe(text: str) -> str:
    """Détecte le sexe cible depuis le texte du chunk."""
    text_lower = text.lower()
    scores = {"M": 0, "F": 0}

    for sexe, keywords in SEXE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[sexe] += 1

    if scores["M"] > scores["F"]:   return "M"
    if scores["F"] > scores["M"]:   return "F"
    return "all"


def enrich_chunks(input_file: str, output_file: str):
    logger.info(f"Chargement : {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"{len(chunks)} chunks chargés")

    age_stats  = {"0-3": 0, "4-6": 0, "7-12": 0, "13+": 0, "all": 0}
    sexe_stats = {"M": 0, "F": 0, "all": 0}

    for chunk in chunks:
        text = chunk.get("text", "")

        # Ajouter age_group si absent
        if "age_group" not in chunk:
            chunk["age_group"] = detect_age_group(text)
        age_stats[chunk["age_group"]] = age_stats.get(chunk["age_group"], 0) + 1

        # Ajouter sexe si absent
        if "sexe" not in chunk:
            chunk["sexe"] = detect_sexe(text)
        sexe_stats[chunk["sexe"]] = sexe_stats.get(chunk["sexe"], 0) + 1

    # Sauvegarder
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Sauvegardé → {output_file}")
    print(f"\n{'='*50}")
    print(f"  Chunks enrichis : {len(chunks)}")
    print(f"  Age groups      : {age_stats}")
    print(f"  Sexe            : {sexe_stats}")
    print(f"{'='*50}")


if __name__ == "__main__":
    enrich_chunks(INPUT_FILE, OUTPUT_FILE)