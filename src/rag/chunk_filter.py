import json, logging
import numpy as np
from pathlib import Path

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
    logger.warning("sentence-transformers non installé → pip install sentence-transformers")


# CONFIGURATION

EMBED_MODEL  = "intfloat/multilingual-e5-large"
TOP_K        = 5     # chunks par recherche
TOP_K_PROFIL = 3     # chunks par aspect profil
SCORE_THRESHOLD = 0.6  # seuil déclenchement web search

# Groupes d'âge : mapping âge réel → age_group chunk
AGE_GROUP_MAP = [
    (0,  3,  "0-3"),
    (4,  6,  "4-6"),
    (7,  12, "7-12"),
    (13, 99, "13+"),
]

def get_age_group(age: int) -> str:
    for lo, hi, label in AGE_GROUP_MAP:
        if lo <= age <= hi:
            return label
    return "all"


# REQUÊTES AUTOMATIQUES PAR ASPECT PROFIL

def build_profile_queries(profile: dict) -> dict:
    """
    Génère des requêtes automatiques depuis le profil pour la Recherche 2.

    Returns:
        { "aspect": "requête texte", ... }
        Seulement les aspects actifs dans le profil.
    """
    queries = {}

    def val(key, default=3):
        v = profile.get(key)
        try: return int(v) if v is not None else default
        except: return default

    trouble = str(profile.get("prediction", "TSA")).upper()

    # ── Communication ─────────────────────────────────────────────────────
    comm = val("PR_QF1A", 1)
    if comm == 2:
        queries["communication"] = (
            "méthodes de communication alternative et augmentative pour enfant TSA"
        )
    elif comm == 3:
        queries["communication"] = (
            "communication non verbale PECS pictogrammes enfant autiste sans langage"
        )

    # ── Comportements actifs ──────────────────────────────────────────────
    if val("PR_QO1_A_COMBINE") == 1:
        queries["comportement_agression"] = (
            "gérer agression physique enfant autiste stratégies désescalade"
        )
    if val("PR_QO1_C_COMBINE") == 1:
        queries["comportement_automutilation"] = (
            "automutilation enfant TSA prévention gestion comportement"
        )
    if val("PR_QO1_B_COMBINE") == 1:
        queries["comportement_destruction"] = (
            "destruction biens enfant autiste intervention comportementale"
        )
    if val("PR_QO1_E_COMBINE") == 1:
        queries["comportement_fugue"] = (
            "fugue errance enfant autiste prévention sécurité"
        )

    # ── Santé / comorbidités ──────────────────────────────────────────────
    if val("PR_QN1_D") in (1, 2):
        queries["sante_anxiete"] = (
            f"troubles anxieux {'autisme TSA' if trouble == 'TSA' else 'déficience intellectuelle'} "
            "gestion anxiété enfant"
        )
    if val("PR_QN1_G") in (1, 2):
        queries["sante_epilepsie"] = (
            "épilepsie enfant autiste crises convulsions recommandations"
        )
    if val("PR_QN1_C") in (1, 2):
        queries["sante_respiratoire"] = (
            "asthme maladie respiratoire enfant handicap gestion quotidien"
        )

    # ── Niveau de soutien ─────────────────────────────────────────────────
    severity = val("PR_QQ", 1)
    if severity >= 3:
        queries["soutien_intensif"] = (
            f"accompagnement intensif {'TSA sévère' if trouble == 'TSA' else 'déficience intellectuelle sévère'} "
            "soutien quotidien parents"
        )

    # ── Mobilité / autonomie ──────────────────────────────────────────────
    if val("PR_QH1B") == 1:
        queries["mobilite"] = (
            "fauteuil roulant enfant handicap autonomie déplacement"
        )
    if val("PR_QK1") >= 3:
        queries["alimentation"] = (
            "troubles alimentaires enfant autiste alimentation sélective"
        )

    return queries


# CLASSE PRINCIPALE

class ChunkFilter:
    """
    Double recherche RAG filtrée par profil d'enfant.

    Recherche 1 : chunks "réponse"   → filtre trouble + age_group
    Recherche 2 : chunks "profil"    → filtre par aspect + requête auto
    """

    def __init__(self, chunks_file: str, faiss_index_file: str,
                 metadata_file: str):
        """
        Args:
            chunks_file      : chemin vers rag_chunks_meta.json
            faiss_index_file : chemin vers faiss_index.bin
            metadata_file    : chemin vers metadata.json
        """
        # Charger les chunks
        logger.info(f"Chargement chunks : {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"{len(self.chunks)} chunks chargés")

        # Charger FAISS
        if FAISS_OK:
            logger.info("Chargement index FAISS...")
            self.index = faiss.read_index(faiss_index_file)
            logger.info("FAISS chargé ✔")
        else:
            self.index = None

        # Charger métadonnées FAISS (mapping index → chunk_id)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.faiss_metadata = json.load(f)

        # Charger le modèle d'embedding
        if ST_OK:
            logger.info(f"Chargement embedding ({EMBED_MODEL})...")
            self.embedder = SentenceTransformer(EMBED_MODEL)
            logger.info("Embedding ✔")
        else:
            self.embedder = None

        # Index rapide : chunk_id → chunk
        self.chunk_index = {c["chunk_id"]: c for c in self.chunks}

        logger.info("ChunkFilter initialisé ✔")

    # ── FILTRAGE DES CHUNKS ───────────────────────────────────────────────────

    def filter_chunks(self, trouble: str = None, age_group: str = None,
                      categorie: str = None, sexe: str = None) -> list:
        """
        Filtre les chunks selon les critères donnés.
        Les critères None ou "all" ne filtrent pas.

        Returns:
            Liste des chunks filtrés
        """
        result = self.chunks

        if trouble and trouble != "all":
            trouble_up = trouble.upper()
            result = [
                c for c in result
                if (c.get("trouble") or "").upper() in (trouble_up, "MIXTE", "ALL")
            ]

        if age_group and age_group != "all":
            result = [
                c for c in result
                if (c.get("age_group") or "all") in (age_group, "all")
            ]

        if categorie and categorie != "all":
            result = [
                c for c in result
                if (c.get("categorie") or "").lower() == categorie.lower()
            ]

        if sexe and sexe != "all":
            result = [
                c for c in result
                if (c.get("sexe") or "all") in (sexe, "all")
            ]

        logger.info(f"Filtre → {len(result)}/{len(self.chunks)} chunks")
        return result

    # ── EMBEDDING + RECHERCHE VECTORIELLE ────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Encode un texte en vecteur."""
        if self.embedder is None:
            raise ImportError("sentence-transformers non disponible")
        vec = self.embedder.encode(
            f"query: {text}",
            normalize_embeddings=True
        )
        return vec.astype("float32").reshape(1, -1)

    def _search_in_subset(self, query: str, subset_chunks: list,
                          top_k: int) -> list:
        """
        Recherche vectorielle dans un sous-ensemble de chunks.

        Stratégie :
            1. Récupérer les IDs FAISS du sous-ensemble
            2. Chercher dans FAISS avec top_k * 10 candidats
            3. Filtrer pour garder seulement ceux du sous-ensemble
            4. Retourner top_k

        Returns:
            Liste de dicts { chunk, score }
        """
        if not subset_chunks:
            return []

        # Ensemble des chunk_ids autorisés
        allowed_ids = {c["chunk_id"] for c in subset_chunks}

        # Encoder la requête
        query_vec = self._embed(query)

        # Chercher largement dans FAISS (top_k * 20 pour avoir assez de candidats)
        k_search = min(top_k * 20, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k_search)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_metadata):
                continue
            chunk_id = self.faiss_metadata[idx].get("chunk_id")
            if chunk_id not in allowed_ids:
                continue
            chunk = self.chunk_index.get(chunk_id)
            if chunk:
                # Score normalisé 0-1 (FAISS inner product avec vecteurs normalisés)
                score = float(np.clip(dist, 0.0, 1.0))
                results.append({
                    "chunk"      : chunk,
                    "score"      : round(score, 4),
                    "chunk_id"   : chunk_id,
                    "source_nom" : chunk.get("source_nom", "?"),
                    "categorie"  : chunk.get("categorie", "?"),
                })
            if len(results) >= top_k:
                break

        return results

    # ── SCORING ───────────────────────────────────────────────────────────────

    def _compute_avg_score(self, chunks: list) -> float:
        """
        Calcule le score moyen des chunks retournés.
        Utilisé pour décider si on déclenche la recherche web.

        Returns:
            Score moyen entre 0 et 1. 0 si aucun chunk.
        """
        if not chunks:
            return 0.0
        return sum(h["score"] for h in chunks) / len(chunks)

    # ── RECHERCHE 1 : RÉPONSE ─────────────────────────────────────────────────

    def search_response(self, question: str, trouble: str,
                        age_group: str, sexe: str = "all",
                        top_k: int = TOP_K) -> list:
        """
        Recherche 1 : chunks pour répondre à la question.
        Filtre dur : trouble + age_group (+ sexe optionnel)

        Returns:
            Liste de dicts { chunk, score }
        """
        logger.info(f"Recherche 1 (réponse) | trouble={trouble} age={age_group}")
        subset = self.filter_chunks(trouble=trouble, age_group=age_group,
                                    sexe=sexe)
        return self._search_in_subset(question, subset, top_k)

    # ── RECHERCHE 2 : CONTEXTE PROFIL ─────────────────────────────────────────

    def search_profile_context(self, profile: dict,
                                top_k: int = TOP_K_PROFIL) -> dict:
        """
        Recherche 2 : chunks pour contextualiser selon le profil.
        Requêtes automatiques générées depuis le profil.

        Returns:
            { "aspect": [ {chunk, score}, ... ], ... }
        """
        trouble   = str(profile.get("prediction", "TSA")).upper()
        queries   = build_profile_queries(profile)
        results   = {}

        for aspect, query in queries.items():
            logger.info(f"Recherche profil | aspect={aspect}")

            # Filtre : trouble uniquement (pas d'age_group pour le contexte profil)
            subset = self.filter_chunks(trouble=trouble)
            hits   = self._search_in_subset(query, subset, top_k)

            if hits:
                results[aspect] = hits

        return results

    # ── PIPELINE COMPLET ──────────────────────────────────────────────────────

    def search(self, question: str, profile: dict,
               top_k: int = TOP_K) -> dict:
        """
        Double recherche RAG complète + construction contexte profil LLM.

        Args:
            question : message du parent (déjà traduit en français)
            profile  : dict JSON du profil enfant
            top_k    : nombre de chunks pour la recherche réponse

        Returns:
            {
                "response_chunks"  : [ {chunk, score}, ... ],
                "profile_chunks"   : { "communication": [...], ... },
                "avg_score"        : 0.74,
                "web_triggered"    : False,
                "web_results"      : [...],
                "prompt_block"     : str,        ← chunks pour le prompt
                "profile_context"  : str,        ← contexte profil pour le prompt
                "flags"            : { ... },
                "active_modules"   : [ ... ],
                "filter_used"      : { trouble, age_group, sexe },
            }
        """
        # ── Extraire paramètres de filtre ─────────────────────────────────
        trouble   = str(profile.get("prediction", "TSA")).upper()
        age       = int(profile.get("Age_Years", profile.get("age", 0)) or 0)
        age_group = get_age_group(age)
        sexe      = str(profile.get("Sex", profile.get("sexe", "all")))
        if sexe in ("1", "MALE", "HOMME"):     sexe = "M"
        elif sexe in ("2", "FEMALE", "FEMME"): sexe = "F"
        else:                                   sexe = "all"

        # ── Construire contexte profil (ex profile_filter.py) ─────────────
        profile_context, flags, active_modules = self._build_profile_context(
            profile, trouble, age, age_group, sexe
        )

        # ── Recherche 1 : Réponse ─────────────────────────────────────────
        response_chunks = self.search_response(
            question, trouble, age_group, sexe, top_k
        )

        # ── Scoring + Web search ──────────────────────────────────────────
        avg_score     = self._compute_avg_score(response_chunks)
        web_results   = []
        web_triggered = False

        if avg_score < SCORE_THRESHOLD:
            logger.warning(
                f"Score RAG bas ({avg_score:.2f} < {SCORE_THRESHOLD}) "
                f"→ déclenchement web search"
            )
            web_triggered = True
            try:
                from web_search import WebSearch
                ws = WebSearch()
                web_results = ws.search(question, trouble)
            except Exception as e:
                logger.error(f"Web search échouée : {e}")

        # ── Recherche 2 : Contexte Profil ─────────────────────────────────
        profile_chunks = self.search_profile_context(profile, TOP_K_PROFIL)

        # ── Construction du bloc prompt ───────────────────────────────────
        prompt_block = self._build_prompt_block(
            response_chunks, profile_chunks, web_results
        )

        return {
            "response_chunks": response_chunks,
            "profile_chunks" : profile_chunks,
            "avg_score"      : round(avg_score, 4),
            "web_triggered"  : web_triggered,
            "web_results"    : web_results,
            "prompt_block"   : prompt_block,
            "profile_context": profile_context,   # ← nouveau
            "flags"          : flags,              # ← nouveau
            "active_modules" : active_modules,    # ← nouveau
            "filter_used"    : {
                "trouble"  : trouble,
                "age_group": age_group,
                "sexe"     : sexe,
            },
        }

    # ── CONTEXTE PROFIL (ex profile_filter.py) ───────────────────────────────

    def _build_profile_context(self, profile: dict, trouble: str,
                                age: int, age_group: str,
                                sexe: str) -> tuple:
        """
        Construit le contexte textuel du profil enfant pour le prompt LLM.
        Remplace profile_filter.py — intégré directement ici.

        Returns:
            (profile_context: str, flags: dict, active_modules: list)
        """
        def val(key, default=3):
            v = profile.get(key)
            try: return int(v) if v is not None else default
            except: return default

        # Flags
        flags = {
            "non_verbal"      : val("PR_QF1A") in (2, 3),
            "agressif"        : val("PR_QO1_A_COMBINE") == 1,
            "auto_mutilation" : val("PR_QO1_C_COMBINE") == 1,
            "destruction"     : val("PR_QO1_B_COMBINE") == 1,
            "fugue"           : val("PR_QO1_E_COMBINE") == 1,
            "epilepsie"       : val("PR_QN1_G") in (1, 2),
            "anxiete"         : val("PR_QN1_D") in (1, 2),
            "diabete"         : val("PR_QN1_F") in (1, 2),
            "lesion_cerebrale": val("PR_QN1_H") in (1, 2),
            "fauteuil_roulant": val("PR_QH1B") == 1,
            "aide_auditive"   : val("PR_QI1", 1) >= 4,
            "aide_visuelle"   : val("PR_QJ1", 1) >= 4,
            "aide_repas"      : val("PR_QK1", 1) >= 3,
        }
        severity = val("PR_QQ", 1)
        severity = severity if severity in (1, 2, 3, 4) else 1

        soutien_labels = {
            1: "Soutien léger (non quotidien)",
            2: "Soutien quotidien réduit",
            3: "Soutien quotidien modéré",
            4: "Soutien quotidien important",
        }

        # Modules actifs
        modules = []
        if trouble == "TSA":              modules += ["tsa", "teacch"]
        elif trouble == "RM":             modules += ["rm", "autonomie"]
        elif trouble == "MIXTE":          modules += ["tsa", "rm", "teacch", "autonomie"]
        if flags["non_verbal"]:           modules.append("aac")
        if flags["agressif"] or flags["auto_mutilation"]: modules += ["crise", "comportement"]
        if trouble in ("TSA", "MIXTE") and severity >= 2: modules.append("sensoriel")
        if flags["anxiete"]:              modules.append("anxiete")
        if flags["epilepsie"]:            modules.append("epilepsie")
        active_modules = list(dict.fromkeys(modules))

        # Texte profil
        sexe_label  = "garçon" if sexe == "M" else "fille" if sexe == "F" else "enfant"
        age_str     = f"{age} ans" if age > 0 else "âge non précisé"
        trouble_map = {
            "TSA"  : "Trouble du Spectre Autistique (TSA)",
            "RM"   : "Déficience Intellectuelle (RM)",
            "MIXTE": "TSA avec Déficience Intellectuelle (MIXTE)",
        }
        conf     = profile.get("confidence")
        conf_str = f" (confiance : {conf:.0%})" if conf else ""
        comm_map = {1: "Langage parlé", 2: "Communication alternative",
                    3: "Pas de communication verbale"}
        comm_label = comm_map.get(val("PR_QF1A", 1), "Inconnu")

        lines = [
            f"Profil : {sexe_label} de {age_str}.",
            f"Diagnostic estimé : {trouble_map.get(trouble, trouble)}{conf_str}.",
            f"Niveau de soutien : {soutien_labels.get(severity)} (niveau {severity}/4).",
            f"Communication : {comm_label}.",
        ]
        if flags["non_verbal"]:
            lines.append("→ Non verbal : recommander AAC / PECS / pictogrammes.")

        comportements = [k for k, v in {
            "agression physique": flags["agressif"],
            "automutilation"    : flags["auto_mutilation"],
            "destruction"       : flags["destruction"],
            "fugues"            : flags["fugue"],
        }.items() if v]
        if comportements:
            lines.append(f"Comportements actifs : {', '.join(comportements)}.")
            lines.append("→ Priorité : stratégies de gestion de crise.")

        comorbidites = [k for k, v in {
            "épilepsie"        : flags["epilepsie"],
            "troubles anxieux" : flags["anxiete"],
            "diabète"          : flags["diabete"],
            "lésion cérébrale" : flags["lesion_cerebrale"],
        }.items() if v]
        if comorbidites:
            lines.append(f"Comorbidités : {', '.join(comorbidites)}.")

        lines.append(f"Modules activés : {', '.join(active_modules)}.")
        profile_context = "\n".join(lines)

        return profile_context, flags, active_modules

    # ── CONSTRUCTION DU BLOC PROMPT ───────────────────────────────────────────

    def _build_prompt_block(self, response_chunks: list,
                             profile_chunks: dict,
                             web_results: list = None) -> str:
        lines = []

        # ── Bloc 1 : Sources réponse ──────────────────────────────────────
        lines.append("=== Sources de réponse ===")
        if response_chunks:
            for i, hit in enumerate(response_chunks, 1):
                c = hit["chunk"]
                source = c.get('source_nom') or c.get('source', '?')
                score  = hit.get('score', 0)
                lines.append(
                    f"[Source {i} — {source} | score={score:.2f}]\n"
                    f"{c['text'][:400].strip()}\n"
                )
        else:
            lines.append("Aucun chunk trouvé.")

        # ── Bloc 2 : Sources web (si score bas) ───────────────────────────
        if web_results:
            lines.append("\n=== Sources web (complément) ===")
            for i, r in enumerate(web_results, 1):
                lines.append(
                    f"[{i}] ({r.get('source', '?')}) "
                    f"{r.get('snippet', '')[:300].strip()}"
                )

        # ── Bloc 3 : Contexte profil ──────────────────────────────────────
        if profile_chunks:
            lines.append("\n=== Contexte profil ===")
            aspect_labels = {
                "communication"              : "Communication",
                "comportement_agression"     : "Comportement — Agression",
                "comportement_automutilation": "Comportement — Automutilation",
                "comportement_destruction"   : "Comportement — Destruction",
                "comportement_fugue"         : "Comportement — Fugue",
                "sante_anxiete"              : "Santé — Anxiété",
                "sante_epilepsie"            : "Santé — Épilepsie",
                "sante_respiratoire"         : "Santé — Respiratoire",
                "soutien_intensif"           : "Soutien intensif",
                "mobilite"                   : "Mobilité",
                "alimentation"               : "Alimentation",
            }
            for aspect, hits in profile_chunks.items():
                label = aspect_labels.get(aspect, aspect)
                lines.append(f"\n[{label}]")
                for i, hit in enumerate(hits, 1):
                    c      = hit["chunk"]
                    source = c.get('source_nom') or c.get('source', '?')
                    score  = hit.get('score', 0)
                    lines.append(
                        f"[Source {i} — {source} | score={score:.2f}]\n"
                        f"{c['text'][:200].strip()}\n"
                    )

        return "\n".join(lines)


# TEST

if __name__ == "__main__":

    cf = ChunkFilter(
        chunks_file      = "data/rag/chunk/rag_chunks_meta.json",
        faiss_index_file = "data/rag/vector/faiss_index.bin",
        metadata_file    = "data/rag/vector/metadata.json",
    )

    profil = {
        "prediction"         : "TSA",
        "confidence"         : 0.89,
        "Age_Years"          : 5,
        "Sex"                : "M",
        "PR_QF1A"            : 3,    # non verbal
        "PR_QQ"              : 3,    # soutien modéré
        "PR_QN1_D"           : 1,    # anxiété
        "PR_QO1_A_COMBINE"   : 1,    # agression active
        "PR_QO1_C_COMBINE"   : 1,    # automutilation active
    }

    question = "Comment calmer mon enfant pendant une crise ?"

    print(f"\n{'='*60}")
    print("  TEST — Double Recherche RAG")
    print(f"{'='*60}")
    print(f"  Question : {question}")
    print(f"  Profil   : TSA | 5 ans | non verbal | agression\n")

    results = cf.search(question, profil)

    print(f"  Filtre appliqué  : {results['filter_used']}")
    print(f"  Chunks réponse   : {len(results['response_chunks'])}")
    print(f"  Aspects profil   : {list(results['profile_chunks'].keys())}")
    print(f"\n{'-'*60}")
    print("  PROMPT BLOCK :")
    print(f"{'-'*60}")
    print(results["prompt_block"])
    print(f"{'='*60}")