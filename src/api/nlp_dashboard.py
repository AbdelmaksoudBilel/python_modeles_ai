"""
src/api/nlp_dashboard.py — VERSION CORRIGÉE
================================================================================
BUG CORRIGÉ — Topics NLP toujours le même sujet (~9 messages)

Causes identifiées :
  1. n_clusters=5 avec 9 messages → KMeans avec n_samples < n_clusters
     sklearn ajuste mais la qualité est très mauvaise
  2. Tous les messages parlent du même sujet (crises TSA) → 1 vrai cluster
  3. min_df=2 dans TfidfVectorizer élimine les termes rares avec peu de messages
     → vocabulaire très réduit → tous dans le même cluster

Fixes :
  (a) Auto-ajustement de n_clusters : min(n_clusters, max(2, n_samples // 3))
      → avec 9 messages : n_clusters = min(5, 3) = 3
  (b) min_df abaissé à 1 quand peu de messages
  (c) Meilleur label de topic : mot le plus fréquent du cluster
  (d) Avertissement clair si trop peu de messages pour le clustering
================================================================================
"""

import re, logging
from collections import Counter
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

try:
    import spacy
    nlp_fr = spacy.load("fr_core_news_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    logger.warning("spaCy fr non disponible → lemmatisation désactivée")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from lingua import Language, LanguageDetectorBuilder
    _detector = LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ARABIC, Language.ENGLISH
    ).build()
    LINGUA_OK = True
except ImportError:
    LINGUA_OK = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False


STOPWORDS_FR = set([
    "mon","ma","mes","le","la","les","de","du","un","une","des","ce","cet",
    "cette","ces","je","il","elle","nous","vous","ils","elles","on","me","te",
    "se","lui","leur","y","en","est","sont","a","ont","être","avoir","faire",
    "que","qui","quoi","comment","pourquoi","quand","où","quel","quelle",
    "et","ou","mais","donc","or","ni","car","pour","par","sur","sous","dans",
    "avec","sans","entre","vers","son","sa","ses","leurs","plus","très","bien",
    "aussi","tout","tous","toute","toutes","pas","ne","si","même","autre",
    "alors","puis","déjà","encore","toujours","jamais","souvent","parfois",
    "veux","peut","dois","faut","va","vais","fait","dit","voir","savoir",
    "enfant","fils","fille","ça","là","ici","comme","quand","après","avant",
    "pendant","depuis","jusqu","bonjour","merci","aide","besoin",
])
STOPWORDS_AR = set(["من","إلى","عن","على","في","مع","هو","هي","هم","أنا","أنت","نحن","هذا","هذه","التي","الذي","ما","لا","قد","كان","أن","إن","لم","كل","بعد","قبل","عند","لكن","أو","و","ولكن","إذا","لأن","حتى","ثم"])
STOPWORDS_EN = set(["the","a","an","in","on","at","to","for","of","and","or","but","is","are","was","were","be","have","has","had","do","does","did","my","your","his","her","our","their","it","this","that","i","you","he","she","we","they","not","no","yes","can","will"])
ALL_STOPWORDS = STOPWORDS_FR | STOPWORDS_AR | STOPWORDS_EN

POSITIVE_WORDS = set(["progrès","amélioration","mieux","bien","calme","heureux","sourire","réussit","avance","évolution","positif","excellent","super","merci","content","satisfait","efficace","fonctionne","marche"])
NEGATIVE_WORDS = set(["crise","agressif","violence","pleure","cri","difficile","problème","impossible","peur","stress","anxieux","fatigue","épuisé","découragé","échec","refuse","blesse","frappe","mord","détruit","fugue","danger"])


class NLPRequest(BaseModel):
    messages       : List[str]
    n_keywords     : int = 20
    n_questions    : int = 10
    n_clusters     : int = 5
    min_word_length: int = 4


class NLPPipeline:

    def __init__(self, min_len: int = 4):
        self.min_len = min_len

    def translate_to_fr(self, text: str, lang: str) -> str:
        if lang == "fr" or not TRANSLATOR_OK: return text
        try:
            return GoogleTranslator(source=lang, target="fr").translate(text) or text
        except: return text

    def clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def detect_lang(self, text: str) -> str:
        if not LINGUA_OK: return "fr"
        try:
            lang = _detector.detect_language_of(text)
            if lang is None: return "fr"
            return {"FRENCH":"fr","ARABIC":"ar","ENGLISH":"en"}.get(lang.name, "fr")
        except: return "fr"

    def tokenize(self, text: str) -> List[str]:
        return [w for w in text.split() if len(w) >= self.min_len and w not in ALL_STOPWORDS]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if not SPACY_OK: return tokens
        try:
            doc = nlp_fr(" ".join(tokens))
            return [t.lemma_ for t in doc if t.lemma_ not in ALL_STOPWORDS and len(t.lemma_) >= self.min_len and not t.is_punct]
        except: return tokens

    def extract_questions(self, text: str) -> List[str]:
        patterns = [
            r"[Cc]omment\s+.{5,80}[?؟]?", r"[Pp]ourquoi\s+.{5,80}[?؟]?",
            r"[Qq]u[ée]\s+.{5,80}[?؟]?", r"[Ee]st[-\s]ce\s+.{5,80}[?؟]?",
            r"[Qq]uand\s+.{5,80}[?؟]?", r".{10,80}[?؟]",
        ]
        questions = []
        for pattern in patterns:
            for q in re.findall(pattern, text):
                if len(q.strip()) > 10:
                    questions.append(q.strip())
        return questions

    def sentiment(self, text: str) -> str:
        words   = set(text.lower().split())
        pos_cnt = len(words & POSITIVE_WORDS)
        neg_cnt = len(words & NEGATIVE_WORDS)
        if neg_cnt > pos_cnt:   return "negative"
        elif pos_cnt > neg_cnt: return "positive"
        else:                   return "neutral"

    def cluster_topics(self, texts: List[str], n_clusters: int = 5) -> List[dict]:
        """
        FIX : Auto-ajustement de n_clusters selon le nombre de textes.
        Règle : n_clusters = min(n_clusters_demandé, max(2, n_samples // 3))

        Exemples :
          9 messages   → n_clusters = min(5, 3) = 3
          30 messages  → n_clusters = min(5, 10) = 5
          6 messages   → n_clusters = min(5, 2) = 2
        """
        if not SKLEARN_OK or len(texts) < 2:
            return []

        # FIX : ajustement automatique n_clusters
        n_samples         = len(texts)
        n_clusters_auto   = min(n_clusters, max(2, n_samples // 3))

        if n_clusters_auto < 2:
            logger.info(f"Trop peu de messages ({n_samples}) pour le clustering → ignoré")
            return []

        if n_clusters_auto < n_clusters:
            logger.info(
                f"n_clusters ajusté : {n_clusters} → {n_clusters_auto} "
                f"(n_samples={n_samples})"
            )

        # FIX : min_df=1 quand peu de messages (évite vocabulaire vide)
        min_df = 2 if n_samples >= 20 else 1

        try:
            vectorizer = TfidfVectorizer(
                max_features = min(200, n_samples * 5),
                ngram_range  = (1, 2),
                stop_words   = list(ALL_STOPWORDS),
                min_df       = min_df,
            )
            X = vectorizer.fit_transform(texts)

            if X.shape[1] == 0:
                logger.warning("Vectoriseur TF-IDF vide → pas de clustering")
                return []

            km = KMeans(
                n_clusters = n_clusters_auto,
                random_state = 42,
                n_init = 10,
                max_iter = 100,
            )
            km.fit(X)

            feature_names = vectorizer.get_feature_names_out()
            clusters = []

            for i in range(n_clusters_auto):
                center   = km.cluster_centers_[i]
                top_idx  = center.argsort()[-8:][::-1]
                keywords = [feature_names[j] for j in top_idx if center[j] > 0]
                count    = int((km.labels_ == i).sum())

                # FIX : meilleur label = mot le plus fréquent du cluster
                cluster_texts = " ".join([texts[j] for j, l in enumerate(km.labels_) if l == i])
                word_freq     = Counter(cluster_texts.split())
                top_word      = max(
                    (w for w in word_freq if len(w) >= 4 and w not in ALL_STOPWORDS),
                    key=word_freq.get, default=keywords[0] if keywords else f"Sujet {i+1}"
                )

                clusters.append({
                    "topic"   : f"Sujet {i+1}",
                    "keywords": keywords[:5],
                    "count"   : count,
                    "label"   : top_word,
                })

            return sorted(clusters, key=lambda x: x["count"], reverse=True)

        except Exception as e:
            logger.warning(f"Clustering échoué : {e}")
            return []


@router.post("/dashboard/nlp")
async def nlp_analysis(req: NLPRequest):
    """
    Analyse NLP — avec gestion des petits volumes de messages.
    """
    nlp = NLPPipeline(min_len=req.min_word_length)

    all_tokens    = []
    all_questions = []
    sentiments    = {"positive": 0, "neutral": 0, "negative": 0}
    lang_dist     = {"fr": 0, "ar": 0, "en": 0}
    total_length  = 0
    cleaned_texts = []

    for raw in req.messages:
        if not raw or not raw.strip():
            continue

        lang    = nlp.detect_lang(raw)
        lang_dist[lang] = lang_dist.get(lang, 0) + 1

        text_fr  = nlp.translate_to_fr(raw, lang)
        cleaned  = nlp.clean(text_fr)
        cleaned_texts.append(cleaned)
        total_length += len(cleaned.split())

        tokens = nlp.tokenize(cleaned)
        tokens = nlp.lemmatize(tokens)
        all_tokens.extend(tokens)

        all_questions.extend(nlp.extract_questions(text_fr))

        sent = nlp.sentiment(cleaned)
        sentiments[sent] += 1

    n_messages = len(cleaned_texts)

    word_freq    = Counter(all_tokens)
    top_keywords = [
        {"word": w, "count": c, "freq": round(c / max(len(all_tokens), 1) * 100, 2)}
        for w, c in word_freq.most_common(req.n_keywords)
    ]

    q_counter    = Counter([q.lower().strip().rstrip("?؟").strip() for q in all_questions])
    top_questions = [
        {"question": q, "count": c}
        for q, c in q_counter.most_common(req.n_questions)
        if len(q) > 10
    ]

    # FIX : clustering avec auto-ajustement n_clusters
    topic_clusters = nlp.cluster_topics(cleaned_texts, req.n_clusters)

    word_cloud = [{"text": w, "value": c} for w, c in word_freq.most_common(50)]

    total_msgs = max(sum(sentiments.values()), 1)
    sentiment_scores = {
        k: {"count": v, "percent": round(v / total_msgs * 100, 1)}
        for k, v in sentiments.items()
    }

    # Message d'avertissement si peu de données
    warning = None
    if n_messages < 10:
        warning = (
            f"Seulement {n_messages} messages analysés. "
            f"Le clustering et l'analyse de sentiment sont moins fiables "
            f"avec peu de données. Recommandation : ≥ 30 messages."
        )
    elif n_messages < 30:
        warning = f"{n_messages} messages analysés. Pour de meilleurs résultats, ≥ 30 messages recommandés."

    logger.info(
        f"NLP ✔ | messages={n_messages} | tokens={len(all_tokens)} | "
        f"questions={len(all_questions)} | clusters={len(topic_clusters)}"
    )

    return {
        "top_keywords"           : top_keywords,
        "top_questions"          : top_questions,
        "topic_clusters"         : topic_clusters,
        "sentiment"              : sentiment_scores,
        "word_cloud_data"        : word_cloud,
        "avg_msg_length"         : round(total_length / max(n_messages, 1), 1),
        "lang_distribution"      : {
            k: {"count": v, "percent": round(v / max(n_messages, 1) * 100, 1)}
            for k, v in lang_dist.items()
        },
        "total_messages_analyzed": n_messages,
        # FIX : infos de clustering pour le frontend
        "clustering_info"        : {
            "n_clusters_requested": req.n_clusters,
            "n_clusters_used"     : len(topic_clusters),
            "adjusted"            : len(topic_clusters) < req.n_clusters,
            "warning"             : warning,
        },
    }