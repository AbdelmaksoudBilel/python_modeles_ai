import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    DDG_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDG_OK = True
    except ImportError:
        DDG_OK = False
        logger.warning("ddgs non installé → pip install ddgs")


# DOMAINES FIABLES PAR TROUBLE

TRUSTED_DOMAINS = {
    "TSA": [
        "has-sante.fr",
        "autisme.gouv.fr",
        "autismesociete.org",
        "autismeinfos.fr",
        "maison-autisme.fr",
        "autisme-france.fr",
        "pubmed.ncbi.nlm.nih.gov",
        "cairn.info",
        "orpha.net",
        "handicap.fr",
        "santepubliquefrance.fr",
        "ameli.fr",
        "passeportsante.net",
        "vidal.fr",
    ],
    "RM": [
        "unapei.org",
        "who.int",
        "has-sante.fr",
        "apei.fr",
        "pubmed.ncbi.nlm.nih.gov",
        "cairn.info",
        "orpha.net",
        "handicap.fr",
        "santepubliquefrance.fr",
        "ameli.fr",
        "passeportsante.net",
        "afpep.fr",
    ],
    "MIXTE": [
        "has-sante.fr",
        "autisme.gouv.fr",
        "autismesociete.org",
        "autisme-france.fr",
        "unapei.org",
        "who.int",
        "pubmed.ncbi.nlm.nih.gov",
        "cairn.info",
        "orpha.net",
        "handicap.fr",
        "santepubliquefrance.fr",
        "ameli.fr",
        "passeportsante.net",
    ],
}

# Nombre max de résultats web
MAX_RESULTS     = 5
# Nombre max de tentatives par domaine
MAX_PER_DOMAIN  = 2


# CLASSE PRINCIPALE

class WebSearch:

    def __init__(self):
        if not DDG_OK:
            raise ImportError("pip install ddgs")
        self.ddgs = DDGS()
        logger.info("WebSearch initialisé ✔")

    # ── VÉRIFICATION DOMAINE ──────────────────────────────────────────────────

    def _is_trusted(self, url: str, trusted: list) -> bool:
        """Vérifie si une URL appartient aux domaines fiables."""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace("www.", "")
            return any(domain == d or domain.endswith("." + d)
                       for d in trusted)
        except Exception:
            return False

    # ── RECHERCHE ─────────────────────────────────────────────────────────────

    def search(self, query: str, trouble: str = "TSA",
               max_results: int = MAX_RESULTS) -> list:
        """
        Recherche DuckDuckGo filtrée sur les domaines fiables.

        Args:
            query      : question du parent (en français)
            trouble    : "TSA" | "RM" | "MIXTE"
            max_results: nombre max de résultats

        Returns:
            Liste de dicts :
            [
                {
                    "title"  : "Titre de la page",
                    "url"    : "https://...",
                    "snippet": "Extrait pertinent...",
                    "source" : "has-sante.fr",
                    "domain" : "has-sante.fr",
                },
                ...
            ]
        """
        trouble_up = trouble.upper()
        trusted    = TRUSTED_DOMAINS.get(trouble_up, TRUSTED_DOMAINS["TSA"])

        # Enrichir la requête avec le contexte trouble
        context = "autisme TSA" if trouble_up == "TSA" else \
                  "déficience intellectuelle" if trouble_up == "RM" else \
                  "autisme déficience intellectuelle"
        enriched_query = f"{query} {context} recommandations parents"

        logger.info(f"Recherche web | query='{enriched_query[:60]}...'")

        results   = []
        all_raw   = []   # garder tous les résultats pour fallback
        seen_urls = set()

        try:
            raw = self.ddgs.text(
                enriched_query,
                max_results=max_results * 10,
                region="fr-fr",
            )

            for r in raw:
                url = r.get("href", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                domain = urlparse(url).netloc.lower().replace("www.", "")

                entry = {
                    "title"  : r.get("title", ""),
                    "url"    : url,
                    "snippet": r.get("body", "")[:500],
                    "source" : domain,
                    "domain" : domain,
                }

                # Stocker tous les résultats pour fallback
                all_raw.append(entry)

                # Filtrer par domaine fiable
                if self._is_trusted(url, trusted):
                    results.append(entry)

                if len(results) >= max_results:
                    break

        except Exception as e:
            logger.error(f"Erreur DuckDuckGo : {e}")

        # ── Fallback : si 0 résultats fiables → prendre les premiers ─────
        if not results and all_raw:
            logger.warning(
                "0 résultats sur domaines fiables → fallback résultats généraux"
            )
            # Exclure uniquement les domaines clairement non fiables
            EXCLUDE = {"reddit.com", "twitter.com", "facebook.com",
                       "tiktok.com", "youtube.com", "pinterest.com"}
            for entry in all_raw:
                if entry["domain"] not in EXCLUDE:
                    results.append(entry)
                if len(results) >= max_results:
                    break

        logger.info(f"Web search ✔ → {len(results)} résultats")
        return results

    # ── RECHERCHE PAR DOMAINE FORCÉ ───────────────────────────────────────────

    def search_domain(self, query: str, domain: str,
                      max_results: int = MAX_PER_DOMAIN) -> list:
        """
        Recherche forcée sur un domaine spécifique.
        Utilise l'opérateur site: de DuckDuckGo.

        Args:
            query  : requête
            domain : domaine cible (ex: "has-sante.fr")
        """
        site_query = f"site:{domain} {query}"
        logger.info(f"Recherche domaine | {domain}")

        results = []
        try:
            raw = self.ddgs.text(site_query, max_results=max_results * 3,
                                 region="fr-fr")
            for r in raw:
                url = r.get("href", "")
                if domain not in url:
                    continue
                results.append({
                    "title"  : r.get("title", ""),
                    "url"    : url,
                    "snippet": r.get("body", "")[:500],
                    "source" : domain,
                    "domain" : domain,
                })
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"Erreur recherche domaine {domain} : {e}")

        return results

    # ── FORMAT PROMPT ─────────────────────────────────────────────────────────

    def format_for_prompt(self, results: list) -> str:
        """
        Formate les résultats web pour injection dans le prompt LLM.

        Returns:
            Bloc texte formaté
        """
        if not results:
            return ""

        lines = ["=== Sources web (complément) ==="]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] ({r['source']}) {r['title']}\n"
                f"    {r['snippet'][:300]}..."
            )
        return "\n".join(lines)


# TEST

if __name__ == "__main__":

    ws = WebSearch()

    tests = [
        ("Comment calmer un enfant autiste en crise ?", "TSA"),
        ("Exercices autonomie enfant déficience intellectuelle", "RM"),
    ]

    for query, trouble in tests:
        print(f"\n{'='*60}")
        print(f"  Query   : {query}")
        print(f"  Trouble : {trouble}")
        print(f"{'='*60}")

        results = ws.search(query, trouble)

        if results:
            for r in results:
                print(f"\n  ✔ [{r['source']}]")
                print(f"    {r['title']}")
                print(f"    {r['snippet'][:150]}...")
        else:
            print("  ✘ Aucun résultat fiable trouvé")

        print(f"\n  Bloc prompt :")
        print(ws.format_for_prompt(results))