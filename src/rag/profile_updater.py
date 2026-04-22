import json, logging, re
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_PROFILE_REMARKS = 30

CONTRADICTION_PAIRS = [
    ("non verbal",             "commence à parler"),
    ("non verbal",             "dit quelques mots"),
    ("non verbal",             "parle maintenant"),
    ("parle peu",              "parle beaucoup"),
    ("utilise PECS",           "n'utilise plus PECS"),
    ("crises fréquentes",      "crises réduites"),
    ("crises fréquentes",      "plus de crises"),
    ("agressif",               "moins agressif"),
    ("automutilation",         "plus d'automutilation"),
    ("non scolarisé",          "scolarisé"),
    ("pas autonome",           "plus autonome"),
    ("couches",                "propre"),
    ("troubles du sommeil",    "dort mieux"),
    ("alimentation sélective", "mange mieux"),
]


# =============================================================================
# MAPPING FORMULAIRE PARENT  →  IDs officiels
# =============================================================================
# Format :
#   "QUESTION_ID" : { source, field/keyword, logic }
#
#   source  : "qchat"   → champ Q-Chat-10 (A1..A10)
#             "rm"      → champ DS Survey RM (PR_QH1A, ...)
#             "profile" → mot-clé dans profile_detecter
#   logic   : voir _apply_logic()

FORM_MAPPING: Dict[str, dict] = {

    # ── TWSL — التواصل (Communication) ────────────────────────────────────
    # A1 : répond quand appelé → TWSL_01 + MARF_03
    "TWSL_01" : {"source": "qchat",   "field": "A1",   "logic": "direct"},
    "MARF_03" : {"source": "qchat",   "field": "A1",   "logic": "direct"},
    # A2 : contact visuel → pas de question directe dans éval mais proche MARF_07
    "MARF_07" : {"source": "qchat",   "field": "A2",   "logic": "direct"},
    # A3 : pointe pour demander → TWSL_04
    "TWSL_04" : {"source": "qchat",   "field": "A3",   "logic": "direct"},
    # A6 : suit le regard → MARF_04 (suit d'un son à un autre)
    "MARF_04" : {"source": "qchat",   "field": "A6",   "logic": "direct"},
    # A9 : gestes simples → TWSL_07 (tire les autres par la main)
    "TWSL_07" : {"source": "qchat",   "field": "A9",   "logic": "direct"},
    # PR_QF1A : mode expression (1=langage parlé → TWSL_02=1)
    "TWSL_02" : {"source": "rm",      "field": "PR_QF1A", "logic": "rm_speech"},
    "TWSL_03" : {"source": "rm",      "field": "PR_QF1A", "logic": "rm_speech"},
    # TWSL_05 : semble sourd — inversé par rapport à PR_QI1
    "TWSL_05" : {"source": "rm",      "field": "PR_QI1",  "logic": "rm_aid_inverse"},
    # TWSL_06 : répète les sons → profile
    "TWSL_06" : {"source": "profile", "keyword": "écholalie",           "logic": "absent"},
    # TWSL_08 : bruit bdl parole → profile
    "TWSL_08" : {"source": "profile", "keyword": "non verbal",          "logic": "absent"},

    # ── LABS — اللباس (Habillement) ───────────────────────────────────────
    "LABS_03" : {"source": "profile", "keyword": "s'habille seul",      "logic": "absent"},
    "LABS_05" : {"source": "profile", "keyword": "s'habille seul",      "logic": "absent"},
    "LABS_01" : {"source": "profile", "keyword": "enlève ses habits seul","logic": "absent"},

    # ── AKOL — الأكل (Alimentation) ───────────────────────────────────────
    # PR_QK1 : aide repas (1=jamais → AKOL_04=1)
    "AKOL_04" : {"source": "rm",   "field": "PR_QK1",  "logic": "rm_meal"},
    "AKOL_03" : {"source": "profile", "keyword": "boit seul",           "logic": "absent"},

    # ── HKGE — الحركة العامة (Motricité globale) ──────────────────────────
    # PR_QH1A : aide mobilité (1=jamais → HKGE_01=1)
    "HKGE_01" : {"source": "rm",   "field": "PR_QH1A", "logic": "rm_mobility"},
    # PR_QH1B : fauteuil roulant (1=oui → HKGE_01/02=0)
    "HKGE_02" : {"source": "rm",   "field": "PR_QH1B", "logic": "inverse"},
    "HKGE_03" : {"source": "rm",   "field": "PR_QH1B", "logic": "inverse"},

    # ── HKDQ — الحركة الدقيقة (Motricité fine) ───────────────────────────
    # Pas de champ direct — via profile

    # ── HISS — الادراك الحسي (Sensoriel) ──────────────────────────────────
    # PR_QI1 : prothèses auditives (1=jamais → HISS_03=1)
    "HISS_03"  : {"source": "rm",      "field": "PR_QI1",  "logic": "rm_aid"},
    # PR_QJ1 : aides visuelles (1=jamais → HISS_02=1)
    "HISS_02"  : {"source": "rm",      "field": "PR_QJ1",  "logic": "rm_aid"},
    # Profile → sensorialité
    "HISS_07"  : {"source": "profile", "keyword": "hypersensibilité sonore",   "logic": "present"},
    "HISS_08"  : {"source": "profile", "keyword": "hypersensibilité visuelle", "logic": "present"},
    "HISS_04a" : {"source": "profile", "keyword": "hypersensibilité tactile",  "logic": "present"},
    "HISS_04b" : {"source": "profile", "keyword": "hypersensibilité tactile",  "logic": "present"},
    "HISS_04c" : {"source": "profile", "keyword": "hypersensibilité tactile",  "logic": "present"},
    "HISS_05"  : {"source": "profile", "keyword": "hypersensibilité olfactive","logic": "present"},
    "HISS_01"  : {"source": "profile", "keyword": "distingue couleurs",        "logic": "absent"},

    # ── SLOK — السلوك الاجتماعي (Comportement social) ─────────────────────
    # PR_QO1_A_COMBINE : agression (1=malgré soutien → SLOK_08=1)
    "SLOK_08" : {"source": "rm",      "field": "PR_QO1_A_COMBINE", "logic": "rm_behavior"},
    # PR_QO1_E_COMBINE : fugue → SLOK_02
    "SLOK_02" : {"source": "rm",      "field": "PR_QO1_E_COMBINE", "logic": "rm_behavior"},
    # Profile
    "SLOK_06" : {"source": "profile", "keyword": "joue en groupe",      "logic": "absent"},
    "SLOK_15" : {"source": "profile", "keyword": "partage ses jouets",   "logic": "absent"},

    # ── NZAF — النظافة (Hygiène) ───────────────────────────────────────────
    "NZAF_01" : {"source": "profile", "keyword": "se lave les mains",    "logic": "absent"},
    "NZAF_02" : {"source": "profile", "keyword": "se lave les mains",    "logic": "absent"},
    "NZAF_09" : {"source": "profile", "keyword": "se brosse les dents",  "logic": "absent"},
    "NZAF_07a": {"source": "profile", "keyword": "propre",               "logic": "absent"},
    "NZAF_07b": {"source": "profile", "keyword": "propre",               "logic": "absent"},

    # ── MARF — المعرفة (Cognition) ────────────────────────────────────────
    "MARF_06" : {"source": "profile", "keyword": "reste assis",          "logic": "absent"},
}


# LISTE COMPLÈTE DES IDs OFFICIELS

ALL_QUESTION_IDS = [
    # اللباس
    "LABS_01","LABS_02","LABS_03","LABS_04","LABS_05",
    "LABS_06","LABS_07","LABS_08","LABS_09","LABS_10",
    # الأكل
    "AKOL_01a","AKOL_01b","AKOL_01c","AKOL_01d",
    "AKOL_02a","AKOL_02b","AKOL_02c",
    "AKOL_03","AKOL_04","AKOL_05","AKOL_06",
    # الحركة العامة
    "HKGE_01","HKGE_02","HKGE_03","HKGE_04","HKGE_05",
    "HKGE_06","HKGE_07","HKGE_08","HKGE_09","HKGE_10",
    "HKGE_11","HKGE_12","HKGE_13","HKGE_14","HKGE_15",
    "HKGE_16","HKGE_17","HKGE_18","HKGE_19","HKGE_20",
    "HKGE_21","HKGE_22","HKGE_23","HKGE_24",
    # الحركة الدقيقة
    "HKDQ_01","HKDQ_02","HKDQ_03","HKDQ_04","HKDQ_05",
    "HKDQ_06","HKDQ_07","HKDQ_08","HKDQ_09","HKDQ_10",
    "HKDQ_11","HKDQ_12","HKDQ_13","HKDQ_14","HKDQ_15",
    # التلوين
    "TALW_01a","TALW_01b","TALW_02","TALW_03","TALW_04","TALW_05",
    # الجانبية
    "JANB_01","JANB_02",
    # الصورة الجسمية
    "SJSM_01","SJSM_02","SJSM_03","SJSM_04","SJSM_05","SJSM_06",
    "SJSM_07","SJSM_08","SJSM_09","SJSM_10","SJSM_11","SJSM_12",
    # هيكلة الفضاء
    "FADA_01","FADA_02","FADA_03","FADA_04","FADA_05","FADA_06",
    # تنظيم الزمن
    "ZMAN_01","ZMAN_02","ZMAN_03","ZMAN_04","ZMAN_05","ZMAN_06",
    # الادراك الحسي
    "HISS_01","HISS_02","HISS_03",
    "HISS_04a","HISS_04b","HISS_04c",
    "HISS_05","HISS_06","HISS_07","HISS_08",
    # التواصل
    "TWSL_01","TWSL_02","TWSL_03","TWSL_04",
    "TWSL_05","TWSL_06","TWSL_07","TWSL_08",
    # المعرفة
    "MARF_01","MARF_02","MARF_03","MARF_04","MARF_05","MARF_06","MARF_07",
    "MARF_08a","MARF_08b","MARF_08c","MARF_08d",
    "MARF_09a","MARF_09b","MARF_09c","MARF_09d",
    "MARF_10a","MARF_10b","MARF_10c","MARF_10d",
    "MARF_11a","MARF_11b","MARF_11c","MARF_11d",
    "MARF_12a","MARF_12b","MARF_12c","MARF_12d",
    "MARF_13a","MARF_13b","MARF_13c","MARF_13d","MARF_13e",
    "MARF_14a","MARF_14b","MARF_14c","MARF_14d","MARF_14e",
    "MARF_15","MARF_16",
    "MARF_17a","MARF_17b","MARF_17c",
    # السلوك الاجتماعي
    "SLOK_01","SLOK_02","SLOK_03","SLOK_04","SLOK_05","SLOK_06",
    "SLOK_07","SLOK_08","SLOK_09","SLOK_10","SLOK_11","SLOK_12",
    "SLOK_13","SLOK_14","SLOK_15",
    "SLOK_16a","SLOK_16b","SLOK_16c","SLOK_16d","SLOK_16e",
    "SLOK_16f","SLOK_16g","SLOK_16h",
    # النظافة
    "NZAF_01","NZAF_02","NZAF_03","NZAF_04","NZAF_05","NZAF_06",
    "NZAF_07a","NZAF_07b",
    "NZAF_08a","NZAF_08b","NZAF_08c",
    "NZAF_09","NZAF_10",
]


# LOGIQUE DE CONVERSION

def _apply_logic(logic: str, form_value=None,
                 keyword: str = None, profile: list = None) -> Optional[int]:

    if logic == "direct":
        if form_value in (1, "1", True):  return 1
        if form_value in (0, "0", False): return 0
        return None

    elif logic == "inverse":
        if form_value in (1, "1", True):  return 0
        if form_value in (0, "0", False): return 1
        return None

    elif logic == "present":
        # Mot-clé détecté dans profil → problème → réponse 0
        if profile and keyword:
            found = any(keyword.lower() in p.lower() for p in profile)
            return 0 if found else None
        return None

    elif logic == "absent":
        # Mot-clé détecté → compétence acquise → réponse 1
        if profile and keyword:
            found = any(keyword.lower() in p.lower() for p in profile)
            return 1 if found else None
        return None

    elif logic == "rm_mobility":
        # PR_QH1A : 1=jamais → 1 / 4-5=fréquemment → 0
        if isinstance(form_value, (int, float)):
            return 1 if form_value == 1 else (0 if form_value >= 4 else None)
        return None

    elif logic == "rm_meal":
        # PR_QK1 : 1=jamais aide → mange seul → 1
        if isinstance(form_value, (int, float)):
            return 1 if form_value == 1 else (0 if form_value >= 4 else None)
        return None

    elif logic == "rm_aid":
        # PR_QI1/QJ1 : 1=jamais → pas de problème → 1
        if isinstance(form_value, (int, float)):
            return 1 if form_value == 1 else (0 if form_value >= 3 else None)
        return None

    elif logic == "rm_aid_inverse":
        # TWSL_05 : utilise prothèse auditive → semble sourd=0 (peut entendre)
        if isinstance(form_value, (int, float)):
            return 0 if form_value == 1 else (1 if form_value >= 3 else None)
        return None

    elif logic == "rm_speech":
        # PR_QF1A : 1=langage parlé → 1 / autre=0
        if isinstance(form_value, (int, float)):
            return 1 if form_value == 1 else 0
        return None

    elif logic == "rm_behavior":
        # PR_QO1_x : 1=oui malgré soutien → problème=1 / 3=non → 0
        if isinstance(form_value, (int, float)):
            return 1 if form_value == 1 else (0 if form_value == 3 else None)
        return None

    return None


# PROFILE UPDATER

class ProfileUpdater:

    def __init__(self, llm_client=None):
        self.llm = llm_client

    # 1. MISE À JOUR PROFIL après conversation

    def update(self,
               profile_detecter: list,
               new_question    : str,
               new_answer      : str) -> dict:
        """
        Analyse la conversation, met à jour profile_detecter,
        et génère le JSON d'évaluation mis à jour.
        """
        current = list(profile_detecter)
        new_remarks = []

        if self.llm:
            try:
                new_remarks = self._extract_with_llm(current, new_question, new_answer)
            except Exception as e:
                logger.warning(f"LLM extraction échouée : {e} — fallback règles")

        if not new_remarks:
            new_remarks = self._extract_with_rules(new_question + " " + new_answer)

        added, replaced = [], []

        for remark in new_remarks:
            remark = remark.strip()
            if not remark or self._is_duplicate(current, remark):
                continue
            to_replace = self._find_contradiction(current, remark)
            if to_replace:
                current[current.index(to_replace)] = remark
                replaced.append({"old": to_replace, "new": remark})
            else:
                current.append(remark)
                added.append(remark)

        if len(current) > MAX_PROFILE_REMARKS:
            current = current[-MAX_PROFILE_REMARKS:]

        eval_json = self.generate_eval_from_profile(profile_detecter=current)

        return {
            "profile_detecter": current,
            "eval_json"        : eval_json,
            "updated"          : bool(added or replaced),
            "changes"          : {"added": added, "replaced": replaced},
        }

    # 2. ÉVALUATION INITIALE depuis formulaire parent

    def generate_eval_from_form(self,
                                 child_form      : dict,
                                 profile_detecter: list = None) -> dict:
        """
        Génère { "TWSL_01": 1, "LABS_01": 0, ... } depuis le formulaire parent.

        Args:
            child_form       : { "A1": 1, ..., "PR_QH1A": 2, ... }
            profile_detecter : remarques IA optionnelles

        Returns:
            dict { questionId: 0|1|None }
            None = à remplir manuellement par l'admin
        """
        profile = profile_detecter or []
        result  = {}

        for qid in ALL_QUESTION_IDS:
            mapping = FORM_MAPPING.get(qid)

            if mapping is None:
                result[qid] = None
                continue

            source  = mapping["source"]
            logic   = mapping["logic"]
            keyword = mapping.get("keyword")

            if source in ("qchat", "rm"):
                field    = mapping["field"]
                val      = child_form.get(field)
                result[qid] = _apply_logic(logic, int(val)) if val is not None else None

            elif source == "profile":
                result[qid] = _apply_logic(logic, keyword=keyword, profile=profile)

            else:
                result[qid] = None

        filled = sum(1 for v in result.values() if v is not None)
        logger.info(f"generate_eval_from_form : {filled}/{len(ALL_QUESTION_IDS)} questions pré-remplies")
        return result

    # 3. ÉVALUATION depuis profileDetected seul

    def generate_eval_from_profile(self, profile_detecter: list) -> dict:
        """
        Génère { questionId: 0|1|None } uniquement depuis les remarques profil.
        """
        result = {}
        for qid in ALL_QUESTION_IDS:
            mapping = FORM_MAPPING.get(qid)
            if mapping and mapping["source"] == "profile":
                result[qid] = _apply_logic(
                    mapping["logic"],
                    keyword=mapping.get("keyword"),
                    profile=profile_detecter,
                )
            else:
                result[qid] = None
        return result

    # MÉTHODES PRIVÉES

    def _extract_with_llm(self, current, question, answer):
        prompt = f"""
Tu analyses une conversation entre un parent et un assistant pour enfants TSA/RM.
Extrais UNIQUEMENT les nouvelles informations factuelles sur l'enfant.

Profil actuel : {json.dumps(current, ensure_ascii=False)}
Question parent : {question}
Réponse assistant : {answer}

Retourne un JSON array de courtes remarques (max 8 mots, en français).
Si rien de nouveau : retourne [].
Réponds UNIQUEMENT avec le JSON.
"""
        text = self.llm.generate(prompt, max_tokens=200).strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)

    def _extract_with_rules(self, text: str) -> list:
        text_lower = text.lower()
        SIGNALS = {
            "hypersensibilité sonore"     : ["bruit", "son fort", "oreilles", "couvre les oreilles"],
            "hypersensibilité visuelle"   : ["lumière forte", "lumière vive", "plisse les yeux"],
            "hypersensibilité tactile"    : ["toucher", "textures", "refuse de toucher"],
            "hypersensibilité olfactive"  : ["odeurs", "refuse odeur"],
            "troubles du sommeil"         : ["dort pas", "se réveille", "insomnie"],
            "alimentation sélective"      : ["mange pas", "refuse aliments", "sélectif"],
            "crises fréquentes"           : ["crise", "colère", "agitation"],
            "automutilation"              : ["se blesse", "se tape", "se mord"],
            "non verbal"                  : ["ne parle pas", "pas de parole", "muet", "non verbal"],
            "écholalie"                   : ["répète", "écholalie", "imite les sons"],
            "commence à parler"           : ["commence à dire", "dit quelques mots", "premiers mots"],
            "s'habille seul"              : ["s'habille seul", "met ses habits"],
            "enlève ses habits seul"      : ["enlève ses habits"],
            "se lave les mains"           : ["lave les mains"],
            "se brosse les dents"         : ["brosse dents"],
            "propre"                      : ["propre", "toilettes seul", "couche"],
            "boit seul"                   : ["boit seul", "boit dans un verre"],
            "mange seul"                  : ["mange seul", "autonome repas"],
            "joue en groupe"              : ["joue avec", "jeu collectif"],
            "partage ses jouets"          : ["partage", "donne ses jouets"],
            "reste assis"                 : ["reste assis", "s'assoit bien"],
            "distingue couleurs"          : ["distingue les couleurs", "reconnaît les couleurs"],
        }
        return [rk for rk, kws in SIGNALS.items() if any(kw in text_lower for kw in kws)]

    def _find_contradiction(self, current, new_remark):
        new_lower = new_remark.lower()
        for old_pat, new_pat in CONTRADICTION_PAIRS:
            if new_pat in new_lower:
                for ex in current:
                    if old_pat in ex.lower():
                        return ex
        new_w = set(new_lower.split()[:4])
        for ex in current:
            ex_w = set(ex.lower().split()[:4])
            if len(new_w & ex_w) >= 3 and ex != new_remark:
                return ex
        return None

    def _is_duplicate(self, current, new_remark):
        nl = new_remark.lower().strip()
        for ex in current:
            if ex.lower().strip() == nl:
                return True
            nw = set(nl.split())
            ew = set(ex.lower().split())
            if nw and len(nw & ew) / len(nw) >= 0.8:
                return True
        return False


# TEST RAPIDE

if __name__ == "__main__":
    pu = ProfileUpdater()

    print("\n" + "="*60)
    print("  TEST 1 — Évaluation depuis formulaire parent")
    print("="*60)

    form = {
        "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
        "A6": 0, "A7": 0, "A8": 1, "A9": 0, "A10": 1,
        "PR_QH1A": 1, "PR_QH1B": 2,
        "PR_QK1": 1,
        "PR_QF1A": 2,
        "PR_QI1": 1, "PR_QJ1": 1,
        "PR_QO1_A_COMBINE": 2,
        "PR_QO1_E_COMBINE": 3,
    }
    profile = ["hypersensibilité sonore", "hypersensibilité tactile", "non verbal"]

    result = pu.generate_eval_from_form(form, profile)
    filled = {k: v for k, v in result.items() if v is not None}
    print(f"\n  {len(filled)}/{len(result)} questions pré-remplies")
    print("\n  Extrait (15 premières) :")
    for qid, val in list(filled.items())[:15]:
        print(f"    {qid:<14} → {val}")
    print("    ...")

    print("\n" + "="*60)
    print("  TEST 2 — Mise à jour profil après conversation")
    print("="*60)

    upd = pu.update(
        profile_detecter=["non verbal", "crises fréquentes"],
        new_question="Il commence à dire quelques mots depuis 2 semaines",
        new_answer="Très bonne évolution, continuez les pictogrammes...",
    )
    print(f"\n  Profil mis à jour : {upd['profile_detecter']}")
    print(f"  Ajouts           : {upd['changes']['added']}")
    print(f"  Remplacements    : {upd['changes']['replaced']}")
    profile_filled = {k: v for k, v in upd['eval_json'].items() if v is not None}
    print(f"  Questions profil : {len(profile_filled)} mises à jour")
    print("="*60)