import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
import requests
from urllib.parse import urlencode
import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer

from symptom_utils import normalize_and_split_symptoms
from symptom_mapper import canonicalize_list

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARTICLE_PAGE_SIZE = 5
RAG_DIR = Path("rag_index")
MODEL_PATH = "model/model-best"
LABEL = "SYMPTOM"
TOP_K = 5

W_OVERLAP = 0.70
W_MISS = 0.20
W_EMB = 0.30

CAND_MULT = 12
CAND_MIN = 80


OFFLINE_DISEASES_PATH = Path("diseases.json")


def load_offline_disease_json():
    if not OFFLINE_DISEASES_PATH.exists():
        return []

    with open(OFFLINE_DISEASES_PATH, "r", encoding="utf8") as f:
        return json.load(f)


def normalize_name_for_match(name: str) -> str:
    name = str(name or "").lower().strip()
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_offline_disease_info(disease_name: str) -> Dict[str, Any]:
    diseases = load_offline_disease_json()

    if not diseases:
        return {}

    target = normalize_name_for_match(disease_name)

    for item in diseases:
        current = normalize_name_for_match(item.get("disease_name", ""))
        if current == target:
            return item

    for item in diseases:
        current = normalize_name_for_match(item.get("disease_name", ""))
        if target in current or current in target:
            return item

    return {}


def build_offline_summary(disease_name: str) -> Dict[str, Any]:
    info = get_offline_disease_info(disease_name)

    if not info:
        return {
            "overview": [f"No offline information found for {disease_name}."],
            "causes": [],
            "treatment": [],
            "prevention": [],
            "when_to_see_a_doctor": [],
            "sources": []
        }

    return {
        "overview": [info.get("Overview", "N/A")],
        "causes": [info.get("Causes", "N/A")],
        "treatment": [info.get("Treatment", "N/A")],
        "prevention": [info.get("Prevention", "N/A")],
        "when_to_see_a_doctor": [info.get("When to see a doctor", "N/A")],
        "sources": [
            {
                "title": info.get("Sources", "N/A"),
                "journal": "Offline JSON medical knowledge base",
                "year": "",
                "pmid": "",
                "doi": ""
            }
        ]
    }

def clean_piece(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?:;\"'()[]{}")


def load_rag() -> Tuple[Any, List[Dict[str, Any]], SentenceTransformer, Dict[str, float]]:
    index_path = RAG_DIR / "index.faiss"
    meta_path = RAG_DIR / "meta.json"
    model_name_path = RAG_DIR / "model_name.txt"
    idf_path = RAG_DIR / "idf.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}. Run build_index.py.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run build_index.py.")
    if not model_name_path.exists():
        raise FileNotFoundError(f"Missing {model_name_path}. Run build_index.py.")
    if not idf_path.exists():
        raise FileNotFoundError(f"Missing {idf_path}. Run build_index.py (with IDF).")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    embed_model_name = model_name_path.read_text(encoding="utf8").strip()
    embedder = SentenceTransformer(embed_model_name)
    idf = json.loads(idf_path.read_text(encoding="utf8"))
    return index, meta, embedder, idf


def extract_symptoms_ner(nlp, text: str) -> List[str]:
    doc = nlp(text)
    raw = [ent.text for ent in doc.ents if ent.label_ == LABEL]
    raw = [clean_piece(x) for x in raw if x]

    final = normalize_and_split_symptoms(raw)
    final = [clean_piece(x) for x in final if x]

    final = [x for x in final if x]
    final = list(dict.fromkeys(final))
    return final


def retrieve(index, meta, embedder, idf, query_symptoms: List[str], top_k: int = TOP_K):
    if not query_symptoms:
        return []

    queries = ["Symptoms: " + ", ".join(query_symptoms)]
    queries += [f"Symptom: {s}" for s in query_symptoms]

    q_emb = embedder.encode(queries, normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    cand_n = max(top_k * CAND_MULT, CAND_MIN)
    scores, ids = index.search(q_emb, cand_n)

    best_max, best_sum, best_cnt = {}, {}, {}
    for row_scores, row_ids in zip(scores, ids):
        for s, i in zip(row_scores.tolist(), row_ids.tolist()):
            if i == -1:
                continue
            s = float(s)
            best_max[i] = max(best_max.get(i, -1e9), s)
            best_sum[i] = best_sum.get(i, 0.0) + s
            best_cnt[i] = best_cnt.get(i, 0) + 1

    if not best_max:
        return []

    pooled = {}
    for i in best_max.keys():
        mean_s = best_sum[i] / max(best_cnt[i], 1)
        pooled[i] = 0.85 * best_max[i] + 0.15 * mean_s

    cand_scores = np.array(list(pooled.values()), dtype=np.float32)
    s_min = float(cand_scores.min())
    s_max = float(cand_scores.max())
    denom = (s_max - s_min) if (s_max - s_min) > 1e-6 else 1.0

    def norm_emb(x: float) -> float:
        return (x - s_min) / denom

    qs = list(dict.fromkeys(query_symptoms))
    qs_set = set(qs)

    qs_in_vocab = {s for s in qs_set if s in idf}

    use_overlap = len(qs_in_vocab) > 0

    # IDF-weighted denominator for recall calculation
    den = sum(float(idf.get(s, 1.0)) for s in qs_in_vocab) if use_overlap else 1.0
    if den <= 0:
        den = 1.0

    results = []
    for idx, raw_emb in pooled.items():
        disease = meta[idx]
        ds = set(disease.get("symptoms_canon", disease.get("symptoms", [])))

        # FIX 1: always initialise overlap/missing before use
        overlap = set()
        missing = set()
        ov = 0.0
        miss = 0.0

        if use_overlap:
            overlap = qs_in_vocab.intersection(ds)
            missing = qs_in_vocab.difference(ds)

            ov_num = sum(float(idf.get(s, 1.0)) for s in overlap)
            miss_num = sum(float(idf.get(s, 1.0)) for s in missing)

            # recall: how many of the query symptoms does this disease cover
            ov = max(0.0, min(1.0, ov_num / den))
            miss = max(0.0, min(1.0, miss_num / den))

        emb01 = float(norm_emb(raw_emb))

        # FIX 2: specificity_boost is now safe (overlap always defined above)
        # Penalise diseases whose profile is much larger than the overlap
        # e.g. atelectasis has 30 symptoms but only 2 match → precision = 2/30
        # strep throat has 8 symptoms and 2 match → precision = 2/8 (much better)
        disease_profile_size = max(len(ds), 1)
        precision = len(overlap) / disease_profile_size
        recall = len(overlap) / max(len(qs_in_vocab), 1)

        # F1-like harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Tie-breaker: when multiple diseases have identical symptom overlap,
        # prefer diseases with smaller profiles (more specific / focused diseases).
        # e.g. gastroenteritis (8 symptoms) beats ulcerative colitis (25 symptoms)
        # when both match the same 4 query symptoms.
        # We add a small inverse-size bonus so it only matters during ties.
        specificity_tiebreak = 1.0 / (1.0 + disease_profile_size)

        # FIX 3: rebalanced formula
        # - F1 is the primary signal (precision × recall): catches the atelectasis problem
        # - ov (IDF-weighted recall) is secondary: rewards rare-symptom matches
        # - embedding is tertiary and capped: prevents semantic drift from dominating
        # - miss penalises diseases that require symptoms the user doesn't have
        # - specificity_tiebreak breaks ties between equally-matching diseases
        final = (
            (0.50 * f1)
            + (0.30 * ov)
            - (0.25 * miss)
            + (0.10 * emb01)
            + (0.05 * specificity_tiebreak)
        )
        results.append((final, float(raw_emb), float(ov), float(miss), disease))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def explain(query_symptoms: List[str], disease_meta: dict) -> dict:
    disease_symptoms = set(disease_meta.get("symptoms_canon", disease_meta.get("symptoms", [])))
    qs = set(query_symptoms)

    overlap = sorted(qs.intersection(disease_symptoms))
    missing = sorted(qs.difference(disease_symptoms))
    extra = sorted(disease_symptoms.difference(qs))[:15]

    return {
        "overlap": overlap,
        "missing_from_disease_profile": missing,
        "other_common_symptoms_for_disease": extra,
    }


FOLLOWUP_MARGIN = 0.15   # dacă diferența dintre locul 1 și locul 2 e sub 15%, cerem follow-up
MAX_QUESTIONS = 3        # maxim 3 întrebări odată


def needs_followup(results) -> bool:
    """Returnează True dacă primele 2 rezultate sunt prea apropiate."""
    if len(results) < 2:
        return False
    score1 = results[0][0]
    score2 = results[1][0]
    # Normalizare: dacă ambele scoruri sunt pozitive, comparam relativ
    if score1 <= 0:
        return False
    gap = (score1 - score2) / max(abs(score1), 1e-6)
    return gap < FOLLOWUP_MARGIN


def get_discriminating_questions(results, known_symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    Găsește simptomele care separă cel mai bine bolile din top rezultate.

    Strategia:
    - Luăm top 3 boli candidate
    - Pentru fiecare simptom din profilele lor, calculăm un scor de discriminare:
        * simptomul există la boala #1 dar NU la #2 și #3 → întrebare valoroasă
        * simptomul există la TOATE bolile → nu ajută
        * simptomul pe care utilizatorul l-a menționat deja → sărim peste el
    - Returnăm top MAX_QUESTIONS simptome ca întrebări yes/no
    """
    if not results:
        return []

    known = set(s.lower().strip() for s in known_symptoms)

    # Luăm profilurile pentru top 3 (sau câte sunt)
    candidates = results[:min(3, len(results))]
    profiles = []
    for _, _, _, _, m in candidates:
        syms = set(m.get("symptoms_canon", m.get("symptoms", [])))
        profiles.append(syms)

    if not profiles:
        return []

    # Toate simptomele din toate profilurile candidate
    all_symptoms = set()
    for p in profiles:
        all_symptoms.update(p)

    # Eliminăm simptomele deja cunoscute
    candidates_syms = all_symptoms - known

    if not candidates_syms:
        return []

    questions = []
    top_profile = profiles[0]
    disease_names = [r[4].get("disease", "") for r in candidates]

    for sym in candidates_syms:
        # În câte boli candidate apare simptomul
        present_in = [i for i, p in enumerate(profiles) if sym in p]

        # Simptomele prezente în TOATE bolile nu discriminează
        if len(present_in) == len(profiles):
            continue

        # Simptomele prezente în NICIO boală nu sunt relevante
        if len(present_in) == 0:
            continue

        # Scor de discriminare:
        # +2 dacă e în boala #1 dar nu în #2
        # +1 dacă e în boala #1 dar nu în #3
        # -1 dacă e în bolile concurente dar nu în #1 (ajută la excludere)
        disc_score = 0
        in_top = sym in top_profile

        if in_top:
            for i in range(1, len(profiles)):
                if sym not in profiles[i]:
                    disc_score += (2 if i == 1 else 1)
        else:
            # Simptomul e în competitori dar nu în top → dacă răspunde yes, eliminăm top
            disc_score = 1

        if disc_score > 0:
            # Care boli ar fi confirmate / excluse de acest simptom
            confirms = [disease_names[i] for i, p in enumerate(profiles) if sym in p]
            excludes = [disease_names[i] for i, p in enumerate(profiles) if sym not in p]

            questions.append({
                "symptom": sym,
                "question": f"Do you also have {sym}?",
                "disc_score": disc_score,
                "confirms_diseases": confirms,
                "excludes_diseases": excludes,
            })

    # Sortăm după scor discriminare descrescător, luăm top MAX_QUESTIONS
    questions.sort(key=lambda x: x["disc_score"], reverse=True)
    return questions[:MAX_QUESTIONS]


def apply_followup_answers(
    results,
    answers: Dict[str, bool],
    idf: Dict[str, float],
) -> list:
    """
    Re-scorează rezultatele după răspunsurile yes/no ale utilizatorului.

    answers = {"fever": True, "ear pain": False, ...}

    Logica:
    - Simptom confirmat (yes) + prezent în profil → bonus
    - Simptom confirmat (yes) + absent din profil → penalizare
    - Simptom negat (no) + prezent în profil → penalizare (boala necesită simptomul dar pacientul nu îl are)
    """
    if not answers or not results:
        return results

    updated = []
    for final, emb, ov, miss, m in results:
        ds = set(m.get("symptoms_canon", m.get("symptoms", [])))
        bonus = 0.0

        for sym, has_it in answers.items():
            sym_idf = float(idf.get(sym, 1.0))
            weight = sym_idf / 10.0  # normalizat să nu domine

            if has_it and sym in ds:
                bonus += weight        # confirmare → boost
            elif has_it and sym not in ds:
                bonus -= weight * 1.5  # are simptomul dar boala nu îl include → penalizare
            elif not has_it and sym in ds:
                bonus -= weight * 0.8  # nu are simptomul dar boala îl necesită → penalizare mică

        updated.append((final + bonus, emb, ov, miss, m))

    updated.sort(key=lambda x: x[0], reverse=True)
    return updated


def clean_article_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = clean_article_text(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def safe_request(url, headers, retries=3):
    for attempt in range(retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response

        print(f"Retry {attempt+1}... status:", response.status_code)
        time.sleep(2 * (attempt + 1))  # backoff
    return response

def search_medical_articles(disease_name: str, page_size: int = ARTICLE_PAGE_SIZE) -> List[Dict[str, Any]]:
    disease_name = str(disease_name or "").strip()
    if not disease_name:
        return []
    #query = f"{disease_name} review treatment diagnosis"
    query = f'{disease_name} (review OR guideline OR treatment OR diagnosis OR prevention)'
    try:

        headers = {"User-Agent": "Mozilla/5.0",
                   "Accept": "application/json"
                   }
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

        params = {
            "query": query,
            "format": "json",
            "pageSize":max(page_size * 5, 35),
            "resultType": "core"
        }

        url = f"{base_url}?{urlencode(params)}"

        response = safe_request(url, headers)#response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        if "application/json" not in response.headers.get("Content-Type", ""):
            print("Non-JSON response:", response.text[:200])
            return []

        data = response.json()


        raw_results = data.get("resultList", {}).get("result", [])
        articles = []

        disease_l = disease_name.lower()

        bad_terms = [
            "atopic dermatitis",
            "dermatitis",
            "multiple sclerosis",
            "bell's palsy",
            "facial palsy",
            "stroke",
            "neuropathy",
            "acl",
            "athletes",
            "animal study",
            "mouse",
            "mice",
            "rat",
            "rats",
            "in vitro",
        ]

        good_terms = [
            "review",
            "guideline",
            "management",
            "diagnosis",
            "treatment",
            "clinical",
            "symptoms",
            "prevention"
        ]
        for item in raw_results:
            title = clean_article_text(item.get("title", ""))
            abstract = clean_article_text(item.get("abstractText", ""))
            if len(title) == 0:
                continue

            journal = item.get("journalTitle", "")
            year = item.get("pubYear", "")
            authors = item.get("authorString", "")
            doi = item.get("doi", "")
            pmid = item.get("pmid", "")

            combined = f"{title} {abstract}".lower()

            title_l = title.lower()

            if disease_l not in combined and disease_l not in title_l:
                continue

            if any(term in combined for term in bad_terms):
                continue

            if len(abstract.split()) < 20:
                continue

            score = 0

            if disease_l in title.lower():
                score += 5
            if disease_l in abstract.lower():
                score += 3

            for term in good_terms:
                if term in combined:
                    score += 1

            for term in [
                "symptom", "diagnosis", "treatment", "therapy",
                "management", "prevention", "risk factor",
                "complication", "clinical presentation"
            ]:
                if term in combined:
                    score += 1

            articles.append({
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
                "doi": doi,
                "pmid": pmid,
                "_score": score
            })

        articles.sort(key=lambda x: x["_score"], reverse=True)

        final_articles = []
        for article in articles[:page_size]:
            article.pop("_score", None)
            final_articles.append(article)

        return final_articles

    except Exception as e:
        print(f"Error searching medical articles: {e}")
        return []


def search_medical_articles_fallback(disease_name: str, page_size: int = ARTICLE_PAGE_SIZE) -> List[Dict[str, Any]]:
    disease_name = str(disease_name or "").strip()
    if not disease_name:
        return []

    query = (
        f'"{disease_name}" AND (symptoms OR diagnosis OR treatment OR prevention OR management)'
    )

    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "lite", #"core",
        "sort": "RELEVANCE"
    }

    try:
        response = requests.get(EUROPE_PMC_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        raw_results = data.get("resultList", {}).get("result", [])
        articles = []

        disease_l = disease_name.lower()

        for item in raw_results:
            title = clean_article_text(item.get("title", ""))
            abstract = clean_article_text(item.get("abstractText", ""))
            combined = f"{title} {abstract}".lower()

            if disease_l not in combined:
                continue


            articles.append({
                "title": title,
                "abstract": abstract,
                "journal": item.get("journalTitle", ""),
                "year": item.get("pubYear", ""),
                "authors": item.get("authorString", ""),
                "doi": item.get("doi", ""),
                "pmid": item.get("pmid", "")
            })

        return articles[:page_size]

    except Exception as e:
        print(f"Error in fallback article search: {e}")
        return []


def pick_sentences(sentences: List[str], keywords: List[str], max_sentences: int = 2) -> List[str]:
    selected = []
    keywords = [k.lower() for k in keywords]

    bad_sentence_terms = [
        "methods",
        "objective",
        "objectives",
        "conclusion",
        "conclusions",
        "observational study",
        "case-control study",
        "retrospective study",
        "prospective study",
        "randomized trial",
        "this study",
        "our study"
    ]

    for sent in sentences:
        sent_l = sent.lower().strip()

        if any(term in sent_l for term in bad_sentence_terms):
            continue

        if len(sent.split()) < 7:
            continue

        if any(k in sent_l for k in keywords):
            selected.append(sent)

        if len(selected) >= max_sentences:
            break

    return selected


def build_summary_from_articles(disease_name: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not articles:
        return {
            "overview": [f"No article summary found for {disease_name}."],
            "causes": [],
            "treatment": [],
            "prevention": [],
            "when_to_see_a_doctor": [],
            "sources": []
        }

    all_sentences = []
    for article in articles:
        all_sentences.extend(split_sentences(article.get("abstract", "")))

    overview = pick_sentences(
        all_sentences,
        ["is a", "characterized by", "defined as", "condition", "disorder", "syndrome"],
        max_sentences=2
    )

    causes = pick_sentences(
        all_sentences,
        ["cause", "causes", "risk factor", "risk factors", "etiology", "associated with"],
        max_sentences=3
    )

    treatment = pick_sentences(
        all_sentences,
        ["treatment", "treated", "therapy", "management", "medication", "medications", "drug", "drugs"],
        max_sentences=3
    )

    prevention = pick_sentences(
        all_sentences,
        ["prevention", "prevent", "preventive", "reduce risk", "screening"],
        max_sentences=3
    )

    when_to_see_a_doctor = pick_sentences(
        all_sentences,
        ["seek medical attention", "consult a doctor", "clinical evaluation", "urgent care", "emergency"],
        max_sentences=2
    )

    sources = []
    for article in articles[:3]:
        sources.append({
            "title": article.get("title", ""),
            "journal": article.get("journal", ""),
            "year": article.get("year", ""),
            "pmid": article.get("pmid", ""),
            "doi": article.get("doi", "")
        })

    return {
        "overview": overview,
        "causes": causes,
        "treatment": treatment,
        "prevention": prevention,
        "when_to_see_a_doctor": when_to_see_a_doctor,
        "sources": sources
    }


def format_summary_as_text(disease_name: str, summary: Dict[str, Any]) -> str:
    def join_sentences(items: List[str], fallback: str) -> str:
        cleaned = [x.strip() for x in items if str(x).strip()]
        if not cleaned:
            return fallback
        text = " ".join(cleaned)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    overview_text = join_sentences(
        summary.get("overview", []),
        f"No overview information was found for {disease_name}."
    )

    causes_text = join_sentences(
        summary.get("causes", []),
        "No clear cause-related information was identified in the retrieved articles."
    )

    treatment_text = join_sentences(
        summary.get("treatment", []),
        "No clear treatment-related information was identified in the retrieved articles."
    )

    prevention_text = join_sentences(
        summary.get("prevention", []),
        "No clear prevention-related information was identified in the retrieved articles."
    )

    doctor_text = join_sentences(
        summary.get("when_to_see_a_doctor", []),
        "No clear advice about when to seek medical care was identified in the retrieved articles."
    )

    final_text = (
        f"\n===== MEDICAL ARTICLE SUMMARY FOR {disease_name.upper()} =====\n\n"
        f"Overview: {overview_text}\n\n"
        f"Causes / Risk factors: {causes_text}\n\n"
        f"Treatment / Management: {treatment_text}\n\n"
        f"Prevention: {prevention_text}\n\n"
        f"When to see a doctor: {doctor_text}\n"
    )

    return final_text


def normalize_disease_name(disease_name: str) -> str:
    disease_name = str(disease_name or "").strip().lower()
    disease_name = re.sub(r"\s*\(.*?\)\s*", " ", disease_name)
    disease_name = re.sub(r"\s+", " ", disease_name).strip()
    return disease_name


def normalize_disease_name_for_search(disease_name: str) -> str:
    disease_name = str(disease_name or "").strip().lower()
    disease_name = re.sub(r"\s*\(.*$", "", disease_name).strip()

    mapping = {
        "diabetes": "diabetes mellitus",
        "flu": "influenza",
        "strep throat": "streptococcal pharyngitis",
        "high blood pressure": "hypertension",
        "heart attack": "myocardial infarction",
        "common cold": "upper respiratory infection",
    }

    return mapping.get(disease_name, disease_name)



def deliver_details(results, symptoms):
    output = {
        "matches": [],
        "top_disease": None,
        "articles": [],
        "summary": {}
    }

    if not results:
        print("No results. Provide more symptoms.\n")
        return output

    print("\nTop matches:")

    for rank, (final, emb, ov, miss, m) in enumerate(results, start=1):
        info = explain(symptoms, m)

        match_entry = {
            "rank": rank,
            "final": float(final),
            "disease": m.get("disease", ""),
            "overlap_symptoms": info.get("overlap", []),
            "missing_from_profile": info.get("missing_from_disease_profile", [])
        }

        output["matches"].append(match_entry)

        print(f"\n{rank}. {m.get('disease', '')} | final={final:.3f} emb={emb:.3f} ov_idf={ov:.3f} miss_idf={miss:.3f}")
        print("   overlap:", info.get("overlap", []))

        if info.get("missing_from_disease_profile"):
            print("   query-not-in-profile:", info["missing_from_disease_profile"])

    if not output["matches"]:
        return output

    top_disease = results[0][4].get("disease", "")
    top_disease_clean = normalize_disease_name_for_search(top_disease)

    output["top_disease"] = {
        "clean": top_disease_clean
    }

    print(f"\nTop 1 disease selected for summary: {top_disease_clean}")

    summary = build_offline_summary(top_disease_clean)
    output["summary"] = summary
    output["articles"] = summary.get("sources", [])

    formatted_summary = format_summary_as_text(top_disease_clean, summary)
    print(formatted_summary)

    print("Sources:")
    for src in summary.get("sources", []):
        print("-", src.get("title", ""), "|", src.get("year", ""))

    return output


def main(text):
    nlp = spacy.load(MODEL_PATH)
    index, meta, embedder, idf = load_rag()

    print("Write symptom description (or 'exit').\n")

    while True:
        #text = input("> ").strip()
        if not text or text.lower() in {"exit", "quit"}:
            break

        symptoms = extract_symptoms_ner(nlp, text)
        symptoms = canonicalize_list(symptoms, semantic=True)

        print("\nDetected symptoms (canonical):", symptoms if symptoms else "(none)")

        results = retrieve(index, meta, embedder, idf, symptoms, top_k=TOP_K)
        if not results:
            print("No results. Provide more symptoms.\n")
            continue

        print("\nTop matches:")
        for rank, (final, emb, ov, miss, m) in enumerate(results, start=1):
            info = explain(symptoms, m)
            print(f"\n{rank}. {m['disease']} | final={final:.3f} emb={emb:.3f} ov_idf={ov:.3f} miss_idf={miss:.3f}")
            print("   overlap:", info["overlap"])
            if info["missing_from_disease_profile"]:
                print("   query-not-in-profile:", info["missing_from_disease_profile"])

        top_disease = results[0][4]["disease"]
        top_disease_clean = normalize_disease_name_for_search(top_disease)

        print(f"\nTop 1 disease selected for summary: {top_disease_clean}")

        articles = search_medical_articles(top_disease_clean, page_size=ARTICLE_PAGE_SIZE)
        if len(articles) < 2:
            articles = search_medical_articles_fallback(top_disease_clean, page_size=ARTICLE_PAGE_SIZE)
        if not articles:
            print("\nNo real medical articles were found for this disease.\n")
            continue

        summary = build_summary_from_articles(top_disease_clean, articles)
        formatted_summary = format_summary_as_text(top_disease_clean, summary)
        print(formatted_summary)

        print("Sources:")
        for src in summary.get("sources", []):
            print("-", src.get("title", ""), "|", src.get("year", ""))
        print()

if __name__ == "__main__":
    pass