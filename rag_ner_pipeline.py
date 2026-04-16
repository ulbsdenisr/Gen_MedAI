import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
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

    # 1) queries
    queries = ["Symptoms: " + ", ".join(query_symptoms)]
    queries += [f"Symptom: {s}" for s in query_symptoms]

    q_emb = embedder.encode(queries, normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    cand_n = max(top_k * CAND_MULT, CAND_MIN)
    scores, ids = index.search(q_emb, cand_n)

    # 2) pooling per candidate
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

    # 3) normalize embedding scores to 0..1 over candidates
    cand_scores = np.array(list(pooled.values()), dtype=np.float32)
    s_min = float(cand_scores.min())
    s_max = float(cand_scores.max())
    denom = (s_max - s_min) if (s_max - s_min) > 1e-6 else 1.0

    def norm_emb(x: float) -> float:
        return (x - s_min) / denom

    # 4) query set + IMPORTANT: only in-vocab symptoms count for overlap/miss
    qs = list(dict.fromkeys(query_symptoms))
    qs_set = set(qs)

    qs_in_vocab = {s for s in qs_set if s in idf}
    oov_count = len(qs_set) - len(qs_in_vocab)

    # If everything is OOV, do not force overlap/miss (let embeddings decide)
    use_overlap = len(qs_in_vocab) > 0

    den = sum(float(idf.get(s, 1.0)) for s in qs_in_vocab) if use_overlap else 1.0
    if den <= 0:
        den = 1.0

    results = []
    for idx, raw_emb in pooled.items():
        disease = meta[idx]
        ds = set(disease.get("symptoms_canon", disease.get("symptoms", [])))

        if use_overlap:
            overlap = qs_in_vocab.intersection(ds)
            missing = qs_in_vocab.difference(ds)

            ov_num = sum(float(idf.get(s, 1.0)) for s in overlap)
            miss_num = sum(float(idf.get(s, 1.0)) for s in missing)

            ov = ov_num / den
            miss = miss_num / den

            ov = max(0.0, min(1.0, float(ov)))
            miss = max(0.0, min(1.0, float(miss)))
        else:
            ov, miss = 0.0, 0.0

        emb01 = float(norm_emb(raw_emb))

        # optional: tiny boost for richer queries that are mostly OOV
        # (prevents long detailed text from being disadvantaged)
        oov_boost = min(0.05, 0.01 * oov_count)

        final = (0.75 * ov) - (0.45 * miss) + (0.20 * emb01) + oov_boost

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


def search_medical_articles(disease_name: str, page_size: int = ARTICLE_PAGE_SIZE) -> List[Dict[str, Any]]:
    query = f'"{disease_name}" AND (causes OR treatment OR prevention OR symptoms)'

    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core"
    }

    try:
        response = requests.get(EUROPE_PMC_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        results = data.get("resultList", {}).get("result", [])
        articles = []

        for item in results:
            articles.append({
                "title": item.get("title", ""),
                "abstract": clean_article_text(item.get("abstractText", "")),
                "journal": item.get("journalTitle", ""),
                "year": item.get("pubYear", ""),
                "authors": item.get("authorString", ""),
                "doi": item.get("doi", ""),
                "pmid": item.get("pmid", ""),
            })

        return articles

    except Exception as e:
        print(f"Error searching medical articles: {e}")
        return []


def pick_sentences(sentences: List[str], keywords: List[str], max_sentences: int = 2) -> List[str]:
    selected = []
    keywords = [k.lower() for k in keywords]

    for sent in sentences:
        sent_l = sent.lower()
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
        ["cause", "causes", "risk factor", "associated with", "etiology", "trigger"],
        max_sentences=3
    )

    treatment = pick_sentences(
        all_sentences,
        ["treatment", "treated", "therapy", "management", "medication", "drug", "intervention"],
        max_sentences=3
    )

    prevention = pick_sentences(
        all_sentences,
        ["prevention", "prevent", "reduce risk", "protective", "screening", "lifestyle"],
        max_sentences=3
    )

    when_to_see_a_doctor = pick_sentences(
        all_sentences,
        ["seek medical", "consult", "doctor", "emergency", "urgent", "clinical evaluation"],
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

def main():
    nlp = spacy.load(MODEL_PATH)
    index, meta, embedder, idf = load_rag()

    print("Write symptom description (or 'exit').\n")

    while True:
        text = input("> ").strip()
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
        print(f"\nTop 1 disease selected for summary: {top_disease}")

        articles = search_medical_articles(top_disease, page_size=ARTICLE_PAGE_SIZE)
        summary = build_summary_from_articles(top_disease, articles)

        print("\n===== SUMMARY FROM REAL MEDICAL ARTICLES =====")

        print("\nOverview:")
        for s in summary.get("overview", []):
            print("-", s)

        print("\nCauses:")
        for s in summary.get("causes", []):
            print("-", s)

        print("\nTreatment:")
        for s in summary.get("treatment", []):
            print("-", s)

        print("\nPrevention:")
        for s in summary.get("prevention", []):
            print("-", s)

        print("\nWhen to see a doctor:")
        for s in summary.get("when_to_see_a_doctor", []):
            print("-", s)

        print("\nSources:")
        for src in summary.get("sources", []):
            print("-", src.get("title", ""), "|", src.get("year", ""))

        print()

if __name__ == "__main__":
    main()
