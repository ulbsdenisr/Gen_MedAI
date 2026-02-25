import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer

from symptom_utils import normalize_and_split_symptoms
from symptom_mapper import canonicalize_list


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


def main():
    nlp = spacy.load(MODEL_PATH)
    index, meta, embedder, idf = load_rag()

    print("Write symptom description (or 'exit').\n")

    while True:
        text = input("> ").strip()
        if not text or text.lower() in {"exit", "quit"}:
            break

        symptoms = extract_symptoms_ner(nlp, text)

        # IMPORTANT: enable semantic mapper fallback
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

        print()


if __name__ == "__main__":
    main()
