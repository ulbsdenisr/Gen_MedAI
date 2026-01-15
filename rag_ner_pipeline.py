"""
RAG + NER integrat end-to-end (local)

Flux:
1) Text utilizator
2) spaCy NER (SYMPTOM) -> simptome brute
3) normalize_and_split_symptoms -> simptome separate/curățate
4) canonicalize_list -> mapare simplă la forme standard ("high fever" -> "fever")
5) Retrieval FAISS (embeddings) -> candidați
6) Re-rank cu scor hibrid:
   - overlap ponderat cu IDF (simptome rare contează mai mult)
   - + embedding similarity
7) Afișează top boli + explicații
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer

from symptom_utils import normalize_and_split_symptoms
from symptom_mapper import canonicalize_list  # modulul tău cu reguli


# ===== CONFIG =====
RAG_DIR = Path("rag_index")
MODEL_PATH = "model/model-best"     # spaCy NER antrenat
LABEL = "SYMPTOM"
TOP_K = 5

# Ponderi scor final
W_OVERLAP = 0.75   # overlap ponderat IDF
W_EMB = 0.25       # embedding similarity

# Câți candidați luăm din FAISS înainte de rerank
CAND_MULT = 10     # top_k * CAND_MULT (minim 50)
# ==================


def clean_piece(s: str) -> str:
    """Curățare simplă a textului."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'()[]{}")
    return s


def load_rag() -> Tuple[Any, List[Dict[str, Any]], SentenceTransformer, Dict[str, float]]:
    """Încarcă indexul FAISS, meta, embedder și idf."""
    index_path = RAG_DIR / "index.faiss"
    meta_path = RAG_DIR / "meta.json"
    model_name_path = RAG_DIR / "model_name.txt"
    idf_path = RAG_DIR / "idf.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Lipsește {index_path}. Rulează build_index.py.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Lipsește {meta_path}. Rulează build_index.py.")
    if not model_name_path.exists():
        raise FileNotFoundError(f"Lipsește {model_name_path}. Rulează build_index.py.")
    if not idf_path.exists():
        raise FileNotFoundError(f"Lipsește {idf_path}. Rulează build_index.py (cu IDF).")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    embed_model_name = model_name_path.read_text(encoding="utf8").strip()
    embedder = SentenceTransformer(embed_model_name)
    idf = json.loads(idf_path.read_text(encoding="utf8"))

    return index, meta, embedder, idf


def extract_symptoms_ner(nlp, text: str) -> List[str]:
    """Extrage simptome cu NER și le normalizează/sparte."""
    doc = nlp(text)
    raw = [ent.text for ent in doc.ents if ent.label_ == LABEL]
    raw = [clean_piece(x) for x in raw]

    final = normalize_and_split_symptoms(raw)
    final = [clean_piece(x) for x in final]

    # scoate goluri/duplicate
    final = [x for x in final if x]
    final = list(dict.fromkeys(final))
    return final


def weighted_overlap_score(qs: set, ds: set, idf: Dict[str, float]) -> float:
    """
    Overlap ponderat cu IDF:
    - simptomele rare contează mai mult
    - returnează valoare între 0..1 (aprox)
    """
    overlap = qs.intersection(ds)
    num = sum(float(idf.get(s, 1.0)) for s in overlap)
    den = sum(float(idf.get(s, 1.0)) for s in qs) or 1.0
    return num / den


def retrieve(index, meta, embedder, idf, query_symptoms: List[str], top_k: int = TOP_K):
    """Caută candidați cu FAISS și rerank cu overlap ponderat IDF."""
    if not query_symptoms:
        return []

    query = "Symptoms: " + ", ".join(query_symptoms)
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    cand_n = max(top_k * CAND_MULT, 50)
    scores, ids = index.search(q_emb, cand_n)

    qs = set(query_symptoms)
    results = []

    for emb_score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue

        disease = meta[idx]
        ds = set(disease["symptoms"])

        ov = weighted_overlap_score(qs, ds, idf)
        final = W_OVERLAP * ov + W_EMB * float(emb_score)

        results.append((final, float(emb_score), float(ov), disease))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def explain(query_symptoms: List[str], disease_meta: dict) -> dict:
    """Explicație: overlap + ce lipsește + simptome extra."""
    disease_symptoms = set(disease_meta["symptoms"])
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

    print("Scrie o descriere a simptomelor (sau 'exit').\n")

    while True:
        text = input("> ").strip()
        if not text or text.lower() in {"exit", "quit"}:
            break

        # 1) NER -> simptome
        symptoms = extract_symptoms_ner(nlp, text)

        # 2) Canonicalizare (mapare simplă)
        symptoms = canonicalize_list(symptoms)

        print("\nSimptome detectate (canonical):", symptoms if symptoms else "(nimic)")

        # 3) Retrieval + rerank
        results = retrieve(index, meta, embedder, idf, symptoms, top_k=TOP_K)
        if not results:
            print("Nu am găsit rezultate. Încearcă să descrii mai multe simptome.\n")
            continue

        # 4) Afișare
        print("\nTop potriviri:")
        for rank, (final, emb, ov, m) in enumerate(results, start=1):
            info = explain(symptoms, m)
            print(f"\n{rank}. {m['disease']} | final={final:.3f} emb={emb:.3f} overlap_idf={ov:.3f}")
            print("   overlap:", info["overlap"])
            if info["missing_from_disease_profile"]:
                print("   query-not-in-profile:", info["missing_from_disease_profile"])

        print()


if __name__ == "__main__":
    main()
