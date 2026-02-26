import re
import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Dict

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from symptom_utils import normalize_and_split_symptoms
from symptom_mapper import canonicalize_list

CSV_PATH = "Disease and symptoms dataset.csv"
OUT_DIR = Path("rag_index")
EMBED_MODEL = "all-MiniLM-L6-v2"


def clean_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?:;\"'()[]{}")


def to_bool(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            return int(v) == 1
        except Exception:
            return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "present"}


def normalize_symptoms(symptoms_raw: List[str]) -> List[str]:
    x = [clean_text(s) for s in symptoms_raw if s and str(s).strip()]
    x = normalize_and_split_symptoms(x)
    x = [clean_text(s) for s in x if s and str(s).strip()]
    x = canonicalize_list(x)
    x = [s for s in x if s]
    return list(dict.fromkeys(x))


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (disease + symptoms).")

    disease_col = df.columns[0]
    symptom_cols = list(df.columns[1:])

    symptom_cols_clean = [clean_text(c) for c in symptom_cols]
    symptom_cols_clean = normalize_symptoms(symptom_cols_clean)

    if len(symptom_cols_clean) != len(symptom_cols):
        symptom_cols_clean = [clean_text(c) for c in symptom_cols]

    col_map: Dict[str, str] = {raw: clean for raw, clean in zip(symptom_cols, symptom_cols_clean)}

    grouped: Dict[str, set] = {}

    for _, row in df.iterrows():
        disease = clean_text(row[disease_col])
        if not disease:
            continue

        present_raw = []
        for col in symptom_cols:
            if to_bool(row[col]):
                present_raw.append(col_map[col])

        present = normalize_symptoms(present_raw)

        if disease not in grouped:
            grouped[disease] = set()
        grouped[disease].update(present)

    if not grouped:
        raise ValueError("No disease profiles built. Check CSV formatting.")

    symptom_df = Counter()
    for sym_set in grouped.values():
        for s in sym_set:
            symptom_df[s] += 1

    N = len(grouped)
    symptom_idf = {s: math.log((N + 1) / (dfreq + 1)) + 1.0 for s, dfreq in symptom_df.items()}

    (OUT_DIR / "idf.json").write_text(json.dumps(symptom_idf, ensure_ascii=False, indent=2), encoding="utf8")

    diseases = sorted(grouped.keys())
    docs = []
    meta = []

    for d in diseases:
        symptoms = sorted(grouped[d])
        doc = "Disease: " + d + ". Symptoms: " + ", ".join(symptoms)
        docs.append(doc)
        meta.append({"disease": d, "symptoms_canon": symptoms})

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT_DIR / "model_name.txt").write_text(EMBED_MODEL, encoding="utf8")

    print("RAG index saved to rag_index/:")
    print(" - index.faiss")
    print(" - meta.json")
    print(" - idf.json")
    print(" - model_name.txt")


if __name__ == "__main__":
    main()
