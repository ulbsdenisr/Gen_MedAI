"""
Construiește indexul RAG dintr-un CSV boală–simptome:
- agregă simptomele pe boală (union peste rânduri)
- calculează IDF pentru fiecare simptom
- creează embeddings cu sentence-transformers
- salvează index FAISS + metadate
"""

import re
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


#  CONFIG
CSV_PATH = "Disease and symptoms dataset.csv"   # numele exact al fișierului CSV
OUT_DIR = Path("rag_index")
EMBED_MODEL = "all-MiniLM-L6-v2"
# ==================


def clean_symptom(s: str) -> str:
    """Curăță și normalizează un simptom (lowercase, fără punctuație)."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'()[]{}")
    return s


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1) Citește CSV
    df = pd.read_csv(CSV_PATH)
    if df.shape[1] < 2:
        raise ValueError("CSV-ul trebuie să aibă cel puțin 2 coloane (boală + simptome).")

    disease_col = df.columns[0]
    symptom_cols = list(df.columns[1:])

    # Curățăm numele simptomelor (header-ele coloanelor)
    symptom_cols_clean = [clean_symptom(c) for c in symptom_cols]

    # 2) Agregăm simptomele pe boală (union peste rânduri)
    grouped = {}  # boală -> set(simptome)
    for _, row in df.iterrows():
        disease = clean_symptom(row[disease_col])
        flags = row[symptom_cols].values

        present = []
        for i, v in enumerate(flags):
            try:
                iv = int(float(v))  # suport pentru 0/1, 0.0/1.0
            except Exception:
                iv = 0
            if iv == 1:
                present.append(symptom_cols_clean[i])

        if disease not in grouped:
            grouped[disease] = set()
        grouped[disease].update(present)

    if not grouped:
        raise ValueError("Nu s-au putut construi profilele bolilor. Verifică CSV-ul.")

    # 3) Calculează DF și IDF pentru fiecare simptom
    symptom_df = Counter()
    for sym_set in grouped.values():
        for s in sym_set:
            symptom_df[s] += 1

    N = len(grouped)
    symptom_idf = {
        s: math.log((N + 1) / (dfreq + 1)) + 1.0
        for s, dfreq in symptom_df.items()
    }

    (OUT_DIR / "idf.json").write_text(
        json.dumps(symptom_idf, ensure_ascii=False, indent=2),
        encoding="utf8",
    )

    # 4) Construiește documente și metadate
    diseases = sorted(grouped.keys())
    docs = []
    meta = []

    for d in diseases:
        symptoms = sorted(grouped[d])
        doc = f"Boală: {d}. Simptome: " + ", ".join(symptoms)
        docs.append(doc)
        meta.append({
            "disease": d,
            "symptoms": symptoms
        })

    print(f"Construite {len(docs)} profile de boală.")

    # 5) Embeddings
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    # 6) Index FAISS (cosine similarity)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 7) Salvare artefacte
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    (OUT_DIR / "model_name.txt").write_text(EMBED_MODEL, encoding="utf8")

    print("Index RAG salvat în folderul `rag_index/`:")
    print(" - index.faiss")
    print(" - meta.json")
    print(" - idf.json")
    print(" - model_name.txt")


if __name__ == "__main__":
    main()
