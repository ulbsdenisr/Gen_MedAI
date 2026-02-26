import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

RAG_DIR = Path("rag_index")
IDF_PATH = RAG_DIR / "idf.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
OUT_VOCAB = RAG_DIR / "symptom_vocab.json"
OUT_INDEX = RAG_DIR / "symptom_vocab.faiss"
OUT_MODEL = RAG_DIR / "symptom_vocab_model.txt"


def main():
    if not IDF_PATH.exists():
        raise FileNotFoundError(f"Missing {IDF_PATH}. Run build_index.py first.")

    idf = json.loads(IDF_PATH.read_text(encoding="utf8"))
    vocab = sorted([k.strip() for k in idf.keys() if str(k).strip()])

    if not vocab:
        raise ValueError("Empty vocabulary from idf.json")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(vocab, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    OUT_VOCAB.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf8")
    faiss.write_index(index, str(OUT_INDEX))
    OUT_MODEL.write_text(EMBED_MODEL, encoding="utf8")

    print("Semantic mapper artifacts saved to rag_index/:")
    print(" - symptom_vocab.json")
    print(" - symptom_vocab.faiss")
    print(" - symptom_vocab_model.txt")


if __name__ == "__main__":
    main()