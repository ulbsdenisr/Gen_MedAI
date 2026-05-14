import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?:;\"'()[]{}")


@dataclass
class SemanticMapper:
    rag_dir: Path = Path("rag_index")
    score_threshold: float = 0.72
    margin: float = 0.10
    topn: int = 2

    _vocab: Optional[List[str]] = None
    _index: Optional[faiss.Index] = None
    _model: Optional[SentenceTransformer] = None

    def load(self):
        vocab_path = self.rag_dir / "symptom_vocab.json"
        index_path = self.rag_dir / "symptom_vocab.faiss"
        model_path = self.rag_dir / "symptom_vocab_model.txt"

        if not vocab_path.exists() or not index_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Missing semantic mapper files. Run build_semantic_mapper.py first."
            )

        self._vocab = json.loads(vocab_path.read_text(encoding="utf8"))
        self._index = faiss.read_index(str(index_path))

        model_name = model_path.read_text(encoding="utf8").strip()
        self._model = SentenceTransformer(model_name)

        return self

    def _ensure_loaded(self):
        if self._vocab is None or self._index is None or self._model is None:
            self.load()

    def _pick(self, scores: np.ndarray, ids: np.ndarray) -> Tuple[str, float]:
        best_id = int(ids[0])
        best_score = float(scores[0])

        if best_id < 0:
            return "", best_score

        if best_score < self.score_threshold:
            return "", best_score

        if len(ids) > 1 and int(ids[1]) != -1:
            second_score = float(scores[1])
            if (best_score - second_score) < self.margin:
                return "", best_score

        return clean_text(self._vocab[best_id]), best_score

    def map_one(self, phrase: str) -> Tuple[str, float]:
        self._ensure_loaded()

        p = clean_text(phrase)
        if not p:
            return "", 0.0

        emb = self._model.encode([p], normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")

        scores, ids = self._index.search(emb, self.topn)

        mapped, score = self._pick(scores[0], ids[0])

        if not mapped:
            return p, float(scores[0][0])

        return mapped, score

    def map_list(self, phrases: List[str]) -> List[str]:
        self._ensure_loaded()

        cleaned = [clean_text(p) for p in phrases]
        cleaned = [p for p in cleaned if p]

        if not cleaned:
            return []

        emb = self._model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        emb = np.asarray(emb, dtype="float32")

        scores, ids = self._index.search(emb, self.topn)

        out: List[str] = []

        for i, original in enumerate(cleaned):
            mapped, _ = self._pick(scores[i], ids[i])
            out.append(mapped if mapped else original)

        return list(dict.fromkeys(out))