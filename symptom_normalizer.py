from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd


class SymptomNormalizer:
    def __init__(self,
                 model_name="all-MiniLM-L6-v2",
                 formal_symptoms_path="formal symptoms.txt",
                 threshold=0.5):

        self.threshold = threshold

        # Load model ONCE
        self.model = SentenceTransformer(model_name)

        # Load formal symptoms
        with open(formal_symptoms_path, "r", encoding="utf-8") as f:
            self.formal = [line.strip() for line in f if line.strip()]

        # Precompute embeddings ONCE
        self.formal_emb = self.model.encode(
            self.formal,
            convert_to_tensor=True
        )

    def normalize(self, symptom_text):
        """
        Returns:
            normalized_name (str or None),
            similarity_score (float)
        """

        span_emb = self.model.encode(symptom_text, convert_to_tensor=True)

        scores = util.cos_sim(span_emb, self.formal_emb)[0]
        idx = torch.argmax(scores).item()
        best_score = scores[idx].item()

        if best_score >= self.threshold:
            return self.formal[idx], best_score

        return None, best_score

    def normalize_if_certain(self, symptom_text):
        """
        Returns normalized symptom ONLY if above threshold.
        Otherwise returns None.
        """

        normalized, score = self.normalize(symptom_text)

        if normalized is not None:
            return {
                "original": symptom_text,
                "normalized": normalized,
                "score": score
            }

        return None
    def find_diseases(self, symptom_list):
        df = pd.read_csv(
            "disease dataset.csv",
            low_memory=False
        )
        df["match_score"] = df[symptom_list].sum(axis=1)
        disease_scores = (
            df.groupby(df.columns[0])["match_score"]
            .max()
            .reset_index()
        )
        disease_scores.columns = ["disease", "score"]
        ranked_diseases = disease_scores.sort_values(
            by="score",
            ascending=False
        )
        MIN_MATCHES = 2
        ranked_diseases = ranked_diseases[
            ranked_diseases["score"] >= MIN_MATCHES
            ]
        print(ranked_diseases.head(5))
        return ranked_diseases.head(5)