#obtain normalized symptoms
#only take symptoms whose similarity score is over 50
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("formal symptoms.txt", "r", encoding="utf-8") as f:
    formal = [line.strip() for line in f if line.strip()]

formal_emb = model.encode(formal, convert_to_tensor=True)

def normalize(span):
    span_emb = model.encode(span, convert_to_tensor=True)
    scores = util.cos_sim(span_emb, formal_emb)[0]
    idx = scores.argmax().item()
    if scores[idx] > 0.3:
        return formal[idx], scores[idx].item()
    return None, scores[idx].item()

raw_symptoms = [
    "tummy hurts",
    "feel very tired",
    "head is pounding",
    "burning feeling in my chest",
    "coughing",
    "three orange cats",
    "head"
]
results = []

for symptom in raw_symptoms:
    normalized, score = normalize(symptom)
    results.append({
        "original": symptom,
        "normalized": normalized,
        "score": score
    })
print("All results:")
for r in results:
    print(r)
certainty_threshold = 0.5
print()
print("Symptoms with certainty over 0.5:")
certain_symptoms = [
    r for r in results
    if r["normalized"] is not None and r["score"] >= certainty_threshold
]
for s in certain_symptoms:
    print(s)

###### SEARCH DISEASES #####
import pandas as pd

df = pd.read_csv(
    "disease dataset.csv",
    low_memory=False
)
certain_symptoms_names = [
    r["normalized"] for r in certain_symptoms
    if r["normalized"] is not None
]
df["match_score"] = df[certain_symptoms_names].sum(axis=1)
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