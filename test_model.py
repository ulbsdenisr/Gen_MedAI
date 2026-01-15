import spacy
from symptom_utils import normalize_and_split_symptoms

MODEL_PATH = "model/model-best"

def main():
    nlp = spacy.load(MODEL_PATH)

    tests = [
        "I have a high fever and chills with severe headache.",
        "My throat is sore and I've been coughing a lot.",
        "I feel dizzy and have trouble concentrating.",
    ]

    for text in tests:
        doc = nlp(text)
        raw = [ent.text.lower() for ent in doc.ents if ent.label_ == "SYMPTOM"]
        final = normalize_and_split_symptoms(raw)

        print("\nTEXT:", text)
        print("RAW:", raw)
        print("FINAL:", final)

if __name__ == "__main__":
    main()
