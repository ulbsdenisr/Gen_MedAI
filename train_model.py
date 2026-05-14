"""
Train spaCy NER model for medical symptom extraction
"""

import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding, fix_random_seed


TRAIN_PATH = Path("train.spacy")
OUTPUT_DIR = Path("model/model_with_textcats")

LABEL = "SYMPTOM"
EPOCHS = 20
DROPOUT = 0.2
BATCH_SIZES = (4, 32)


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Run convert_annotations.py first.")

    fix_random_seed(42)
    random.seed(42)

    nlp = spacy.blank("en")

    ner = nlp.add_pipe("ner")
    ner.add_label(LABEL)

    docbin = DocBin().from_disk(TRAIN_PATH)
    docs = list(docbin.get_docs(nlp.vocab))

    if not docs:
        raise ValueError("No training documents found in train.spacy.")

    examples = [
        Example.from_dict(
            doc,
            {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]},
        )
        for doc in docs
    ]

    optimizer = nlp.initialize(get_examples=lambda: examples)

    for epoch in range(EPOCHS):
        random.shuffle(examples)
        losses = {}

        batches = minibatch(examples, size=compounding(*BATCH_SIZES, 1.5))

        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Losses: {losses}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)

    print(f"NER model trained and saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()