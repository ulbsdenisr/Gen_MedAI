"""
Train spaCy NER model for medical symptom extraction
"""
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch


TRAIN_PATH = "train.spacy"
OUTPUT_DIR = Path("model/model-best")

LABEL = "SYMPTOM"
EPOCHS = 20
DROPOUT = 0.2
BATCH_SIZES = (4, 32)  # începe mic, crește treptat


def main():
    # 1) Blank model
    nlp = spacy.blank("en")

    # 2) NER pipe
    ner = nlp.add_pipe("ner")
    ner.add_label(LABEL)

    # 3) Load training data
    docbin = DocBin().from_disk(TRAIN_PATH)
    docs = list(docbin.get_docs(nlp.vocab))

    # 4) Build examples
    examples = []
    for doc in docs:
        examples.append(
            Example.from_dict(
                doc,
                {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]},
            )
        )

    # 5) Initialize
    optimizer = nlp.initialize(get_examples=lambda: examples)

    # 6) Train loop
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        losses = {}

        batches = minibatch(examples, size=spacy.util.compounding(*BATCH_SIZES, 1.5))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

        print(f"Epoch {epoch+1}/{EPOCHS} - Losses: {losses}")

    # 7) Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)
    print(f"NER model trained and saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
