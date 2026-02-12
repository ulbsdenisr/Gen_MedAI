"""
Train spaCy NER model for medical symptom extraction
"""
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path
import random

# 1. Create blank NLP object
nlp = spacy.blank("en")

# 2. Add NER pipeline component
ner = nlp.add_pipe("ner")

# 3. Add labels
ner.add_label("SYMPTOM")

# 4. Load training dataset
docbin = DocBin().from_disk("train.spacy")
docs = list(docbin.get_docs(nlp.vocab))

# 5. Initialize model
optimizer = nlp.initialize()

# 6. Training loop
EPOCHS = 100

for epoch in range(EPOCHS):
    random.shuffle(docs)
    losses = {}
    examples = []

    for doc in docs:
        # Create Example from gold annotations already in doc
        example = Example.from_dict(
            doc,
            {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
        )
        examples.append(example)

    nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1}/{EPOCHS} - Losses: {losses}")

# 7. Save model
output_dir = Path("model/model-best")
output_dir.mkdir(parents=True, exist_ok=True)
nlp.to_disk(output_dir)

print("NER model trained and saved successfully")