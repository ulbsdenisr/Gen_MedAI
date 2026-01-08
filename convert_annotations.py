
"""
Convert JSON annotations to spaCy format
"""
import json
import spacy
from spacy.tokens import DocBin

nlp_blank = spacy.blank("en")
db = DocBin()

with open("annotations_final2.json", "r", encoding="utf8") as f:
    data = json.load(f)

for text, annot in data["annotations"]:
    doc = nlp_blank.make_doc(text)
    ents = []

    for start, end, label in annot["entities"]:
        span = doc.char_span(
            start,
            end,
            label=label,
            alignment_mode="contract"
        )
        if span:
            ents.append(span)

    doc.ents = ents
    db.add(doc)

db.to_disk("train.spacy")
print("train.spacy created successfully")