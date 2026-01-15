"""
Convert JSON annotations to spaCy format (DocBin)
"""
import json
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

INPUT_JSON = "annotations_final2.json"
OUTPUT_SPACY = "train.spacy"

nlp_blank = spacy.blank("en")
db = DocBin()

with open(INPUT_JSON, "r", encoding="utf8") as f:
    data = json.load(f)

total_spans = 0
skipped_spans = 0
examples = 0

for text, annot in data["annotations"]:
    doc = nlp_blank.make_doc(text)
    spans = []

    for start, end, label in annot["entities"]:
        total_spans += 1
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            skipped_spans += 1
            continue
        spans.append(span)

    # elimină suprapuneri/duplicate corect
    doc.ents = filter_spans(spans)

    db.add(doc)
    examples += 1

db.to_disk(OUTPUT_SPACY)
print(f"{OUTPUT_SPACY} created successfully")
print(f"Examples: {examples}")
print(f"Total spans: {total_spans} | Skipped (bad alignment): {skipped_spans}")
