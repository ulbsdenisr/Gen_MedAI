import spacy
import json
from spacy.training.example import Example
from spacy.util import minibatch

def get_examples_severity():
    for text, ann in train_data_severity:
        doc = nlp.make_doc(text)
        yield Example.from_dict(doc, ann)


def get_examples_status():
    for text, ann in train_data_status:
        doc = nlp.make_doc(text)
        yield Example.from_dict(doc, ann)

def predict_attributes(nlp, doc):
    textcat_severity = nlp.get_pipe("textcat_severity")
    textcat_status   = nlp.get_pipe("textcat_status")

    results = []
    for ent in doc.ents:
        symptom_doc = nlp.make_doc(ent.text)

        sev_scores = textcat_severity.predict([symptom_doc])[0]
        sev_label = max(
            zip(textcat_severity.labels, sev_scores),
            key=lambda x: x[1]
        )[0]

        status_scores = textcat_status.predict([symptom_doc])[0]
        status_label = max(
            zip(textcat_status.labels, status_scores),
            key=lambda x: x[1]
        )[0]

        results.append({
            "symptom": ent.text,
            "severity": sev_label,
            "status": status_label
        })

    return results


if __name__ == "__main__":
    nlp = spacy.load("model/model-best")
    print("Pipeline components:", nlp.pipe_names)
    textcat_severity = nlp.add_pipe("textcat", name="textcat_severity", last=True)
    textcat_status = nlp.add_pipe("textcat", name="textcat_status", last=True)

    for label in ["mild", "moderate", "severe"]:
        textcat_severity.add_label(label)
    for label in ["new", "worsening", "improving", "unchanged", "resolved"]:
        textcat_status.add_label(label)

    with open("train_data_severity.json", "r", encoding="utf-8") as f:
        train_data_severity = json.load(f)
    with open("train_data_status.json", "r", encoding="utf-8") as f:
        train_data_status = json.load(f)

    with nlp.select_pipes(enable=["textcat_severity"]):
        nlp.initialize(get_examples_severity)
    with nlp.select_pipes(enable=["textcat_status"]):
        nlp.initialize(get_examples_status)

    examples_severity = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in train_data_severity]
    examples_status = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in train_data_status]

    with nlp.select_pipes(enable=["textcat_severity"]):
        optimizer = nlp.resume_training()
        for i in range(50):
            losses = {}
            batches = minibatch(examples_severity, size=2)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Severity Iter {i}, Losses: {losses}")

    with nlp.select_pipes(enable=["textcat_status"]):
        optimizer = nlp.resume_training()
        for i in range(50):
            losses = {}
            batches = minibatch(examples_status, size=2)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)
            print(f"Status Iter {i}, Losses: {losses}")

    output_dir = "model/model_with_textcats"
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

