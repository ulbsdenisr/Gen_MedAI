# Gen_MedAI
# Medical AI - Symptom Extraction with spaCy

A custom Named Entity Recognition (NER) model built with spaCy for extracting medical symptoms from natural language text.

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── config.cfg               # spaCy model configuration
├── convert_annotations.py   # Convert JSON annotations to spaCy format
├── train_model.py          # Train the NER model
├── symptom_utils.py        # Utility functions for symptom normalization
├── test_model.py           # Test the trained model
├── annotations_final2.json # Training data (to be added)
└── model/                  # Directory for saved models
    └── model-best/         # Best trained model
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have your training data in `annotations_final2.json` with the following format:
```json
{
  "annotations": [
    ["text sample", {"entities": [[start, end, "SYMPTOM"], ...]}],
    ...
  ]
}
```

## Usage

### 1. Convert Annotations
Convert your JSON annotations to spaCy's binary format:
```bash
python convert_annotations.py
```

### 2. Train Model
Train the NER model:
```bash
python train_model.py
```

The model will train for 20 epochs and save the best version to `model/model-best/`.

### 3. Test Model
Test the trained model:
```bash
python test_model.py
```

## Features

- **Custom NER Model**: Trained specifically for medical symptom extraction
- **Symptom Normalization**: Splits compound symptoms and removes duplicates
- **Pattern Recognition**: Handles common separators (", and, with, plus, accompanied by")

## Example

```python
import spacy
from symptom_utils import normalize_and_split_symptoms

nlp = spacy.load("model/model-best")

text = "I have a high fever and chills with severe headache"
doc = nlp(text)

raw_entities = [ent.text.lower() for ent in doc.ents if ent.label_ == "SYMPTOM"]
final_symptoms = normalize_and_split_symptoms(raw_entities)

print(final_symptoms)
# Output: ['high fever', 'chills', 'severe headache']
```

## Training Results

The model is trained for 20 epochs with the following configuration:
- Language: English
- Pipeline: NER only
- Dropout: 0.2
- GPU allocator: PyTorch

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]