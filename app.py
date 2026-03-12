from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from head_training import predict_attributes
import json
from pathlib import Path
import sys

from history_manager import HistoryManager
from symptom_normalizer import SymptomNormalizer

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global variables for models (loaded on demand)
nlp = None
index = None
meta = None
embedder = None
idf = None
rag_loaded = False

def write_to_history(nlp,text):
    doc=nlp(text)
    history_list=predict_attributes(nlp,doc)
    history_manager = HistoryManager()
    history_manager.append_to_history(history_list)
    history_manager.export_to_pdf()
    symptom_nor = SymptomNormalizer()
    symptom_list = []
    for result in history_list:
        symptom_list.append(symptom_nor.normalize_if_certain(result['symptom'])['normalized'])
    return symptom_list

def load_models():
    """Load models on demand"""
    global nlp, index, meta, embedder, idf, rag_loaded

    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("model/model_with_textcats")
            print("✓ spaCy NER model loaded")
        except Exception as e:
            print(f"✗ Error loading NER model: {e}")
            nlp = None

    if not rag_loaded:
        try:
            from rag_ner_pipeline import load_rag
            index, meta, embedder, idf = load_rag()
            rag_loaded = True
            print("✓ RAG index loaded")
        except Exception as e:
            print(f"✗ Error loading RAG index: {e}")
            index, meta, embedder, idf = None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user message and return medical insights"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Load models if not already loaded
        load_models()

        if nlp is None:
            return jsonify({'error': 'NER model not available. Please check model training.'}), 503

        # Extract symptoms using NER
        try:
            from rag_ner_pipeline import extract_symptoms_ner
            ####THIS IS WHERE EVERYTHING GOES####
            #symptoms = extract_symptoms_ner(nlp, user_message) #extracts and normalizes them
            symptoms=write_to_history(nlp,user_message)
            print(symptoms)
            #I have a severe fever and a pounding headache
            #['severe fever', 'pounding headache']

        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            symptoms = []

        # Retrieve relevant information using RAG
        retrieved = []
        if rag_loaded and index is not None:
            try:
                from rag_ner_pipeline import retrieve
                retrieved = retrieve(index, meta, embedder, idf, symptoms)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                retrieved = []

        # Format retrieved diseases
        diseases = []
        if retrieved:
            for doc in retrieved[:3]:  # Top 3 results
                try:
                    # Handle the tuple format: (score1, score2, score3, score4, disease_data)
                    if isinstance(doc, tuple) and len(doc) >= 5:
                        disease_data = doc[4]  # The 5th element contains the disease info
                        if isinstance(disease_data, dict) and 'disease' in disease_data:
                            diseases.append(disease_data['disease'])
                    # Handle dict format as fallback
                    elif isinstance(doc, dict) and 'disease' in doc:
                        diseases.append(doc['disease'])
                    # Handle string format as fallback
                    elif isinstance(doc, str):
                        try:
                            import json
                            doc_data = json.loads(doc)
                            if isinstance(doc_data, dict) and 'disease' in doc_data:
                                diseases.append(doc_data['disease'])
                        except:
                            pass
                except Exception as e:
                    print(f"Error parsing document: {e}")
                    continue

        # Format response
        response = {
            'user_message': user_message,
            'extracted_symptoms': symptoms,
            'possible_diseases': diseases,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Check if models are loaded"""
    load_models()  # Try to load models
    return jsonify({
        'ner_model': nlp is not None,
        'rag_index': rag_loaded,
        'ready': nlp is not None
    })


if __name__ == '__main__':
    print("\n🚀 Starting Medical AI Chat Interface...")
    print("📍 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
