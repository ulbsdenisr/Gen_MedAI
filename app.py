from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path
import sys

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global variables for models (loaded on demand)
nlp = None
index = None
meta = None
embedder = None
idf = None
rag_loaded = False

def load_models():
    """Load models on demand"""
    global nlp, index, meta, embedder, idf, rag_loaded

    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("model/model-best")
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
            symptoms = extract_symptoms_ner(nlp, user_message)
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

        # Format response
        response = {
            'user_message': user_message,
            'extracted_symptoms': symptoms,
            'retrieved_documents': retrieved[:3],  # Top 3 results
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
