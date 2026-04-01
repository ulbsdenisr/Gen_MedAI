from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from head_training import predict_attributes
from chat_manager import ChatManager
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
chat_manager=ChatManager()

def write_to_history(nlp,text):
    doc=nlp(text)
    history_list=predict_attributes(nlp,doc)
    history_manager = HistoryManager()
    warnings=history_manager.append_to_history(history_list)
    history_manager.export_to_pdf()
    symptom_nor = SymptomNormalizer()
    symptom_list = []
    for result in history_list:
        symptom_list.append(symptom_nor.normalize_if_certain(result['symptom'])['normalized'])
    return symptom_list,warnings

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
        try:
            chat_manager.save_message("user", user_message)
        except Exception as e:
            print("Chat save failed:", e)

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
            symptoms,warnings=write_to_history(nlp,user_message)
            print(symptoms)
            print(warnings)
            # warnings could be list of strings or other types
            warnings_str = (
                ", ".join(warnings) if isinstance(warnings, list)
                else str(warnings) if warnings
                else ""
            )
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
            'warnings':warnings_str,
            'status': 'success'
        }
        chat_manager.save_message("assistant", json.dumps(response))
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

@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    ##to be called when doing the new chat button
    chat_manager.export_current_conversation()  # save previous chat
    conversation_id = chat_manager.start_new_chat()
    return jsonify({"conversation_id": conversation_id})

@app.route('/api/load_chat', methods=['POST'])
def load_chat():
    #loads id of "current chat" when user
    #wants to converse in an older chat
    #needed to save messages to the older chat
    data = request.json
    new_id = data.get("conversation_id")
    chat_manager.export_current_conversation()#exports current conversation before switching
    chat_manager.set_current_conversation(new_id)
    return jsonify({"status": "loaded"})

@app.route('/api/get_chat/<conversation_id>', methods=['GET'])
def get_chat(conversation_id):
    """Get messages from a specific conversation"""
    try:
        from pathlib import Path
        chat_file = Path('chats') / f"{conversation_id}.json"
        
        if not chat_file.exists():
            return jsonify({"error": "Chat not found"}), 404
        
        with open(chat_file, 'r') as f:
            chat_data = json.load(f)
        
        # Handle both formats: list of messages and dict with messages key
        if isinstance(chat_data, dict) and 'messages' in chat_data:
            messages = chat_data['messages']
        elif isinstance(chat_data, list):
            messages = chat_data
        else:
            messages = []
        
        return jsonify({"messages": messages})
    except Exception as e:
        print(f"Error retrieving chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat_history', methods=['GET'])
def chat_history():
    """Get list of all available chats"""
    try:
        from pathlib import Path
        import os
        chat_dir = Path('chats')
        if not chat_dir.exists():
            return jsonify({"chats": []})
        
        # Get all JSON files in chats directory
        chat_files = sorted(chat_dir.glob('*.json'), key=os.path.getmtime, reverse=True)
        chats = []
        
        for chat_file in chat_files:
            try:
                with open(chat_file, 'r') as f:
                    chat_data = json.load(f)
                    # Extract the conversation ID (filename without extension)
                    conv_id = chat_file.stem
                    # Get first message as preview
                    preview = ""
                    if isinstance(chat_data, list) and len(chat_data) > 0:
                        first_msg = chat_data[0]
                        if isinstance(first_msg, dict) and 'content' in first_msg:
                            preview = first_msg['content'][:100]
                    
                    chats.append({
                        "id": conv_id,
                        "timestamp": chat_file.stat().st_mtime,
                        "preview": preview
                    })
            except Exception as e:
                print(f"Error reading chat file {chat_file}: {e}")
                continue
        
        return jsonify({"chats": chats})
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Starting Medical AI Chat Interface...")
    print("📍 Open http://localhost:5000 in your browser")
    chat_manager.export_all_conversations_to_json()
    app.run(debug=True, host='0.0.0.0', port=5000)

##My head hurts, I'm feeling nauseous and I have a congested nose