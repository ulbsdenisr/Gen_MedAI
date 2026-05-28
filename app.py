from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from head_training import predict_attributes
from chat_manager import ChatManager
import json
from user_manager import UserManager
from pathlib import Path
import sys
import os
from flask import send_file
#from history_manager import HistoryManager
from medical_history_manager import MedicalHistoryManager
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
current_user_id = None
chat_manager=ChatManager()
user_manager = UserManager()

def write_to_history(nlp, text):
    doc = nlp(text)

    symptom_list = []
    warnings = []

    # 1. Incearca varianta veche cu predict_attributes,
    # doar daca modelul are componentele textcat necesare
    if "textcat_severity" in nlp.pipe_names and "textcat_status" in nlp.pipe_names:
        try:
            history_list = predict_attributes(nlp, doc)

            history_manager = MedicalHistoryManager()
            warnings = history_manager.append_to_history(
                history_list,
                user_id=current_user_id,
                conversation_id=chat_manager.current_conversation_id
            )

            symptom_nor = SymptomNormalizer()

            for result in history_list:
                raw_symptom = result.get("symptom", "")

                normalized = symptom_nor.normalize_if_certain(raw_symptom)

                if isinstance(normalized, dict) and normalized.get("normalized"):
                    symptom_list.append(normalized["normalized"])
                elif raw_symptom:
                    symptom_list.append(raw_symptom)

        except Exception as e:
            print("History prediction failed:", e)

    # 2. Fallback nou: daca modelul are doar NER
    if not symptom_list:
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                symptom_list.append(ent.text)

    symptom_list = list(dict.fromkeys(symptom_list))

    return symptom_list, warnings

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
        if current_user_id is not None:
            try:
                chat_manager.save_message("user", user_message,current_user_id)
            except Exception as e:
                print("Chat save failed:", e)

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Load models if not already loaded
        load_models()

        if nlp is None:
            return jsonify({'error': 'NER model not available. Please check model training.'}), 503

        # Extract symptoms using NER
        warnings_str = ""
        try:
            from rag_ner_pipeline import extract_symptoms_ner, deliver_details
            ####THIS IS WHERE EVERYTHING GOES####
            #symptoms = extract_symptoms_ner(nlp, user_message) #extracts and normalizes them
            #print("FIRST SYMPTOMS")
            #print(symptoms)
            symptoms, warnings = write_to_history(nlp, user_message)
            from symptom_mapper import canonicalize_list
            from symptom_utils import normalize_and_split_symptoms
            # Re-split inainte de canonicalize — NER poate extrage "thirst, frequent urination" ca una
            symptoms = normalize_and_split_symptoms(symptoms)
            symptoms = canonicalize_list(symptoms, semantic=True)
            # Re-split dupa canonicalize — canonicalize poate produce "thirst, frequent urination"
            final_symptoms = []
            for s in symptoms:
                parts = [p.strip() for p in s.split(',') if p.strip() and len(p.strip()) > 2]
                final_symptoms.extend(parts)
            symptoms = list(dict.fromkeys(final_symptoms))
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

        ####TRY####
        details = deliver_details(retrieved, symptoms)
        print(details)

        if not details or not details.get("matches"):
            response = {
                "user_message": user_message,
                "extracted_symptoms": symptoms,
                "possible_diseases": [],
                "warnings": warnings_str,
                "top_disease": None,
                "status": "no_results"
            }

            if current_user_id is not None:
                chat_manager.save_message("assistant", json.dumps(response), current_user_id)

            return jsonify(response)

        finals = [match["final"] for match in details["matches"]]

        # Normalizare robusta: shiftam scorurile la pozitiv inainte de %
        min_score = min(finals)
        shift = max(0, -min_score) + 1e-6
        shifted = [f + shift for f in finals]
        total = sum(shifted)

        possible_diseases = [
            {
                "rank": match["rank"],
                "percentage": (shifted[i] / total) * 100 if total > 0 else 0,
                "disease": match["disease"],
                "overlap_symptoms": match.get("overlap_symptoms", [])
            }
            for i, match in enumerate(details["matches"])
        ]

        top_disease_data = {
            "name": details["top_disease"]["clean"] if details.get("top_disease") else "",
            "articles": details.get("articles", []),
            "summary": details.get("summary", {})
        }

        # --- Follow-up logic ---
        followup_questions = []
        needs_fu = False
        try:
            from rag_ner_pipeline import needs_followup, get_discriminating_questions
            if needs_followup(retrieved):
                needs_fu = True
                followup_questions = get_discriminating_questions(retrieved, symptoms)
        except Exception as e:
            print(f"Follow-up generation failed: {e}")

        response = {
            'user_message': user_message,
            'extracted_symptoms': symptoms,
            'possible_diseases': possible_diseases,
            'warnings': warnings_str,
            "top_disease": top_disease_data,
            'status': 'success',
            # follow-up fields
            'needs_followup': needs_fu,
            'followup_questions': followup_questions,
        }
        if current_user_id is not None:
            chat_manager.save_message("assistant", json.dumps(response),current_user_id)
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


@app.route('/api/followup', methods=['POST'])
def followup():
    """
    Re-score diseases after the user answers yes/no follow-up questions.

    Request body:
    {
        "symptoms": ["sore throat", "dizziness"],      # original symptoms
        "answers": {"fever": true, "ear pain": false}  # follow-up answers
    }
    """
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        answers = data.get('answers', {})   # {symptom: true/false}

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        load_models()
        if not rag_loaded or index is None:
            return jsonify({'error': 'RAG index not available'}), 503

        from rag_ner_pipeline import (
            retrieve, deliver_details, apply_followup_answers,
            needs_followup, get_discriminating_questions
        )

        # Re-retrieve with original symptoms
        retrieved = retrieve(index, meta, embedder, idf, symptoms)

        # Apply yes/no answers to re-score
        if answers:
            retrieved = apply_followup_answers(retrieved, answers, idf)

        details = deliver_details(retrieved, symptoms)

        if not details or not details.get("matches"):
            return jsonify({'status': 'no_results', 'possible_diseases': []}), 200

        finals = [m["final"] for m in details["matches"]]

        # Scorurile pot fi negative dupa apply_followup_answers
        # Folosim min-max normalization: shiftam toate la pozitiv, apoi calculam %
        min_score = min(finals)
        shift = max(0, -min_score) + 1e-6   # shift astfel incat cel mai mic sa fie > 0
        shifted = [f + shift for f in finals]
        total = sum(shifted)

        possible_diseases = [
            {
                "rank": m["rank"],
                "percentage": (shifted[i] / total) * 100 if total > 0 else 0,
                "disease": m["disease"],
                "overlap_symptoms": m.get("overlap_symptoms", []),
            }
            for i, m in enumerate(details["matches"])
        ]

        top_disease_data = {
            "name": details["top_disease"]["clean"] if details.get("top_disease") else "",
            "articles": details.get("articles", []),
            "summary": details.get("summary", {}),
        }

        # Check if still ambiguous after answers → offer more questions
        still_ambiguous = needs_followup(retrieved)
        more_questions = []
        if still_ambiguous:
            already_answered = set(answers.keys())
            all_known = set(symptoms) | already_answered
            more_questions = get_discriminating_questions(retrieved, list(all_known))

        response = {
            'status': 'success',
            'extracted_symptoms': symptoms,
            'answered_symptoms': answers,
            'possible_diseases': possible_diseases,
            'top_disease': top_disease_data,
            'needs_followup': still_ambiguous,
            'followup_questions': more_questions,
        }

        if current_user_id is not None:
            chat_manager.save_message("assistant", json.dumps(response), current_user_id)

        return jsonify(response)

    except Exception as e:
        print(f"Error in followup endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    global current_user_id
    if current_user_id is None:
        chat_manager.current_conversation_id = None
        return jsonify({"conversation_id": None})
    try:
        chat_manager.export_current_conversation(current_user_id)
    except Exception as e:
        print("Export failed:", e)

    conversation_id = chat_manager.start_new_chat()
    return jsonify({"conversation_id": conversation_id})

@app.route('/api/load_chat', methods=['POST'])
def load_chat():
    #loads id of "current chat" when user
    #wants to converse in an older chat
    #needed to save messages to the older chat
    data = request.json
    new_id = data.get("conversation_id")
    chat_manager.export_current_conversation(current_user_id)#exports current conversation before switching
    chat_manager.set_current_conversation(new_id)
    return jsonify({"status": "loaded"})

@app.route('/api/get_chat/<conversation_id>', methods=['GET'])
def get_chat(conversation_id):
    """Get messages from a specific conversation"""
    try:
        if current_user_id is None:
            return jsonify({"error": "Not logged in"}), 401

        # Incearca mai intai din fisierul JSON exportat
        chat_file = Path('chats') / str(current_user_id) / f"{conversation_id}.json"

        if chat_file.exists():
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            raw_messages = chat_data.get('messages', chat_data) if isinstance(chat_data, dict) else chat_data
        else:
            # Fallback: citeste direct din SQLite
            raw_messages = chat_manager.get_chat_by_conversation(conversation_id, current_user_id)

        # Parseaza mesajele assistant care sunt JSON strings
        messages = []
        for msg in raw_messages:
            role = msg.get('role', '')
            message_content = msg.get('message', msg.get('content', ''))
            timestamp = msg.get('timestamp', '')

            if role == 'assistant' and isinstance(message_content, str):
                # Incearca sa parseze JSON-ul din mesajul assistant
                try:
                    parsed = json.loads(message_content)
                    messages.append({
                        "role": role,
                        "timestamp": timestamp,
                        "parsed": parsed,  # datele structurate pentru displayAIResponse
                        "message": message_content
                    })
                except (json.JSONDecodeError, TypeError):
                    messages.append({
                        "role": role,
                        "timestamp": timestamp,
                        "message": message_content,
                        "parsed": None
                    })
            else:
                messages.append({
                    "role": role,
                    "timestamp": timestamp,
                    "message": message_content,
                    "parsed": None
                })

        return jsonify({"messages": messages})
    except Exception as e:
        print(f"Error retrieving chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat_history', methods=['GET'])
def chat_history():
    """Get list of all available chats"""
    if current_user_id is None:
        return jsonify({"chats": []})
    try:
        chat_dir = Path('chats') / str(current_user_id)
        if not chat_dir.exists():
            return jsonify({"chats": []})
        
        # Get all JSON files in chats directory
        chat_files = sorted(chat_dir.glob('*.json'), key=os.path.getmtime, reverse=True)
        chats = []
        
        for chat_file in chat_files:
            try:
                with open(chat_file, 'r') as f:
                    chat_data = json.load(f)
                conv_id = chat_file.stem
                preview = ""
                messages = []
                if isinstance(chat_data, dict) and 'messages' in chat_data:
                    messages = chat_data['messages']
                elif isinstance(chat_data, list):
                    messages = chat_data
                # Cauta primul mesaj de tip "user" pentru preview
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        preview = msg.get('message', msg.get('content', ''))[:100]
                        break
                import datetime
                chats.append({
                    "id": conv_id,
                    "timestamp": chat_file.stat().st_mtime,
                    "preview": preview or f"Chat {conv_id}"
                })
            except Exception as e:
                print(f"Error reading chat file {chat_file}: {e}")
                continue
        
        return jsonify({"chats": chats})
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def signup():
    global current_user_id
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user_id = user_manager.create_user(username, password)

    if user_id:
        current_user_id = user_id
        return jsonify({"status": "success", "user_id": user_id})
    else:
        return jsonify({"error": "Username already exists"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    global current_user_id
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user_id = user_manager.login_user(username, password)

    if user_id:
        current_user_id = user_id
        return jsonify({"status": "success", "user_id": user_id})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    global current_user_id
    current_user_id = None
    chat_manager.current_conversation_id = None
    return jsonify({"status": "logged_out"})

@app.route('/api/export_history_json', methods=['POST'])
def export_history_json():
    if current_user_id is None:
        return jsonify({"error": "Not logged in"}), 401

    conversation_id = chat_manager.current_conversation_id

    history_manager = MedicalHistoryManager()

    path = history_manager.export_chat_history_json(
        current_user_id,
        conversation_id
    )

    return jsonify({"file": path})

@app.route('/api/export_history_pdf', methods=['POST'])
def export_history_pdf():
    if current_user_id is None:
        return jsonify({"error": "Not logged in"}), 401

    conversation_id = chat_manager.current_conversation_id
    history_manager = MedicalHistoryManager()

    # Generate JSON first
    json_path = history_manager.export_chat_history_json(
        current_user_id,
        conversation_id
    )

    # Generate PDF
    pdf_path = f"medical_exports/{current_user_id}_{conversation_id}.pdf"

    history_manager.export_json_to_pdf(json_path, pdf_path)

    # 🔥 THIS is the important part
    return send_file(
        pdf_path,
        as_attachment=True,  # forces download
        download_name=f"medical_history_{conversation_id}.pdf",
        mimetype='application/pdf'
    )

@app.route('/api/get_history_timeline', methods=['GET'])
def get_history_timeline():
    if current_user_id is None:
        return jsonify({"error": "Not logged in"}), 401

    conversation_id = chat_manager.current_conversation_id

    history_manager = MedicalHistoryManager()

    data = history_manager.get_chat_history(
        current_user_id,
        conversation_id
    )
    print("DATA")
    print(data)
    return jsonify({"history": data})


if __name__ == '__main__':
    print("\n🚀 Starting Medical AI Chat Interface...")
    print("📍 Open http://localhost:5000 in your browser")
    chat_manager.export_all_conversations_to_json()
    app.run(debug=True, host='0.0.0.0', port=5000)

##My head hurts, I'm feeling nauseous and I have a congested nose
##I have fever, cough, difficulty breathing,fatigue,sore throat,runny nose, muscle pain