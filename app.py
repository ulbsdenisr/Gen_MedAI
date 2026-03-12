from flask import Flask, request, jsonify
from flask import render_template
from head_training import predict_attributes
from history_manager import HistoryManager
from symptom_normalizer import SymptomNormalizer
from rag_retrieval import DiseaseRAGResponder
import spacy

app = Flask(__name__)

# Initialize once (VERY IMPORTANT — don't load model per request)
rag = DiseaseRAGResponder(
    index_path="rag_index/index.faiss",
    metadata_path="rag_index/meta.json"
)

def process_user_input(user_text):
    """
    This is where your full pipeline runs:
    - NER
    - disease prediction
    - RAG build_reply
    """
    nlp2 = spacy.load("model/model_with_textcats")
    rag = DiseaseRAGResponder(
        index_path="rag_index/index.faiss",
        metadata_path="rag_index/meta.json"
    )
    doc = nlp2(user_text)
    results = list(predict_attributes(nlp2, doc))
    history_manager = HistoryManager()
    history_manager.append_to_history(results)
    history_manager.export_to_pdf()
    symptom_nor = SymptomNormalizer()
    symptom_list = []
    for result in results:
        symptom_list.append(symptom_nor.normalize_if_certain(result['symptom'])['normalized'])
    disease_list = symptom_nor.find_diseases(symptom_list)
    reply = rag.build_reply(disease_list["disease"].tolist())
    return reply

@app.route("/")
def home():
    return render_template("frontend.html")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")

    response_text = process_user_input(user_message)

    return jsonify({"reply": response_text})


if __name__ == "__main__":
    app.run(debug=True)