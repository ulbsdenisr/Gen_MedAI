from head_training import predict_attributes
from history_manager import HistoryManager
from normalize_symptoms import disease_scores
from symptom_normalizer import SymptomNormalizer
from rag_retrieval import DiseaseRAGResponder
import spacy
nlp2 = spacy.load("model/model_with_textcats")
rag = DiseaseRAGResponder(
    index_path="rag_index/index.faiss",
    metadata_path="rag_index/meta.json"
)
#text="I have a really bad stomach ache that is getting worse."
#text = "I feel pain in my stomach that gets worse by the day."
#text="I've been experiencing mild joint pain and extreme fatigue and now I also have a headache."
text="I have a severe headache that gets worse by the day, I'm nauseous and dizzy and have abdominal bloating."
doc = nlp2(text)

print(text)
results = list(predict_attributes(nlp2, doc))
print(results)
history_manager=HistoryManager()
history_manager.append_to_history(results)
history_manager.export_to_pdf()
symptom_nor=SymptomNormalizer()
symptom_list=[]
for result in results:
    symptom_list.append(symptom_nor.normalize_if_certain(result['symptom'])['normalized'])
disease_list=symptom_nor.find_diseases(symptom_list)
reply = rag.build_reply(disease_list["disease"].tolist())
print(reply)
#for entry in results:
#    print(entry)
#history_manager.append_to_history(results)
#history_manager.export_to_pdf()





