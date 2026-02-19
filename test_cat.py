from head_training import predict_attributes
from history_manager import HistoryManager
import spacy
nlp2 = spacy.load("model/model_with_textcats")


text="I've been experiencing mild joint pain and extreme fatigue and now I also have a headache."
doc = nlp2(text)

print(text)
results = list(predict_attributes(nlp2, doc))
history_manager = HistoryManager()
for entry in results:
    print(entry)
history_manager.append_to_history(results)
history_manager.export_to_pdf()


print()

text="I have a really bad stomach ache that is getting worse."
doc = nlp2(text)
print(text)
for entry in predict_attributes(nlp2,doc):
    print(entry)
text = "I feel pain in my stomach that gets worse by the day."
doc = nlp2(text)
print(text)
for entry in predict_attributes(nlp2, doc):
    print(entry)



