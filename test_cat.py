from head_training import predict_attributes
import spacy
nlp2 = spacy.load("model/model_with_textcats")


text="I've been experiencing mild joint pain and extreme fatigue and now I also have a headache."
doc = nlp2(text)
print(text)
print(doc.cats)
print(doc.ents)
for entry in predict_attributes(nlp2,doc):
    print(entry)

print()

text="I have a bad stomach ache and it's not getting better."
doc = nlp2(text)
print(text)
for entry in predict_attributes(nlp2,doc):
    print(entry)

