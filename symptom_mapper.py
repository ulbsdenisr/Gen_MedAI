import re

# reguli rapide (poți extinde)
REWRITE = [
    (r"\bhigh fever\b", "fever"),
    (r"\bfeverish\b", "fever"),
    (r"\bsevere headache\b", "headache"),
    (r"\bhead pain\b", "headache"),
    (r"\bcoughing\b", "cough"),
    (r"\bthroat is sore\b", "sore throat"),
]

def canonicalize(symptom: str) -> str:
    s = symptom.strip().lower()
    s = s.strip(" .,!?:;\"'()[]{}")
    for pat, rep in REWRITE:
        s = re.sub(pat, rep, s)
    # normalize adjectives comune (severe/mild/high etc.)
    s = re.sub(r"\b(severe|mild|moderate|high|low|extreme)\b\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_list(symptoms):
    out = []
    for s in symptoms:
        cs = canonicalize(s)
        if cs:
            out.append(cs)
    # remove duplicates keep order
    return list(dict.fromkeys(out))
