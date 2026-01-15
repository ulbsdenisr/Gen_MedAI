import re

SPLIT_PATTERN = r",| and | with | plus | accompanied by "

def normalize_and_split_symptoms(entities):
    results = []
    for ent in entities:
        parts = re.split(SPLIT_PATTERN, ent, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip(" .,!?:;")
            if len(p) > 2:
                results.append(p)
    return list(dict.fromkeys(results))
