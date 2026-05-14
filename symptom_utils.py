import re

SPLIT_PATTERN = r",|\band\b|\bwith\b|\bplus\b|\baccompanied by\b"


def clean_piece(text: str) -> str:
    text = str(text).strip().lower()

    # elimina inceputuri inutile
    text = re.sub(r"^(i have|i am having|i'm having|ive got|i got)\s+", "", text)

    # normalizeaza spatii
    text = re.sub(r"\s+", " ", text)

    return text.strip(" .,!?:;")


def normalize_and_split_symptoms(entities):
    results = []

    for ent in entities:
        if not ent:
            continue

        ent = clean_piece(ent)

        parts = re.split(SPLIT_PATTERN, ent, flags=re.IGNORECASE)

        for p in parts:
            p = clean_piece(p)

            if len(p) > 2:
                results.append(p)

    # dedupe pastrand ordinea
    return list(dict.fromkeys(results))