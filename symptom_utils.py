import re

SPLIT_PATTERN = r",|\band\b|\bwith\b|\bplus\b|\baccompanied by\b"


def clean_piece(text: str) -> str:
    text = str(text).strip().lower()

    # elimina inceputuri inutile (extins cu variante comune)
    text = re.sub(
        r"^(i have|i am having|i'm having|ive got|i got"
        r"|i feel|i am feeling|i've been having|i've been feeling"
        r"|i am|i'm|i experience|i notice|i noticed"
        r"|also have|also feel|also having|also feeling"
        r"|also experience|as well as|along with"
        r"|there is|there's|there are)\s+",
        "", text
    )

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