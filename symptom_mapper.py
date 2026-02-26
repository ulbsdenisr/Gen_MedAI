import re
from typing import List, Optional, Tuple

from symptom_semantic_mapper import SemanticMapper

REWRITE = [
    (r"\bhigh fever\b", "fever"),
    (r"\bfeverish\b", "fever"),
    (r"\bsevere headache\b", "headache"),
    (r"\bhead pain\b", "headache"),
    (r"\bcoughing\b", "cough"),
    (r"\bthroat is sore\b", "sore throat"),
    (r"\bpersistent dry cough\b", "cough"),
    (r"\bdry cough\b", "cough"),
    (r"\bshortness of breath when walking\b", "shortness of breath"),
    (r"\bdyspnea on exertion\b", "shortness of breath"),
]

_STOP_PREFIX = r"\b(including|with|having|i have|ive|i've|i am|im|i'm)\b\s*"


def canonicalize(symptom: str) -> str:
    s = symptom.strip().lower()
    s = s.strip(" .,!?:;\"'()[]{}")
    s = re.sub(_STOP_PREFIX, "", s)

    for pat, rep in REWRITE:
        s = re.sub(pat, rep, s)

    # scoate intensitati (dar pastreaza esenta)
    s = re.sub(r"\b(severe|mild|moderate|high|low|extreme)\b\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _should_try_semantic(s: str) -> bool:
    """
    Heuristic: semantic mapping are sens pentru:
      - expresii (au spatiu)
      - termeni mai lungi
      - abrevieri/coduri (ex: adhd) -> optional, dar util
    """
    if not s:
        return False
    if " " in s:
        return True
    if len(s) >= 8:       # ex: "restless", "anorexia"
        return True
    if s.isalpha() and 3 <= len(s) <= 6:
        # ex: "adhd", "ptsd" etc. (optional dar foarte util)
        return True
    return False


def canonicalize_list(
    symptoms: List[str],
    semantic: bool = False,
    mapper: Optional[SemanticMapper] = None,
    *,
    keep_original: bool = True,
) -> List[str]:
    """
    Best version:
    - curata + dedupe
    - semantic map cu fallback (nu mai arunca simptome!)
    - optional: pastreaza si originalul + mapped (creste recall)
    """
    # 1) basic canonicalize
    out: List[str] = []
    for s in symptoms:
        cs = canonicalize(s)
        if cs:
            out.append(cs)
    out = list(dict.fromkeys(out))  # dedupe pastrând ordinea

    if not semantic or not out:
        return out

    mapper = mapper or SemanticMapper()

    final: List[str] = []
    for s in out:
        if not _should_try_semantic(s):
            final.append(s)
            continue

        ms, score = mapper.map_one(s)
        ms = canonicalize(ms)

        # 2) fallback robust
        if not ms:
            final.append(s)
            continue

        if ms == s:
            # n-a gasit nimic clar: pastreaza originalul
            final.append(s)
            continue

        # 3) alegere "best": pastreaza mapped
        final.append(ms)

        # 4) optional: pastreaza si originalul ca sa nu pierzi recall (recomandat)
        if keep_original:
            final.append(s)

    # curata din nou + dedupe
    final = [canonicalize(x) for x in final]
    final = [x for x in final if x]
    return list(dict.fromkeys(final))
