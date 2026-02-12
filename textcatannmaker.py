import re
import json

# Severity regex rules
SEVERITY_RULES = {
    "severe": re.compile(r"\b(severe|intense|extreme|unbearable|terrible|awful|really|a lot)\b", re.I),
    "moderate": re.compile(r"\b(moderate|noticeable|considerable|significant)\b", re.I),
    "mild": re.compile(r"\b(mild|slight|minor|a bit|light)\b", re.I),
}

# Status/State regex rules
STATE_RULES = {
    "resolved": re.compile(r"\b(no longer|gone|resolved|disappeared|stopped|cleared up)\b", re.I),
    "worsening": re.compile(r"\b(getting worse|worsening|worse|increased|more intense)\b", re.I),
    "improving": re.compile(r"\b(better|improving|eased|less severe|not as bad)\b", re.I),
    "new": re.compile(r"\b(started|now have|also have|recently developed|began having|just started)\b", re.I),
    "unchanged": re.compile(r"\b(unchanged|same|no change|still|remains)\b", re.I),
}

def rule_label_regex(span_text, full_text=None, window=15):
    text = span_text.lower()
    severity = None
    status = None

    #cauta mai intai cuvintele din regex
    for sev, pattern in SEVERITY_RULES.items():
        if pattern.search(text):
            severity = sev
            break

    for stat, pattern in STATE_RULES.items():
        if pattern.search(text):
            status = stat
            break

    # daca nu au fost gasite cuvinte din regex, verifica contextul
    if (severity is None or status is None) and full_text is not None:
        start_idx = full_text.lower().find(span_text.lower())
        if start_idx >= 0:
            #construieste contextul (=+/- 15 caractere in jurul spanului)
            context = full_text[max(0, start_idx-window): start_idx+len(span_text)+window].lower()
            if severity is None:
                for sev, pattern in SEVERITY_RULES.items():
                    if pattern.search(context):
                        severity = sev
                        break
            if status is None:
                for stat, pattern in STATE_RULES.items():
                    if pattern.search(context):
                        status = stat
                        break

    # Default to "unknown" if still nothing found
    # Default values if nothing found
    if severity is None:
        severity = "moderate"
    if status is None:
        status = "unchanged"
    return severity, status




with open("annotations_final4.json", "r") as f:
    data = json.load(f)

train_data_severity = []
train_data_status = []

for item in data["annotations"]:
    text, annots = item
    for start, end, label in annots["entities"]:
        if label != "SYMPTOM":
            continue
        span_text = text[start:end]

        severity, status = rule_label_regex(span_text, full_text=text)
        train_data_severity.append(
                (span_text, {"cats": {k: int(k==severity) for k in ["mild","moderate","severe"]}})
        )
        train_data_status.append(
                (span_text, {"cats": {k: int(k==status) for k in ["new","worsening","improving","unchanged","resolved"]}})
        )

print("Severity examples:", train_data_severity)
print("Status examples:", train_data_status)
import json

# Save severity examples
with open("train_data_severity.json", "w", encoding="utf-8") as f:
    json.dump(train_data_severity, f, ensure_ascii=False, indent=2)

# Save status examples
with open("train_data_status.json", "w", encoding="utf-8") as f:
    json.dump(train_data_status, f, ensure_ascii=False, indent=2)
