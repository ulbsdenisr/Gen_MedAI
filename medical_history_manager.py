import sqlite3
from datetime import datetime
from symptom_normalizer import SymptomNormalizer


class MedicalHistoryManager:

    def __init__(self, db_path="chat_history.db", threshold=0.5):
        self.db_path = db_path
        self.normalizer = SymptomNormalizer(threshold=threshold)
        self._create_table()

    # -----------------------------
    # DB Setup
    # -----------------------------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            conversation_id TEXT,
            timestamp TEXT,
            symptom TEXT,
            severity TEXT,
            status TEXT
        )
        """)

        conn.commit()
        conn.close()

    # -----------------------------
    # Load History (per chat!)
    # -----------------------------
    def _load_conversation_history(self, user_id, conversation_id):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT timestamp, symptom, severity, status
        FROM medical_history
        WHERE user_id = ? AND conversation_id = ?
        ORDER BY timestamp ASC
        """, (user_id, conversation_id))

        rows = cursor.fetchall()
        conn.close()

        history = {}

        for ts, symptom, severity, status in rows:
            if ts not in history:
                history[ts] = []

            history[ts].append({
                "symptom": symptom,
                "severity": severity,
                "status": status
            })

        return {
            "history": [
                {"timestamp": ts, "entries": entries}
                for ts, entries in sorted(history.items())
            ]
        }

    # -----------------------------
    # Main Entry Point
    # -----------------------------
    def append_to_history(self, entries, user_id, conversation_id):
        if user_id is None or conversation_id is None:
            return []  # silently ignore OR raise error
        conn = self._connect()
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        processed_entries = []

        # 🔹 Normalize + clean
        for entry in entries:
            raw_symptom = entry["symptom"]
            severity = entry["severity"]
            status = entry["status"]

            if raw_symptom.lower().startswith(severity.lower()):
                raw_symptom = raw_symptom[len(severity):].strip()

            normalized = self.normalizer.normalize_if_certain(raw_symptom)

            if normalized:
                processed_entries.append({
                    "symptom": normalized["normalized"],
                    "severity": severity,
                    "status": status
                })

        # 🔥 1. LOAD OLD HISTORY (same user + same chat)
        history_data = self._load_conversation_history(user_id, conversation_id)

        # 🔥 2. CHECK WARNINGS
        warnings = self.check_all_concerns(history_data, processed_entries)

        # 🔥 3. INSERT NEW DATA
        for entry in processed_entries:
            cursor.execute("""
            INSERT INTO medical_history (user_id, conversation_id, timestamp, symptom, severity, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                conversation_id,
                timestamp,
                entry["symptom"],
                entry["severity"],
                entry["status"]
            ))

        conn.commit()
        conn.close()

        return warnings

    # -----------------------------
    # Concern Logic
    # -----------------------------
    def check_all_concerns(self, history_data, current_entries):

        warnings = []

        checks = [
            self._check_persistent_severe,
            self._check_worsening_trend,
            self._check_new_severe_symptom,
            self._check_many_simultaneous_symptoms,
            self._check_reappearing_symptom
        ]

        for check in checks:
            result = check(history_data, current_entries)
            if result:
                if isinstance(result, list):
                    warnings.extend(result)
                else:
                    warnings.append(result)

        if len(warnings) > 0:
            warnings.append("Please consider checking in with a professional!")

        return warnings

    def _check_persistent_severe(self, history_data, current_entries):

        history = history_data.get("history", [])

        if len(history) < 2:
            return None

        last = history[-1]
        second_last = history[-2]

        def lookup(visit):
            return {e["symptom"]: e for e in visit.get("entries", [])}

        last_lookup = lookup(last)
        second_lookup = lookup(second_last)

        warnings = []

        for current in current_entries:

            symptom = current["symptom"]
            severity = current["severity"].lower()
            status = current["status"].lower()

            if severity != "severe" and status != "worsening":
                continue

            prev1 = last_lookup.get(symptom)
            prev2 = second_lookup.get(symptom)

            if not prev1 or not prev2:
                continue

            if (
                prev1["severity"].lower() == "severe"
                or prev1["status"].lower() == "worsening"
            ) and (
                prev2["severity"].lower() == "severe"
                or prev2["status"].lower() == "worsening"
            ):
                warnings.append(
                    f"The '{symptom}' symptom has been severe or worsening."
                )

        return warnings

    def _check_worsening_trend(self, history_data, current_entries):

        severity_rank = {
            "mild": 1,
            "moderate": 2,
            "severe": 3
        }

        history = history_data.get("history", [])

        if len(history) < 1:
            return None

        last = history[-1]
        last_lookup = {e["symptom"]: e for e in last.get("entries", [])}

        warnings = []

        for current in current_entries:

            symptom = current["symptom"]
            prev = last_lookup.get(symptom)

            if not prev:
                continue

            current_rank = severity_rank.get(current["severity"].lower(), 0)
            prev_rank = severity_rank.get(prev["severity"].lower(), 0)

            if current_rank > prev_rank:
                warnings.append(
                    f"The severity of '{symptom}' appears to be worsening."
                )

        return warnings

    def _check_new_severe_symptom(self, history_data, current_entries):

        history = history_data.get("history", [])
        seen = set()

        for visit in history:
            for entry in visit.get("entries", []):
                seen.add(entry["symptom"])

        warnings = []

        for current in current_entries:
            if (
                current["symptom"] not in seen
                and current["severity"].lower() == "severe"
            ):
                warnings.append(
                    f"A new severe symptom was reported: '{current['symptom']}'."
                )

        return warnings

    def _check_many_simultaneous_symptoms(self, history_data, current_entries):

        if len(current_entries) >= 4:
            return "Multiple symptoms reported simultaneously."

        return None

    def _check_reappearing_symptom(self, history_data, current_entries):

        history = history_data.get("history", [])

        warnings = []

        past_symptoms = set()
        for visit in history[:-1]:
            for entry in visit.get("entries", []):
                past_symptoms.add(entry["symptom"])

        last_symptoms = {
            e["symptom"]
            for e in history[-1].get("entries", [])
        } if history else set()

        for current in current_entries:
            symptom = current["symptom"]

            if symptom in past_symptoms and symptom not in last_symptoms:
                warnings.append(
                    f"The symptom '{symptom}' has reappeared after being absent."
                )

        return warnings

    def export_chat_history_json(self, user_id, conversation_id, output_dir="medical_exports"):
        import os, json

        os.makedirs(output_dir, exist_ok=True)

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, symptom, severity, status
            FROM medical_history
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY timestamp ASC
        """, (user_id, conversation_id))

        rows = cursor.fetchall()
        conn.close()

        history = {}

        for ts, symptom, severity, status in rows:
            if ts not in history:
                history[ts] = []

            history[ts].append({
                "symptom": symptom,
                "severity": severity,
                "status": status
            })

        # convert to structured list (better for frontend timeline)
        structured = [
            {
                "timestamp": ts,
                "entries": entries
            }
            for ts, entries in history.items()
        ]

        file_path = os.path.join(output_dir, f"{user_id}_{conversation_id}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_id": user_id,
                "conversation_id": conversation_id,
                "history": structured
            }, f, indent=2)

        return file_path

    def export_json_to_pdf(self, json_path, output_path="history.pdf"):
        import json
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib.pagesizes import A4

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        title = styles["Heading1"]
        normal = styles["Normal"]

        elements.append(Paragraph("Medical History Report", title))
        elements.append(Spacer(1, 0.3 * inch))

        history = data.get("history", [])

        if not history:
            elements.append(Paragraph("No history available.", normal))
        else:
            for visit in history:
                timestamp = visit.get("timestamp", "Unknown")

                elements.append(Paragraph(f"<b>Date:</b> {timestamp}", styles["Heading3"]))
                elements.append(Spacer(1, 0.2 * inch))

                bullets = []

                for entry in visit.get("entries", []):
                    text = f"{entry['symptom']} (Severity: {entry['severity']}, Status: {entry['status']})"
                    bullets.append(ListItem(Paragraph(text, normal)))

                elements.append(ListFlowable(bullets))
                elements.append(Spacer(1, 0.4 * inch))

        doc.build(elements)

        return output_path