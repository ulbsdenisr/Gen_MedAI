from datetime import datetime
import json
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

from symptom_normalizer import SymptomNormalizer


class HistoryManager:

    def __init__(self,
                 filename="history.json",
                 threshold=0.5):

        self.filename = filename
        self.normalizer = SymptomNormalizer(threshold=threshold)

        # Ensure file exists
        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump({"history": []}, f, indent=4)

    # -----------------------------
    # Internal Helpers
    # -----------------------------

    def _load_history(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"history": []}

    def _save_history(self, data):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # -----------------------------
    # Public Methods
    # -----------------------------

    def append_to_history(self, entries):
        """
        Normalizes entries, checks concerns,
        then appends to history.
        """

        data = self._load_history()
        timestamp = datetime.now().isoformat()

        processed_entries = []

        for entry in entries:
            raw_symptom = entry["symptom"]
            severity = entry["severity"]
            status = entry["status"]

            # Remove severity prefix if present
            if raw_symptom.lower().startswith(severity.lower()):
                raw_symptom = raw_symptom[len(severity):].strip()

            normalized = self.normalizer.normalize_if_certain(raw_symptom)

            if normalized:
                processed_entries.append({
                    "symptom": normalized["normalized"],
                    "severity": severity,
                    "status": status
                })

        # Check concerns BEFORE appending
        self._check_recent_concerns(data, processed_entries)

        # Append visit
        data["history"].append({
            "timestamp": timestamp,
            "entries": processed_entries
        })

        self._save_history(data)

    def export_to_pdf(self, output_path="history.pdf"):
        """
        Exports current history to a PDF.
        """

        data = self._load_history()

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        normal_style = styles["Normal"]

        elements.append(Paragraph("Symptom History Report", title_style))
        elements.append(Spacer(1, 0.3 * inch))

        history = data.get("history", [])

        if not history:
            elements.append(Paragraph("No history available.", normal_style))
        else:
            for visit in history:
                timestamp = visit.get("timestamp", "Unknown date")

                elements.append(
                    Paragraph(f"<b>Date:</b> {timestamp}", styles["Heading3"])
                )
                elements.append(Spacer(1, 0.2 * inch))

                bullet_points = []

                for entry in visit.get("entries", []):
                    symptom = entry.get("symptom", "Unknown")
                    severity = entry.get("severity", "Unknown")
                    status = entry.get("status", "Unknown")

                    text = f"{symptom} (Severity: {severity}, Status: {status})"

                    bullet_points.append(
                        ListItem(Paragraph(text, normal_style))
                    )

                elements.append(ListFlowable(bullet_points, bulletType="bullet"))
                elements.append(Spacer(1, 0.4 * inch))

        doc.build(elements)
        print(f"PDF exported to {output_path}")

    # -----------------------------
    # Concern Logic
    # -----------------------------

    def _check_recent_concerns(self, history_data, current_entries):
        """
        Triggers warning if:
        - Current symptom is severe or worsening
        - AND it was also severe/worsening in the last 2 visits
        """

        history = history_data.get("history", [])

        if len(history) < 2:
            return

        last_visit = history[-1]
        second_last_visit = history[-2]

        def build_lookup(visit):
            return {
                entry["symptom"]: entry
                for entry in visit.get("entries", [])
            }

        last_lookup = build_lookup(last_visit)
        second_last_lookup = build_lookup(second_last_visit)

        for current in current_entries:
            symptom = current["symptom"]
            severity = current["severity"].lower()
            status = current["status"].lower()

            current_concerning = (
                severity == "severe" or status == "worsening"
            )

            if not current_concerning:
                continue

            last_entry = last_lookup.get(symptom)
            second_last_entry = second_last_lookup.get(symptom)

            if not last_entry or not second_last_entry:
                continue

            last_concerning = (
                last_entry["severity"].lower() == "severe" or
                last_entry["status"].lower() == "worsening"
            )

            second_last_concerning = (
                second_last_entry["severity"].lower() == "severe" or
                second_last_entry["status"].lower() == "worsening"
            )

            if last_concerning and second_last_concerning:
                print(
                    f"\n⚠ THE {symptom.upper()} SYMPTOM RAISES CONCERNS.\n"
                    "PLEASE CHECK IN WITH A PROFESSIONAL.\n"
                )
