from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(result: dict, symptoms: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MediMind AI Medical Summary", styles["Title"]),
        Paragraph("This report is for educational support only and is not a medical diagnosis.", styles["BodyText"]),
        Spacer(1, 16),
    ]
    table = Table(
        [
            ["Symptoms", symptoms],
            ["Predicted condition", result.get("disease", "Unknown")],
            ["Confidence", f"{int(result.get('confidence', 0) * 100)}%"],
            ["Urgency", result.get("urgency", "Unknown")],
        ],
        colWidths=[130, 330],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#22c6d8")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7e4ea")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story += [table, Spacer(1, 16), Paragraph("Overview", styles["Heading2"]), Paragraph(result.get("description", ""), styles["BodyText"])]
    story += [Spacer(1, 12), Paragraph("Precautions", styles["Heading2"])]
    for item in result.get("precautions", []):
        story.append(Paragraph(f"- {item}", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
