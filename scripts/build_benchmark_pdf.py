"""Renders the 200-question benchmark into a paginated, reviewer-friendly PDF."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def footer(canvas, document) -> None:
    """Draws release identity and page numbering consistently on every PDF page."""

    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#51606d"))
    canvas.drawString(0.65 * inch, 0.45 * inch, "Evidence-Grounded AI Governance Copilot-RAG | Corpus governance-security-expanded-2026-07-30-r2")
    canvas.drawRightString(7.85 * inch, 0.45 * inch, f"Page {document.page}")
    canvas.restoreState()


def build() -> Path:
    """Creates the final PDF from the committed JSONL benchmark without external dependencies."""

    output = ROOT / "artifacts/benchmark"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "ai_governance_rag_benchmark_200.pdf"
    rows = [json.loads(line) for line in (ROOT / "data/evaluation/benchmark_200.jsonl").read_text().splitlines()]
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#123047"), alignment=TA_CENTER)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=4)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#075985"), spaceBefore=8, spaceAfter=6)
    story = [Spacer(1, 1.2 * inch), Paragraph("AI Governance RAG Benchmark", title), Spacer(1, 0.12 * inch), Paragraph("200 versioned evaluation cases with reference answers and source-level evidence mapping", ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=10, leading=13)), Spacer(1, 0.3 * inch)]
    # These categories deliberately mix direct, comparative, boundary, and refusal decisions
    # so the quality gate cannot be passed by a system tuned only for easy lookup questions.
    summary = [["Category", "Count", "Purpose"], ["Direct control", "70", "Ground factual answers"], ["Multi-document", "70", "Evidence synthesis"], ["Scope boundary", "40", "Confusable concepts"], ["Adversarial abstention", "20", "Safe refusal"]]
    table = Table(summary, colWidths=[1.45 * inch, .7 * inch, 3.8 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#075985")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#d5dce2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fc")]), ("PADDING", (0, 0), (-1, -1), 7)]))
    story += [table, Spacer(1, .35 * inch), Paragraph("Evaluation policy", heading), Paragraph("A release fails if faithfulness is below 0.95, citation validity is below 1.00, answer correctness is below 0.85, abstention F1 is below 0.90, or an approved baseline regresses by more than 0.02.", body), PageBreak()]
    for index, row in enumerate(rows):
        # One scenario per page keeps long paragraph prompts, references, and evidence excerpts
        # reviewable without clipping. It also preserves the programme requirement that every
        # generated benchmark PDF is a substantial (90+ page) review artifact.
        story.append(Paragraph(f"{escape(row['id'])}  |  {escape(row['category'].replace('_', ' ').title())}", heading))
        # Evidence originates in PDFs and can contain angle brackets or ampersands. Escape
        # every dynamic field before handing it to ReportLab's HTML-like paragraph parser.
        story.append(Paragraph(f"<b>Question:</b> {escape(row['question'])}", body))
        story.append(Paragraph(f"<b>Reference answer:</b> {escape(row['reference_answer'])}", body))
        if row.get("gold_chunk_ids"):
            story.append(Paragraph(f"<b>Gold chunks:</b> {escape(', '.join(row['gold_chunk_ids']))}", body))
        quotes = row.get("evidence_quotes") or ([row['evidence_quote']] if row.get("evidence_quote") else [])
        if quotes:
            story.append(Paragraph(f"<b>Evidence quote:</b> {escape(' | '.join(quotes))}", body))
        story.append(Spacer(1, 6))
        if index < len(rows) - 1:
            story.append(PageBreak())
    document = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=.65 * inch, rightMargin=.65 * inch, topMargin=.65 * inch, bottomMargin=.65 * inch)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


if __name__ == "__main__":
    print(build())
