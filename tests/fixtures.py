"""Fixture builders producing genuine PDF and DOCX bytes.

The parsers are tested against real files, not stubs. A mocked pdfplumber
would prove only that the mock behaves as I imagined; font-size heading
detection and page boundaries only mean anything against a real render.
"""

from __future__ import annotations

import io

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BODY_SIZE = 10.0
H1_SIZE = 18.0
H2_SIZE = 14.0


def build_pdf(pages: list[list[tuple[str, float, bool]]]) -> bytes:
    """Render pages of (text, font_size, bold) lines to PDF bytes."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    for page in pages:
        y = height - 60
        for text, size, bold in page:
            pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            # Wrap long lines so the renderer does not silently clip them.
            limit = max(20, int((width - 120) / (size * 0.5)))
            words, line = text.split(), ""
            for word in words:
                if len(line) + len(word) + 1 > limit:
                    pdf.drawString(60, y, line)
                    y -= size * 1.35
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                pdf.drawString(60, y, line)
            y -= size * 2.0
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def security_policy_pdf() -> bytes:
    """A three-page policy with numbered sections, an exception clause that
    must stay with its rule, a repeated heading title, and an empty page."""
    body = BODY_SIZE
    return build_pdf(
        [
            [
                ("Acme Corp Information Security Policy", H1_SIZE, True),
                ("Version 3.2", body, False),
                ("Effective Date: 14 March 2024", body, False),
                ("This policy defines the security controls Acme Corp applies to "
                 "customer data and supporting systems.", body, False),
            ],
            [
                ("4. Access Control", H2_SIZE, True),
                ("4.1 Authentication", H2_SIZE, False),
                ("Multi-factor authentication is required for all employee accounts "
                 "that access production systems. Authentication uses SSO backed by "
                 "the corporate identity provider.", body, False),
                ("4.2 Exceptions", H2_SIZE, False),
                ("Break-glass service accounts are exempt from MFA and are instead "
                 "protected by hardware tokens held in a sealed safe. Use of a "
                 "break-glass account triggers an alert to the security team.", body, False),
            ],
            [
                ("5. Encryption", H2_SIZE, True),
                ("5.1 Authentication", H2_SIZE, False),
                ("Customer data is encrypted at rest using AES-256 and in transit "
                 "using TLS 1.2 or higher. Encryption keys are managed in AWS KMS "
                 "and rotated annually.", body, False),
            ],
        ]
    )


def build_docx(items: list[tuple[str, str]], tables: list[list[list[str]]] | None = None) -> bytes:
    """Build DOCX bytes from (style, text) pairs, appending any tables."""
    document = Document()
    for style, text in items:
        if style == "table":
            continue
        document.add_paragraph(text, style=style or None)
    for table_rows in tables or []:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def security_policy_docx() -> bytes:
    return build_docx(
        [
            ("Title", "Acme Corp Data Protection Policy"),
            (None, "Last Updated: 2024-06-01"),
            ("Heading 1", "1. Data Retention"),
            (None, "Customer data is retained for 90 days after account closure."),
            ("Heading 2", "1.1 Deletion Requests"),
            (None, "Deletion requests are fulfilled within 30 days of receipt."),
        ],
        tables=[
            [["Control", "Status"], ["Encryption at rest", "Implemented"],
             ["Penetration testing", "Annual"]],
        ],
    )
