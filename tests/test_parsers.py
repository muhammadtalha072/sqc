"""Parser tests. Real files, real edge cases."""

from __future__ import annotations

from datetime import date

import pytest

from sqc.core.ingestion.parsers import (
    DocumentParseError,
    UnsupportedFormatError,
    parse_bytes,
    parse_docx,
    parse_pdf,
    parse_txt,
)
from sqc.core.ingestion.structure import (
    extract_effective_date,
    extract_version_label,
    looks_like_heading,
    numbering_level,
)
from tests.fixtures import (
    BODY_SIZE,
    H1_SIZE,
    H2_SIZE,
    build_docx,
    build_pdf,
    security_policy_docx,
    security_policy_pdf,
)


# ------------------------------------------------------------------- headings


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4 Access Control", 1),
        ("4.2 Authentication", 2),
        ("4.2.1 Password Rules", 3),
        ("Appendix B", 1),
        ("Section 12 Incident Response", 1),
        ("2024. was a difficult year", None),
        ("We encrypt data at rest.", None),
        ("", None),
    ],
)
def test_numbering_level(text, expected):
    assert numbering_level(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ENCRYPTION AT REST", True),
        ("Encryption at Rest", True),
        ("Incident Response Plan", True),
        ("All customer data is encrypted at rest.", False),
        ("Our approach to encryption, which covers both at rest and in transit, "
         "is documented below", False),
    ],
)
def test_looks_like_heading(text, expected):
    assert looks_like_heading(text) is expected


# ----------------------------------------------------------------------- txt


def test_txt_markdown_headings_and_paragraphs():
    doc = parse_txt(
        b"# Security Policy\n\n## Encryption\n\nData is encrypted with AES-256.\n\n"
        b"- Keys rotate annually\n",
        "policy.md",
    )
    kinds = [(b.kind, b.level, b.text) for b in doc.blocks]
    assert ("heading", 1, "Security Policy") in kinds
    assert ("heading", 2, "Encryption") in kinds
    assert any(k == "paragraph" and "AES-256" in t for k, _, t in kinds)
    assert any(k == "list_item" for k, _, _ in kinds)
    assert all(b.page is None for b in doc.blocks), "txt has no pages; must not invent them"


def test_txt_without_any_headings_still_parses():
    doc = parse_txt(b"We encrypt data at rest.\n\nWe also encrypt it in transit.", "flat.txt")
    assert [b.kind for b in doc.blocks] == ["paragraph", "paragraph"]
    assert all(b.level is None for b in doc.blocks)


def test_empty_file_raises():
    with pytest.raises(DocumentParseError):
        parse_bytes(b"", "empty.txt")


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFormatError):
        parse_bytes(b"data", "questionnaire.xlsx")


def test_malformed_pdf_raises_parse_error():
    with pytest.raises(DocumentParseError):
        parse_pdf(b"%PDF-1.4 this is not really a pdf", "broken.pdf")


def test_malformed_docx_raises_parse_error():
    with pytest.raises(DocumentParseError):
        parse_docx(b"PK\x03\x04 not a real docx", "broken.docx")


# ---------------------------------------------------------------------- docx


def test_docx_uses_styles_and_extracts_tables():
    doc = parse_docx(security_policy_docx(), "dp.docx")
    headings = [(b.level, b.text) for b in doc.blocks if b.kind == "heading"]
    assert (1, "Acme Corp Data Protection Policy") in headings
    assert (1, "1. Data Retention") in headings
    assert (2, "1.1 Deletion Requests") in headings

    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "Encryption at rest | Implemented" in tables[0].text

    assert doc.effective_date == date(2024, 6, 1)
    assert all(b.page is None for b in doc.blocks)
    assert any("page" in w for w in doc.warnings)


def test_docx_with_no_headings_produces_only_paragraphs():
    doc = parse_docx(build_docx([(None, "We rotate keys annually.")]), "flat.docx")
    assert [b.kind for b in doc.blocks] == ["paragraph"]


# ----------------------------------------------------------------------- pdf


def test_pdf_detects_headings_and_real_page_numbers():
    doc = parse_pdf(security_policy_pdf(), "policy.pdf")
    assert doc.page_count == 3

    headings = [(b.level, b.text, b.page) for b in doc.blocks if b.kind == "heading"]
    titles = [t for _, t, _ in headings]
    assert "Acme Corp Information Security Policy" in titles
    assert "4. Access Control" in titles
    assert "4.2 Exceptions" in titles

    # Numbering wins over font size: 4.2 is level 2 even though it is
    # rendered at the same size as the level-1 style elsewhere.
    assert dict((t, lv) for lv, t, _ in headings)["4.2 Exceptions"] == 2

    pages = {t: p for _, t, p in headings}
    assert pages["4. Access Control"] == 2
    assert pages["5. Encryption"] == 3
    assert all(b.page is not None for b in doc.blocks), "pdf pages are known and must be kept"


def test_pdf_extracts_labelled_metadata():
    doc = parse_pdf(security_policy_pdf(), "policy.pdf")
    assert doc.effective_date == date(2024, 3, 14)
    assert doc.version_label == "3.2"


def test_pdf_empty_page_is_warned_not_invented():
    raw = build_pdf([[("Only page with text", BODY_SIZE, False)], []])
    doc = parse_pdf(raw, "gap.pdf")
    assert doc.page_count == 2
    assert any("no extractable text" in w for w in doc.warnings)


def test_pdf_paragraph_spanning_pages_keeps_its_own_page():
    raw = build_pdf(
        [
            [("Retention rules are defined below.", BODY_SIZE, False)],
            [("Backups are retained for thirty five days.", BODY_SIZE, False)],
        ]
    )
    doc = parse_pdf(raw, "split.pdf")
    paragraphs = [b for b in doc.blocks if b.kind == "paragraph"]
    assert {b.page for b in paragraphs} == {1, 2}


def test_pdf_with_no_text_layer_warns_about_ocr():
    doc = parse_pdf(build_pdf([[], []]), "scanned.pdf")
    assert doc.blocks == ()
    assert any("OCR" in w for w in doc.warnings)


def test_pdf_all_body_text_does_not_invent_headings():
    raw = build_pdf([[(f"Sentence number {i} about encryption keys.", BODY_SIZE, False)
                      for i in range(6)]])
    doc = parse_pdf(raw, "flat.pdf")
    assert not [b for b in doc.blocks if b.kind == "heading"]


def test_pdf_body_size_wins_when_headings_outnumber_body_lines():
    """Weighting by characters, not lines, keeps a heading size from being
    elected as the body size in a heading-dense document."""
    lines = [(f"Heading Number {i}", H1_SIZE, True) for i in range(5)]
    lines.append(
        ("This single body paragraph carries far more characters than all of the "
         "headings above it combined, which is exactly why the body size must be "
         "weighted by character count rather than by how many lines use it.",
         BODY_SIZE, False)
    )
    doc = parse_pdf(build_pdf([lines]), "dense.pdf")
    assert len([b for b in doc.blocks if b.kind == "heading"]) == 5


# ------------------------------------------------------------------ metadata


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Effective Date: 14 March 2024", date(2024, 3, 14)),
        ("Last Updated: March 14, 2024", date(2024, 3, 14)),
        ("Revised: 2024-03-14", date(2024, 3, 14)),
        ("Effective: 25/03/2024", date(2024, 3, 25)),
        ("Approved on 1st September 2023", date(2023, 9, 1)),
    ],
)
def test_effective_date_extraction(text, expected):
    found, _ = extract_effective_date(text)
    assert found == expected


def test_ambiguous_numeric_date_is_refused_with_warning():
    found, warnings = extract_effective_date("Effective: 03/04/2024")
    assert found is None, "03/04/2024 could be 3 April or 4 March; must not guess"
    assert any("ambiguous" in w for w in warnings)


def test_unlabelled_date_is_ignored():
    found, _ = extract_effective_date("The incident occurred on 14 March 2024.")
    assert found is None


def test_document_with_no_date_has_none():
    doc = parse_txt(b"# Policy\n\nWe encrypt everything.", "nodate.txt")
    assert doc.effective_date is None
    assert doc.version_label is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Version 3.2", "3.2"), ("v1.0.4", "1.0.4"), ("Rev. 7", "7"), ("no version here", None)],
)
def test_version_extraction(text, expected):
    assert extract_version_label(text) == expected


def test_content_hash_is_stable_and_distinct():
    a = parse_txt(b"same bytes", "a.txt")
    b = parse_txt(b"same bytes", "b.txt")
    c = parse_txt(b"other bytes", "c.txt")
    assert a.content_sha256 == b.content_sha256
    assert a.content_sha256 != c.content_sha256


def test_pdf_sentence_continuing_across_page_break_cites_both_pages():
    """The other half of the page-boundary rule: an unfinished sentence does
    continue onto the next page, and the block must then claim both pages."""
    raw = build_pdf(
        [
            [("Customer data is encrypted at rest using AES-256 and keys are", BODY_SIZE, False)],
            [("rotated annually by the platform team.", BODY_SIZE, False)],
        ]
    )
    doc = parse_pdf(raw, "cont.pdf")
    paragraphs = [b for b in doc.blocks if b.kind == "paragraph"]
    assert len(paragraphs) == 1, "an unfinished sentence should not be split at the page break"
    assert paragraphs[0].pages == (1, 2)
    assert "rotated annually" in paragraphs[0].text


def test_block_rejects_impossible_page_range():
    from sqc.core.ingestion.model import Block

    with pytest.raises(ValueError, match="precedes"):
        Block(text="x", page=5, page_end=2)


# ------------------------------------------------- real-world layout regression


def test_numbered_list_items_in_prose_are_not_headings():
    """Regression from a real university policy. Obligations enumerated as
    '2. Report all breaches ... devices. Such' were promoted to headings,
    and because a bare '2.' reads as depth 1 they reset the heading stack,
    filing the text that followed under the wrong section."""
    from tests.fixtures import lettered_policy_pdf

    doc = parse_pdf(lettered_policy_pdf(), "lettered.pdf")
    headings = [b.text for b in doc.blocks if b.kind == "heading"]

    assert not [h for h in headings if h.startswith(("1.", "2.", "3."))], (
        f"list items promoted to headings: {headings}"
    )
    assert "A. DEFINITIONS" in headings
    assert "B. COMMUNITY MEMBER RESPONSIBILITIES" in headings
    assert "C. RESPONSIBLE OFFICER RESPONSIBILITIES" in headings


def test_running_header_and_footer_are_dropped():
    """Left in place, a running header is detected as a heading on every page,
    so each page opens a new section and the document's real structure is
    replaced by its pagination."""
    from tests.fixtures import lettered_policy_pdf

    doc = parse_pdf(lettered_policy_pdf(), "lettered.pdf")
    all_text = " ".join(b.text for b in doc.blocks)

    assert "Page 2 of 3" not in all_text, "footer leaked into chunk text"
    assert all_text.count("DePaul University Information Security Policy") <= 1
    assert any("running header" in w for w in doc.warnings)


def test_two_page_document_keeps_a_heading_repeated_on_both_pages():
    """Header detection must not fire on short documents, where a genuine
    heading can legitimately appear on both pages."""
    raw = build_pdf([
        [("Access Control", H1_SIZE, True), ("MFA is required for all staff.", BODY_SIZE, False)],
        [("Access Control", H1_SIZE, True), ("Reviews happen quarterly.", BODY_SIZE, False)],
    ])
    doc = parse_pdf(raw, "short.pdf")
    assert [b.text for b in doc.blocks if b.kind == "heading"].count("Access Control") == 2


# Lines taken verbatim from a real university policy, as pdfplumber breaks
# them. Each one was classified as a section heading at some point during
# development, and each reset the heading stack, filing the text that
# followed under a section it had nothing to do with.
REAL_PROSE_LINES = [
    "2. Report all breaches to (or losses/improper uses of) DePaul data, systems or devices. Such",
    "3. Ensure oversight of Service Providers Having Covered Data Access. Any Service Provider",
    "1. Assessing the risks associated with DePaul data, systems or devices. Risk assessment models and",
    "3. Determining whether there has been a 'Breach Requiring Notice' and, if so, notifying the",
    "1. Comply with all University IS Policies.",
    "2. Designing, implementing and monitoring safeguards to help minimize the risks associated with",
]


@pytest.mark.parametrize("line", REAL_PROSE_LINES, ids=lambda s: s[:34])
def test_real_prose_lines_are_never_headings(line):
    assert numbering_level(line) is None


@pytest.mark.parametrize(
    "line",
    ["4.2 Exceptions", "4. Access Control", "1. Data Retention", "2.1 Deletion Requests",
     "4.2.1 Password Rules", "3. Incident Response Plan", "Section 12 Incident Response"],
)
def test_genuine_numbered_headings_survive_the_stricter_rules(line):
    assert numbering_level(line) is not None


def test_numbered_body_line_needs_typographic_emphasis_to_be_a_heading():
    """Shape rules reject most prose list items, but in a PDF the reliable
    discriminator is typography: real headings are bold or larger, list items
    are set in the body face."""
    raw = build_pdf([
        [
            ("4. Access Control", H2_SIZE, True),
            ("1. Use multi-factor authentication on every production account",
             BODY_SIZE, False),
            ("2. Rotate encryption keys at least once every twelve months",
             BODY_SIZE, False),
            ("Access to production systems is reviewed each quarter by the security team.",
             BODY_SIZE, False),
        ]
    ])
    headings = [b.text for b in parse_pdf(raw, "emph.pdf").blocks if b.kind == "heading"]
    assert headings == ["4. Access Control"], f"unexpected headings: {headings}"


def test_ambiguous_short_numbered_line_is_treated_as_body_text():
    """Documents the limit of what parsing can know. With no emphasis and
    four words, a list item and a heading look identical, so the tie breaks
    toward body text: a missed heading costs precision, a false one misfiles
    every chunk after it."""
    raw = build_pdf([
        [
            ("Security Standards", H2_SIZE, True),
            ("1. Use MFA everywhere", BODY_SIZE, False),
            ("This applies to all staff and contractors without exception.", BODY_SIZE, False),
        ]
    ])
    headings = [b.text for b in parse_pdf(raw, "amb.pdf").blocks if b.kind == "heading"]
    assert "1. Use MFA everywhere" not in headings


def test_short_numbered_heading_works_without_emphasis():
    """Some policies set section headings in the body face. A very short
    numbered line is still taken as a heading."""
    raw = build_pdf([
        [
            ("4.2 Exceptions", BODY_SIZE, False),
            ("Break-glass accounts are exempt and use hardware tokens held in a safe.",
             BODY_SIZE, False),
        ]
    ])
    assert "4.2 Exceptions" in [
        b.text for b in parse_pdf(raw, "plain.pdf").blocks if b.kind == "heading"
    ]
