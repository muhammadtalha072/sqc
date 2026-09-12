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
