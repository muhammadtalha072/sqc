"""Format adapters. Each returns a ParsedDocument; nothing downstream knows
which format it came from.

Library choices are licence-driven: pdfplumber (MIT), pypdf (BSD) and
python-docx (MIT). PyMuPDF is deliberately absent because it is AGPL-3.0,
which would force this backend open or require a commercial licence.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections import Counter

from sqc.core.ingestion.injection import strip_hidden_characters
from sqc.core.ingestion.model import Block, DocFormat, ParsedDocument
from sqc.core.ingestion.structure import (
    extract_effective_date,
    extract_version_label,
    levels_from_font_sizes,
    looks_like_heading,
    numbering_level,
)

SUPPORTED_SUFFIXES = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "txt"}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25cf\u25aa]|\(?[a-z0-9]{1,3}[.)])\s+\S")
# A line ending in sentence punctuation has finished its thought; used to
# decide whether a paragraph really continues across a page break.
_ENDS_SENTENCE = re.compile(r"[.!?:;][\"\u201d\)]?\s*$")


class UnsupportedFormatError(ValueError):
    pass


class DocumentParseError(RuntimeError):
    pass


def _clean(text: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", strip_hidden_characters(text)).strip()


def _finalise(
    filename: str,
    doc_format: DocFormat,
    blocks: list[Block],
    raw: bytes,
    page_count: int | None,
    warnings: list[str],
) -> ParsedDocument:
    """Attach metadata and hash. Metadata is read from the assembled text so
    every format uses one extraction path."""
    head = "\n".join(b.text for b in blocks[:60])
    effective_date, date_warnings = extract_effective_date(head)
    warnings.extend(date_warnings)
    if not blocks:
        warnings.append("document produced no extractable text")
    return ParsedDocument(
        filename=filename,
        doc_format=doc_format,
        blocks=tuple(blocks),
        content_sha256=hashlib.sha256(raw).hexdigest(),
        page_count=page_count,
        effective_date=effective_date,
        version_label=extract_version_label(head),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------- txt


def parse_txt(raw: bytes, filename: str) -> ParsedDocument:
    """Plain text and markdown. No page numbers exist, so page stays None."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    blocks: list[Block] = []
    warnings: list[str] = []
    for raw_para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in raw_para.splitlines() if ln.strip()]
        if not lines:
            continue
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                body = _clean(" ".join(buffer))
                if body:
                    kind = "list_item" if _BULLET.match(buffer[0]) else "paragraph"
                    blocks.append(Block(text=body, kind=kind))
                buffer.clear()

        for line in lines:
            md = _MD_HEADING.match(line.strip())
            if md:
                flush()
                title = _clean(md.group(2))
                if title:
                    blocks.append(Block(text=title, kind="heading", level=len(md.group(1))))
                continue
            level = numbering_level(line.strip())
            # A single short line standing alone is a heading candidate.
            if level is not None or (len(lines) == 1 and looks_like_heading(line)):
                flush()
                title = _clean(line)
                if title:
                    blocks.append(Block(text=title, kind="heading", level=level or 1))
                continue
            buffer.append(line)
        flush()
    return _finalise(filename, "txt", blocks, raw, None, warnings)


# --------------------------------------------------------------------- docx


def parse_docx(raw: bytes, filename: str) -> ParsedDocument:
    """DOCX via python-docx.

    Page numbers are None throughout: Word stores no page breaks for
    reflowed text, so any page number here would be invented.
    """
    import io

    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        raise DocumentParseError(f"could not open {filename} as DOCX: {exc}") from exc

    blocks: list[Block] = []
    warnings: list[str] = ["docx has no reliable page numbers; page left unset"]

    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, document)
            text = _clean(para.text)
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            level: int | None = None
            if style.lower().startswith("heading"):
                digits = re.search(r"(\d+)", style)
                level = min(int(digits.group(1)), 6) if digits else 1
            elif style.lower() == "title":
                level = 1
            else:
                level = numbering_level(text)
                if level is None and _is_all_bold(para) and looks_like_heading(text):
                    level = 2
            if level is not None:
                blocks.append(Block(text=text, kind="heading", level=level))
            else:
                kind = "list_item" if _BULLET.match(para.text) or "List" in style else "paragraph"
                blocks.append(Block(text=text, kind=kind))
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rendered = _render_table([[c.text for c in row.cells] for row in table.rows])
            if rendered:
                blocks.append(Block(text=rendered, kind="table"))
    return _finalise(filename, "docx", blocks, raw, None, warnings)


def _is_all_bold(paragraph) -> bool:  # noqa: ANN001 - python-docx type
    runs = [r for r in paragraph.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def _render_table(rows: list[list[str]]) -> str:
    """Serialise a table as pipe-delimited rows.

    Deliberately plain: the value of a control-matrix table is the cell
    text and its row/column pairing, and markdown pipes preserve both while
    staying readable to the model and in a citation.
    """
    cleaned = [[_clean(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    return "\n".join(" | ".join(row) for row in cleaned)


# ---------------------------------------------------------------------- pdf


def _normalise_repeating(text: str) -> str:
    """Collapse digits so 'Page 3 of 7' and 'Page 4 of 7' compare equal."""
    return re.sub(r"\d+", "#", text.strip().lower())


MARGIN_FRACTION = 0.12
"""Fraction of page height at the top and bottom treated as margin. Running
headers and footers live there; body text does not."""


def _find_running_lines(pages: list[list[dict]], page_count: int) -> set[str]:
    """Text repeated in the margin across most pages: headers and footers.

    Left in place they do real damage. A running header set in a larger face
    than the body is detected as a heading on every page, so each page opens
    a new parent section and the document's actual structure is replaced by
    its pagination. Footers additionally pollute chunk text with 'Page 4 of 7'.

    Only applied from three pages up, since on a two-page document a genuine
    heading could legitimately appear on both.
    """
    if page_count < 3:
        return set()
    seen: dict[str, set[int]] = {}
    for lines in pages:
        for line in lines:
            if not line.get("in_margin"):
                continue
            key = _normalise_repeating(line["text"])
            if key:
                seen.setdefault(key, set()).add(line["page"])
    threshold = max(3, round(page_count * 0.5))
    return {key for key, page_numbers in seen.items() if len(page_numbers) >= threshold}


def parse_pdf(raw: bytes, filename: str) -> ParsedDocument:
    """PDF via pdfplumber, keeping real page numbers for citations.

    Headings are found by font size relative to the document's body size,
    with section numbering overriding font size where present.
    """
    import io

    import pdfplumber

    blocks: list[Block] = []
    warnings: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            page_count = len(pdf.pages)
            pages = [_extract_page_lines(page, index + 1) for index, page in enumerate(pdf.pages)]
            empty = [i + 1 for i, lines in enumerate(pages) if not lines]
            if empty:
                warnings.append(
                    f"{len(empty)} page(s) contained no extractable text: "
                    f"{', '.join(map(str, empty[:10]))}"
                    + (" (image-only pages need OCR)" if len(empty) == page_count else "")
                )

            running = _find_running_lines(pages, page_count)
            if running:
                warnings.append(
                    f"dropped {len(running)} running header/footer line(s) repeated across pages"
                )
            all_lines = [
                line
                for page_lines in pages
                for line in page_lines
                if _normalise_repeating(line["text"]) not in running
            ]
            if not all_lines:
                return _finalise(filename, "pdf", [], raw, page_count, warnings)

            body_size = _body_font_size(all_lines)
            size_levels = levels_from_font_sizes([ln["size"] for ln in all_lines], body_size)
            blocks = _lines_to_blocks(all_lines, body_size, size_levels)

            for page_index, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    rendered = _render_table([[c or "" for c in row] for row in table])
                    if rendered:
                        blocks.append(Block(text=rendered, kind="table", page=page_index))
    except DocumentParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(f"could not parse {filename} as PDF: {exc}") from exc
    return _finalise(filename, "pdf", blocks, raw, page_count, warnings)


def _extract_page_lines(page, page_number: int) -> list[dict]:  # noqa: ANN001
    """One dict per visual line, carrying font size, boldness and page.

    Lines inside a detected table region are skipped so table content is not
    emitted twice, once as prose and once as a table block.
    """
    try:
        table_boxes = [t.bbox for t in page.find_tables()]
    except Exception:  # noqa: BLE001 - table detection is best-effort
        table_boxes = []

    lines: list[dict] = []
    page_height = float(page.height or 0.0) or 1.0
    margin = page_height * MARGIN_FRACTION
    for line in page.extract_text_lines(return_chars=True, strip=True) or []:
        text = _clean(line.get("text", ""))
        if not text:
            continue
        top, bottom = line.get("top", 0.0), line.get("bottom", 0.0)
        middle = (top + bottom) / 2
        if any(box[1] <= middle <= box[3] for box in table_boxes):
            continue
        chars = line.get("chars") or []
        sizes = [c.get("size", 0.0) for c in chars if c.get("size")]
        fonts = [str(c.get("fontname", "")) for c in chars]
        lines.append(
            {
                "text": text,
                "page": page_number,
                "size": round(max(sizes), 1) if sizes else 0.0,
                "bold": bool(fonts) and sum("bold" in f.lower() for f in fonts) > len(fonts) / 2,
                "top": top,
                "bottom": bottom,
                "chars": len(text),
                "in_margin": top < margin or bottom > page_height - margin,
            }
        )
    return lines


def _body_font_size(lines: list[dict]) -> float:
    """Most common font size weighted by character count.

    Weighting by characters rather than by line stops a document with many
    short headings from electing a heading size as the body size.
    """
    counter: Counter[float] = Counter()
    for line in lines:
        counter[line["size"]] += line["chars"]
    return counter.most_common(1)[0][0] if counter else 0.0


def _lines_to_blocks(lines: list[dict], body_size: float, size_levels: dict[float, int]) -> list[Block]:
    """Merge consecutive body lines into paragraphs; emit headings alone."""
    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_page: int | None = None
    buffer_page_end: int | None = None

    def flush() -> None:
        nonlocal buffer_page, buffer_page_end
        if buffer:
            text = _clean(" ".join(buffer))
            if text:
                kind = "list_item" if _BULLET.match(buffer[0]) else "paragraph"
                span_end = buffer_page_end if buffer_page_end != buffer_page else None
                blocks.append(Block(text=text, kind=kind, page=buffer_page, page_end=span_end))
            buffer.clear()
            buffer_page = None
            buffer_page_end = None

    previous_bottom: float | None = None
    previous_page: int | None = None
    for line in lines:
        text = line["text"]
        level = _heading_level(line, body_size, size_levels)
        if level is not None:
            flush()
            blocks.append(Block(text=text, kind="heading", level=level, page=line["page"]))
            previous_bottom, previous_page = line["bottom"], line["page"]
            continue

        crossed_page = previous_page is not None and line["page"] != previous_page
        if crossed_page and buffer:
            # A paragraph only continues across a page break if the last line
            # left a sentence unfinished. Otherwise page 2 starts a new block,
            # so its text is never cited against page 1.
            if _ENDS_SENTENCE.search(buffer[-1]):
                flush()
        elif buffer:
            gap = line["top"] - previous_bottom if previous_bottom is not None else 0.0
            if gap > max(6.0, body_size * 0.8):
                flush()

        if not buffer:
            buffer_page = line["page"]
        buffer_page_end = line["page"]
        buffer.append(_dehyphenate(buffer, text))
        previous_bottom, previous_page = line["bottom"], line["page"]
    flush()
    return blocks


def _dehyphenate(buffer: list[str], text: str) -> str:
    """Join words split across a line break, e.g. 'encryp-' + 'tion'."""
    if buffer and buffer[-1].endswith("-") and text[:1].islower():
        buffer[-1] = buffer[-1][:-1] + text.split(" ", 1)[0]
        return text.split(" ", 1)[1] if " " in text else ""
    return text


SHORT_HEADING_WORDS = 3
"""A numbered line this short is taken as a heading even without typographic
emphasis, covering policies that set '4.2 Exceptions' in the body face.

Kept deliberately tight. At four words and no emphasis, '1. Use MFA
everywhere' and '3. Incident Response Plan' are genuinely
indistinguishable, so the tie is broken toward not-a-heading: a missed
heading costs some retrieval precision, while a false one misfiles every
chunk that follows it and produces citations naming the wrong section."""


def _heading_level(line: dict, body_size: float, size_levels: dict[float, int]) -> int | None:
    """Numbering gives the depth; typography decides whether it is a heading.

    Numbering alone is not enough, and assuming it was caused the worst bug
    found so far. Policies enumerate obligations as '2. Designing,
    implementing and monitoring safeguards to help minimize the risks
    associated with', set in the body face and wrapped at the margin. Shape
    rules reject most of those, but a short list item ending in a full stop
    slips through every text-only test. In a PDF the discriminator is right
    there: a real heading is bold, or larger than the body, or very short.
    """
    numbered = numbering_level(line["text"])
    size_level = size_levels.get(line["size"])
    distinguished = line["bold"] or line["size"] > body_size * 1.08

    if numbered is not None:
        if distinguished or len(line["text"].split()) <= SHORT_HEADING_WORDS:
            return numbered
        return None
    if size_level is not None and line["size"] > body_size * 1.08:
        return size_level
    if line["bold"] and looks_like_heading(line["text"]):
        return min(len(size_levels) + 1, 6) if size_levels else 2
    return None


# ----------------------------------------------------------------- dispatch


def parse_bytes(raw: bytes, filename: str) -> ParsedDocument:
    suffix = pathlib.Path(filename).suffix.lower()
    doc_format = SUPPORTED_SUFFIXES.get(suffix)
    if doc_format is None:
        raise UnsupportedFormatError(
            f"unsupported file type '{suffix or filename}'; "
            f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    if not raw:
        raise DocumentParseError(f"{filename} is empty")
    return {"pdf": parse_pdf, "docx": parse_docx, "txt": parse_txt}[doc_format](raw, filename)


def parse_file(path: str | pathlib.Path) -> ParsedDocument:
    path = pathlib.Path(path)
    return parse_bytes(path.read_bytes(), path.name)
