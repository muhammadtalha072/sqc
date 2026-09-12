"""Data shapes shared by every parser and the chunker.

Design rule enforced throughout: a field is None when the source genuinely
does not carry that information. Nothing here is ever inferred to look
tidier. A DOCX has no reliable page numbers without rendering it, so its
blocks carry page=None rather than a plausible-looking guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

BlockKind = Literal["heading", "paragraph", "list_item", "table"]
DocFormat = Literal["pdf", "docx", "txt"]

MAX_HEADING_LEVEL = 6


@dataclass(frozen=True, slots=True)
class Block:
    """One structural unit of a document, in reading order."""

    text: str
    kind: BlockKind = "paragraph"
    level: int | None = None
    """Heading depth 1..6. None for every non-heading block."""
    page: int | None = None
    """1-based page where the block starts, or None when the format has no
    page concept."""
    page_end: int | None = None
    """Last page the block touches. None means it does not span pages, in
    which case it ends on `page`. A paragraph that continues across a page
    break must cite both pages, not just the one it started on."""

    def __post_init__(self) -> None:
        if self.kind == "heading":
            if self.level is None or not 1 <= self.level <= MAX_HEADING_LEVEL:
                raise ValueError(f"heading needs level 1..{MAX_HEADING_LEVEL}, got {self.level}")
        elif self.level is not None:
            raise ValueError(f"{self.kind} block must not carry a heading level")
        if self.page_end is not None:
            if self.page is None:
                raise ValueError("page_end set without a starting page")
            if self.page_end < self.page:
                raise ValueError(f"page_end {self.page_end} precedes page {self.page}")

    @property
    def pages(self) -> tuple[int | None, int | None]:
        """(first, last) pages touched. Both None when pages are unknown."""
        return (self.page, self.page_end if self.page_end is not None else self.page)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A document reduced to ordered blocks plus whatever metadata was real."""

    filename: str
    doc_format: DocFormat
    blocks: tuple[Block, ...]
    content_sha256: str
    page_count: int | None = None
    effective_date: date | None = None
    version_label: str | None = None
    warnings: tuple[str, ...] = ()
    """Non-fatal problems worth surfacing: unresolvable dates, empty pages,
    tables that could not be extracted. These reach the operator, never the
    answering model."""

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(frozen=True, slots=True)
class LeafChunk:
    """What gets embedded and stored. Maps 1:1 onto a row in `chunks`.

    small-to-big: `text` is the precise unit we retrieve on, `parent_text`
    is the wider section handed to the answering model so that qualifiers
    and exceptions travel with the claim.
    """

    chunk_index: int
    text: str
    parent_text: str
    embed_text: str
    heading_path: tuple[str, ...] = ()
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    injection_flags: tuple[str, ...] = field(default=())

    @property
    def heading_text(self) -> str:
        """Scalar form of heading_path for the generated tsvector column.

        Joined with spaces rather than a separator so full-text search sees
        clean lexemes. The DB CHECK requires this to be empty exactly when
        heading_path is empty, which holds because empty headings are
        dropped during parsing.
        """
        return " ".join(self.heading_path)
