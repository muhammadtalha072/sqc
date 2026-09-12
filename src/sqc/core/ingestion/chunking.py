"""Section-aware chunking.

small-to-big, concretely:

  leaf chunk   ~300 tokens, what we embed and retrieve on. Small enough that
               a match means the chunk is genuinely about the question.
  parent text  the enclosing section, what the answering model reads. Large
               enough to carry the qualifier that changes the answer -
               "MFA is required for all staff" vs the next sentence,
               "except break-glass service accounts".

Parent sections never cross a level-1 or level-2 heading. Deeper headings
(H3+) subdivide a parent but do not split it, because in a security policy
the H3s under "4. Access Control" are usually the exceptions and conditions
that belong with the rule.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from sqc.core.ingestion.injection import scan_for_injection
from sqc.core.ingestion.model import Block, LeafChunk, ParsedDocument
from sqc.core.ingestion.structure import numbering_level

DEFAULT_PARENT_BOUNDARY_LEVEL = 2
TARGET_LEAF_TOKENS = 300
MAX_LEAF_TOKENS = 450
MIN_LEAF_TOKENS = 40
MAX_PARENT_TOKENS = 1200
OVERLAP_SENTENCES = 1
MAX_OVERLAP_TOKENS = 80

HARD_MAX_LEAF_TOKENS = MAX_LEAF_TOKENS + MAX_OVERLAP_TOKENS
"""Ceiling no leaf may exceed, guaranteed by construction.

Without a hard ceiling, chunk size is at the mercy of the source text. A
control matrix serialised as pipe-delimited rows contains no sentence
punctuation at all, so sentence splitting returns it as one unit and it
would be stored, and embedded, whole. Real policies are full of such tables.
"""

# Splits on sentence punctuation followed by a new sentence, and on line
# breaks. The line-break rule is what handles tables and enumerated lists,
# where each row is a unit but no row ends in a full stop.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split on word boundaries.

    Only reached when a single unit survives sentence splitting and is still
    too large: a table with no punctuation, a semicolon-joined enumeration,
    or OCR output with no sentence structure. Cutting mid-sentence is ugly,
    and it is still better than an unbounded chunk, because the parent
    section travels with every leaf and restores the context.
    """
    words = text.split()
    if not words:
        return []
    # Inverts the same words-to-tokens ratio estimate_tokens uses.
    per_piece = max(1, int(max_tokens / 1.3))
    return [" ".join(words[i : i + per_piece]) for i in range(0, len(words), per_piece)]


def parent_boundary_level(blocks: tuple[Block, ...]) -> int:
    """Heading depth at which a new parent section begins, per document.

    A fixed level cannot work. In one policy the numbered sections are
    level 1 because the title is unnumbered; in another the title takes
    level 1 and sections sit at level 2. Hard-coding either splits '4.1
    Authentication' away from '4.2 Exceptions', which is precisely the
    rule-and-its-exception pair the parent context exists to keep together.

    So the boundary follows the document's own scheme: the shallowest
    *numbered* section heading, since numbering is the structure a policy
    author actually intended. With no numbering anywhere, fall back to the
    shallowest level that appears more than once, which skips a one-off
    title heading. Parent size stays bounded regardless by MAX_PARENT_TOKENS.
    """
    headings = [b for b in blocks if b.kind == "heading" and b.level is not None]
    if not headings:
        return DEFAULT_PARENT_BOUNDARY_LEVEL

    numbered = [b.level for b in headings if numbering_level(b.text) is not None]
    if numbered:
        return min(numbered)

    counts = Counter(b.level for b in headings)
    repeated = [level for level, n in counts.items() if n >= 2]
    return min(repeated) if repeated else min(counts)


def estimate_tokens(text: str) -> int:
    """Approximate token count without a provider tokenizer.

    Every provider tokenises differently and their vocabularies are not
    available offline, so chunk sizes are treated as targets, not limits.
    The 1.3 factor is the usual English word-to-token ratio; being 10% out
    changes retrieval quality far less than getting section boundaries wrong.
    """
    words = len(text.split())
    return max(1, round(words * 1.3)) if words else 0


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


@dataclass(slots=True)
class _Section:
    """A parent section: content under one H1/H2, plus the headings inside it."""

    heading_path: tuple[str, ...]
    blocks: list[tuple[Block, tuple[str, ...]]]
    """Each content block with the full heading path in force at that point,
    so an H3 subsection keeps its own path while sharing the parent text."""


def _build_sections(blocks: tuple[Block, ...], boundary_level: int) -> list[_Section]:
    """Walk blocks once, maintaining a heading stack, cutting at the
    document's own top section level."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(heading_path=(), blocks=[])

    for block in blocks:
        if block.kind == "heading":
            level = block.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))
            if level <= boundary_level:
                if current.blocks:
                    sections.append(current)
                current = _Section(heading_path=tuple(t for _, t in stack), blocks=[])
            continue
        current.blocks.append((block, tuple(t for _, t in stack)))

    if current.blocks:
        sections.append(current)
    return sections


def _pack_leaves(section: _Section) -> list[list[tuple[Block, tuple[str, ...]]]]:
    """Group a section's blocks into leaf-sized groups.

    A block larger than MAX_LEAF_TOKENS is split on sentence boundaries and
    each piece becomes a synthetic block, so one enormous paragraph cannot
    produce a chunk too big to embed.
    """
    units: list[tuple[Block, tuple[str, ...]]] = []
    for block, path in section.blocks:
        if estimate_tokens(block.text) <= MAX_LEAF_TOKENS:
            units.append((block, path))
            continue
        buffer: list[str] = []
        for sentence in split_sentences(block.text):
            # A "sentence" can still be enormous when the source has no
            # sentence structure, so oversized ones are cut on word
            # boundaries before packing.
            pieces = (
                _hard_split(sentence, TARGET_LEAF_TOKENS)
                if estimate_tokens(sentence) > MAX_LEAF_TOKENS
                else [sentence]
            )
            for piece in pieces:
                candidate = " ".join(buffer + [piece])
                if buffer and estimate_tokens(candidate) > TARGET_LEAF_TOKENS:
                    units.append((_replace_text(block, " ".join(buffer)), path))
                    buffer = [piece]
                else:
                    buffer.append(piece)
        if buffer:
            units.append((_replace_text(block, " ".join(buffer)), path))

    groups: list[list[tuple[Block, tuple[str, ...]]]] = []
    current: list[tuple[Block, tuple[str, ...]]] = []
    for unit in units:
        block, path = unit
        # A change of sub-heading starts a new leaf: mixing two H3 topics in
        # one chunk is what produces a confident answer from the wrong rule.
        path_changed = bool(current) and current[-1][1] != path
        size = estimate_tokens(" ".join(b.text for b, _ in current + [unit]))
        if current and (path_changed or size > MAX_LEAF_TOKENS):
            groups.append(current)
            current = []
        current.append(unit)
        if estimate_tokens(" ".join(b.text for b, _ in current)) >= TARGET_LEAF_TOKENS:
            groups.append(current)
            current = []
    if current:
        # Fold a short tail into the previous leaf rather than emitting a
        # stub, but never at the cost of breaching the cap.
        tail_text = " ".join(b.text for b, _ in current)
        tail_tokens = estimate_tokens(tail_text)
        previous_fits = (
            groups
            and groups[-1][-1][1] == current[0][1]
            and estimate_tokens(" ".join(b.text for b, _ in groups[-1])) + tail_tokens
            <= MAX_LEAF_TOKENS
        )
        if previous_fits and tail_tokens < MIN_LEAF_TOKENS:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _replace_text(block: Block, text: str) -> Block:
    return Block(text=text, kind=block.kind, level=None, page=block.page)


def _window_parent(section_text: str, leaf_text: str) -> str:
    """Cap parent context, centred on the leaf rather than truncated at the top.

    Truncating from the start is the obvious implementation and the wrong
    one: in a long section the leaf that matched is often near the end, so
    head-truncation drops exactly the context the answer needs.
    """
    if estimate_tokens(section_text) <= MAX_PARENT_TOKENS:
        return section_text
    sentences = split_sentences(section_text)
    anchor = next(
        (i for i, s in enumerate(sentences) if s and s[:60] in leaf_text),
        len(sentences) // 2,
    )
    selected, budget = [sentences[anchor]], estimate_tokens(sentences[anchor])
    left, right = anchor - 1, anchor + 1
    while budget < MAX_PARENT_TOKENS and (left >= 0 or right < len(sentences)):
        if left >= 0:
            cost = estimate_tokens(sentences[left])
            if budget + cost > MAX_PARENT_TOKENS:
                break
            selected.insert(0, sentences[left])
            budget += cost
            left -= 1
        if right < len(sentences):
            cost = estimate_tokens(sentences[right])
            if budget + cost > MAX_PARENT_TOKENS:
                break
            selected.append(sentences[right])
            budget += cost
            right += 1
    return " ".join(selected)


def _pages(blocks: list[Block]) -> tuple[int | None, int | None]:
    """Span of pages a leaf touches, using each block's full range so a
    paragraph that crosses a page break cites both pages."""
    pages = [p for b in blocks for p in b.pages if p is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def chunk_document(document: ParsedDocument) -> list[LeafChunk]:
    """Turn a parsed document into storable leaf chunks."""
    chunks: list[LeafChunk] = []
    index = 0

    boundary = parent_boundary_level(document.blocks)

    for section in _build_sections(document.blocks, boundary):
        section_text = "\n\n".join(b.text for b, _ in section.blocks)
        groups = _pack_leaves(section)

        for position, group in enumerate(groups):
            blocks = [b for b, _ in group]
            leaf_text = "\n\n".join(b.text for b in blocks)

            # Overlap: carry the tail of the previous leaf so a claim split
            # across a boundary is still retrievable from either side. Capped,
            # because an unbounded tail would breach HARD_MAX_LEAF_TOKENS.
            if position > 0 and OVERLAP_SENTENCES:
                previous = " ".join(b.text for b, _ in groups[position - 1])
                carried = " ".join(split_sentences(previous)[-OVERLAP_SENTENCES:])
                if estimate_tokens(carried) > MAX_OVERLAP_TOKENS:
                    words = carried.split()
                    carried = " ".join(words[-max(1, int(MAX_OVERLAP_TOKENS / 1.3)) :])
                if carried:
                    leaf_text = carried + "\n\n" + leaf_text

            heading_path = group[-1][1]
            page_start, page_end = _pages(blocks)
            heading_prefix = " > ".join(heading_path)
            embed_text = f"{heading_prefix}\n{leaf_text}" if heading_prefix else leaf_text

            chunks.append(
                LeafChunk(
                    chunk_index=index,
                    text=leaf_text,
                    parent_text=_window_parent(section_text, leaf_text),
                    embed_text=embed_text,
                    heading_path=heading_path,
                    section=heading_path[-1] if heading_path else None,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=estimate_tokens(leaf_text),
                    injection_flags=scan_for_injection(leaf_text),
                )
            )
            index += 1
    return chunks
