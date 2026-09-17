"""Golden dataset format.

A case states what the documents contain, not what a good answer sounds
like. Every expectation is a substring, a section path or a status - things
that can be checked mechanically and argued about by a human reading the
source PDF. Nothing here asks a model whether an answer was good.

Ground truth is recorded with its evidence: `expect_text` names strings that
must appear in the retrieved evidence, and `forbid_literals` names strings
that must not appear in the answer. The second is how hallucination is
measured rather than estimated.
"""

from __future__ import annotations

import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Any

import yaml

VALID_STATUSES = {"supported", "review_required", "refused", "answered"}
"""'answered' is an alias accepted in the YAML for either supported or
review_required, for cases where a human-reviewed answer is acceptable but
the exact status is not worth pinning."""


class DatasetError(ValueError):
    """A malformed case. Raised at load time rather than mid-run, so a typo
    in the golden file cannot quietly become a passing eval."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    expect_status: str
    """supported | review_required | refused | answered"""
    expect_text: tuple[str, ...] = ()
    """Substrings that must appear in the retrieved evidence. This is the
    retrieval ground truth: if these are missing, retrieval failed, whatever
    the answer said."""
    expect_answer_contains: tuple[str, ...] = ()
    """Substrings the answer must contain when the case is answerable."""
    expect_sections: tuple[str, ...] = ()
    """Fragments that must appear in at least one citation's heading path."""
    forbid_literals: tuple[str, ...] = ()
    """Strings that must never appear in the answer. Certifications the
    company does not hold, standards not in the documents, invented figures."""
    category: str = "general"
    rationale: str = ""
    """Why this case exists and what a wrong answer would cost. Required for
    new cases: a golden case nobody can justify is a case nobody will fix
    when it fails."""
    note: str = ""

    @property
    def answerable(self) -> bool:
        return self.expect_status != "refused"


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    document: str
    """A filename the cases were written against, checked at run time so a
    dataset cannot be silently scored against the wrong corpus."""
    cases: tuple[EvalCase, ...] = ()
    documents: tuple[str, ...] = ()
    """Glob patterns, relative to the repository root, naming exactly the
    corpus these cases were written against.

    Declared here rather than assembled by the setup script because the two
    drifted: a setup flag ingested every document on disk into one tenant, so
    twelve cases written against four small fixtures were scored while
    competing with 164 chunks of five unrelated university policies. The
    retrieval numbers from that run were an artefact of the setup, not a
    property of the system."""
    description: str = ""
    source: str = ""
    tags: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)


def _tuple(value: Any, field_name: str, case_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise DatasetError(f"case {case_id}: {field_name} must be a string or list of strings")


def load_dataset(path: str | pathlib.Path) -> Dataset:
    """Load and validate a golden dataset."""
    file_path = pathlib.Path(path)
    if not file_path.exists():
        raise DatasetError(f"dataset not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise DatasetError(f"{file_path}: top level must be a mapping")

    for required in ("name", "document", "cases"):
        if required not in raw:
            raise DatasetError(f"{file_path}: missing '{required}'")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw["cases"] or []):
        if not isinstance(entry, dict):
            raise DatasetError(f"{file_path}: case {index} is not a mapping")
        case_id = str(entry.get("id") or f"case-{index}")
        if case_id in seen:
            raise DatasetError(f"{file_path}: duplicate case id '{case_id}'")
        seen.add(case_id)

        question = entry.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"case {case_id}: question is required")

        status = str(entry.get("expect_status", "")).lower()
        if status not in VALID_STATUSES:
            raise DatasetError(
                f"case {case_id}: expect_status must be one of {sorted(VALID_STATUSES)}"
            )

        expect_text = _tuple(entry.get("expect_text"), "expect_text", case_id)
        if status != "refused" and not expect_text:
            raise DatasetError(
                f"case {case_id}: an answerable case needs expect_text, otherwise a "
                "retrieval failure scores the same as a retrieval success"
            )

        cases.append(
            EvalCase(
                id=case_id,
                question=question.strip(),
                expect_status=status,
                expect_text=expect_text,
                expect_answer_contains=_tuple(
                    entry.get("expect_answer_contains"), "expect_answer_contains", case_id
                ),
                expect_sections=_tuple(entry.get("expect_sections"), "expect_sections", case_id),
                forbid_literals=_tuple(entry.get("forbid_literals"), "forbid_literals", case_id),
                category=str(entry.get("category", "general")),
                rationale=str(entry.get("rationale", "")),
                note=str(entry.get("note", "")),
            )
        )

    if not cases:
        raise DatasetError(f"{file_path}: no cases")

    documents = _tuple(raw.get("documents"), "documents", raw["name"])
    if not documents:
        raise DatasetError(
            f"{file_path}: missing 'documents'. A dataset must name the corpus it was "
            "written against, or its retrieval scores describe whatever happens to be "
            "in the tenant."
        )

    return Dataset(
        name=str(raw["name"]),
        document=str(raw["document"]),
        documents=documents,
        cases=tuple(cases),
        description=str(raw.get("description", "")),
        source=str(raw.get("source", "")),
        tags=raw.get("tags") or {},
    )


EVAL_TENANT_NAMESPACE = uuid.UUID("6b2f0f6a-0000-4000-8000-000000000001")


def tenant_for(dataset_name: str) -> uuid.UUID:
    """Stable tenant id derived from the dataset name.

    Derived rather than configured so two datasets cannot be pointed at one
    tenant by a stray flag, and stable so a recorded cassette stays valid
    across machines instead of being orphaned behind a fresh random id.
    """
    return uuid.uuid5(EVAL_TENANT_NAMESPACE, dataset_name)


def resolve_documents(dataset: Dataset, root: pathlib.Path) -> list[pathlib.Path]:
    """Expand the dataset's globs into real files, in a stable order."""
    found: list[pathlib.Path] = []
    for pattern in dataset.documents:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise DatasetError(
                f"dataset '{dataset.name}' declares documents '{pattern}' but nothing "
                f"matches under {root}"
            )
        found.extend(matches)
    return found
