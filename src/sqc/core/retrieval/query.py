"""Query preparation for lexical retrieval.

Security questionnaires are written in acronyms while policies are written
in prose. A questionnaire asks "Do you enforce MFA?"; the policy says
"multi-factor authentication is required". Dense retrieval closes some of
that gap, but lexical retrieval closes none of it, and lexical retrieval is
exactly what catches the precise terms that matter here - AES-256, TLS 1.2,
RPO, SOC 2.

So the acronym is expanded on the query side only. Stored text is never
rewritten: the evidence a customer sees must be the words their policy
actually uses.

The map below is a starting point, not a tuned artefact. It should be
extended from questionnaires that fail retrieval in the eval suite rather
than by guessing at more entries.
"""

from __future__ import annotations

import re

ACRONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "mfa": ("multi-factor authentication", "two-factor", "2fa"),
    "2fa": ("two-factor authentication", "mfa"),
    "sso": ("single sign-on", "federated authentication", "saml"),
    "rbac": ("role-based access control", "least privilege"),
    "iam": ("identity and access management",),
    "pam": ("privileged access management",),
    "dlp": ("data loss prevention",),
    "siem": ("security information and event management", "log monitoring"),
    "edr": ("endpoint detection and response",),
    "waf": ("web application firewall",),
    "rpo": ("recovery point objective",),
    "rto": ("recovery time objective",),
    "bcp": ("business continuity plan",),
    "dr": ("disaster recovery",),
    "ir": ("incident response",),
    "sdlc": ("software development lifecycle", "secure development"),
    "pii": ("personally identifiable information", "personal data"),
    "phi": ("protected health information",),
    "dpa": ("data processing agreement",),
    "dpia": ("data protection impact assessment",),
    "sca": ("software composition analysis",),
    "sast": ("static application security testing",),
    "dast": ("dynamic application security testing",),
    "vpn": ("virtual private network",),
    "kms": ("key management service", "encryption key management"),
    "mdm": ("mobile device management",),
    "soc": ("service organization control",),
    "vdp": ("vulnerability disclosure program",),
}

STOPWORDS = frozenset(
    """do does did you your yours we our is are was were have has had the a an
    and or of to in on for with any all please provide describe explain list
    what which how when where who whether if then that this these those there
    can could will would should may might must be been being it its as at by
    from into about""".split()
)
"""Dropped from the OR-branch of the tsquery. Postgres already strips true
stopwords, but questionnaire boilerplate like "please describe" survives
stemming and dilutes ranking."""

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*")


def extract_terms(question: str) -> list[str]:
    """Content words, order preserved, duplicates removed."""
    seen: set[str] = set()
    terms: list[str] = []
    for match in _WORD.findall(question):
        token = match.strip(".-").lower()
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def expand_query(question: str) -> tuple[str, list[str]]:
    """Return (text for full-text search, terms used).

    The expansion is appended rather than substituted, so a policy that does
    spell out "MFA" still matches on the acronym itself.
    """
    terms = extract_terms(question)
    expanded: list[str] = list(terms)
    for term in terms:
        for phrase in ACRONYM_EXPANSIONS.get(term, ()):
            for word in extract_terms(phrase):
                if word not in expanded:
                    expanded.append(word)
    return " ".join(expanded), terms


def build_tsquery(question: str) -> str | None:
    """Build an OR-query string for to_tsquery.

    websearch_to_tsquery is friendlier but ANDs its terms, which makes a
    long questionnaire sentence match nothing at all. Ranking, not matching,
    is what should decide relevance here, so terms are ORed and ts_rank_cd
    sorts them out. Terms are quoted to survive punctuation like "aes-256".
    """
    text, _ = expand_query(question)
    terms = text.split()
    if not terms:
        return None
    return " | ".join(f"'{term}'" for term in terms)
