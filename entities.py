"""Single source of truth for record-ID extraction (no third-party deps)."""
import re
from typing import List

# Tokens with at least one digit and one letter, optionally hyphen/underscore joined.
_ENTITY_RE = re.compile(
    r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b"
)

# Technical shorthands that look like IDs but are not records.
_NON_RECORD_ENTITY_RE = re.compile(
    r"^(?:FY|CY)\d{2,4}$|^(?:Q[1-4]|H[12])$|"
    r"^(?:IPV[46]|SHA\d+|MD5|TLS\d+|UTF\d+|HTTP[23]|OAUTH2|BASE64|X86|ARM64|AMD64|EC2|S3|"
    r"H100|H200|A100|B100|B200|GB200|RTX\d+|NVLINK\d*)$",
    flags=re.IGNORECASE,
)

# Fiscal periods, years and ordinals are never record IDs, alone or hyphen-joined
# (Q4-FY25, 2024-Q4, 4th).
_PERIOD_PART_RE = re.compile(
    r"^(?:(?:FY|CY)\d{2,4}|Q[1-4]|H[12]|(?:19|20)\d{2}|\d{1,3}(?:st|nd|rd|th))$",
    flags=re.IGNORECASE,
)

# SEC form names: 10-K, 10-Q, 8-K, 20-F, S-1.
_FORM_RE = re.compile(r"^(?:\d{1,2}-?[KQ]|20-F|S-\d{1,2})$", flags=re.IGNORECASE)


def _is_non_record(token: str) -> bool:
    if _NON_RECORD_ENTITY_RE.match(token) or _FORM_RE.match(token):
        return True
    return all(_PERIOD_PART_RE.match(p) for p in re.split(r"[-_]", token))


def extract_entities(query: str) -> List[str]:
    """Return record-like identifiers in `query`, in order, duplicates preserved."""
    out: List[str] = []
    for m in _ENTITY_RE.findall(query):
        letters = sum(ch.isalpha() for ch in m)
        has_joiner = "-" in m or "_" in m
        if _is_non_record(m) or not (has_joiner or letters >= 2):
            continue
        out.append(m)
    return out

