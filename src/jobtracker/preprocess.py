"""Text cleaning. The same code must be used for training data and live inference."""
from __future__ import annotations

import re

try:  # bs4 is only needed for HTML bodies
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

_UMLAUTS = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"https?://\S+")
_JOBMETA = re.compile(r"view job: url|~[a-z0-9\-]+~", flags=re.IGNORECASE)
_LOOKS_LIKE_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_REPLY_PREFIX = re.compile(r"^\s*((re|aw|fw|fwd|wg)\s*:\s*)+", flags=re.IGNORECASE)
_DOMAIN = re.compile(r"@([A-Za-z0-9.-]+)")


def normalize_umlauts(text: str) -> str:
    for k, v in _UMLAUTS.items():
        text = text.replace(k, v)
    return text


def strip_html(text: str) -> str:
    """Remove tags only when the text actually contains markup (avoids bs4 warnings)."""
    if not _LOOKS_LIKE_HTML.search(text):
        return text
    if BeautifulSoup is None:  # crude fallback
        return re.sub(r"<[^>]+>", " ", text)
    return BeautifulSoup(text, "html.parser").get_text(" ")


def clean_text(text) -> str:
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = strip_html(str(text))
    text = normalize_umlauts(text)
    text = _URL.sub(" URL ", text)
    text = _EMAIL.sub(" EMAIL ", text)
    text = _JOBMETA.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def build_model_text(subject, body) -> str:
    """Subject + body, cleaned. This is the single entry point used at inference time."""
    return clean_text(f"{subject or ''} {body or ''}")


def normalize_subject(subject) -> str:
    """Template key for grouping near-duplicate mails (reply prefixes and digits removed)."""
    s = "" if subject is None or (isinstance(subject, float) and subject != subject) else str(subject)
    s = _REPLY_PREFIX.sub("", normalize_umlauts(s).lower())
    s = re.sub(r"\d+", "0", s)
    return re.sub(r"\s+", " ", s).strip()


def sender_domain(sender) -> str:
    m = _DOMAIN.search("" if sender is None else str(sender))
    return m.group(1).lower() if m else ""
