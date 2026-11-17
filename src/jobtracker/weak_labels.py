"""High-precision keyword labellers used to pre-label a whole mailbox for human review.

They never replace review: `confidence` is "high" only when exactly one intent matches, and the
review queue puts conflicts and model disagreements first.
"""
from __future__ import annotations

import re

from .preprocess import normalize_umlauts
from .rules import ALERT_SUBJECT_PHRASES, ATS_DOMAINS, PORTAL_DOMAINS, authenticity
from .preprocess import sender_domain

INTENT_PATTERNS = {
    # Only genuine invitations. A mere mention ("we may invite you to an interview") is not an interview mail.
    "interview": (
        r"einladung (zum|zu einem|zu unserem) (finalen |ersten |zweiten )?(vorstellungsgespraech|interview|kennenlern|gespraech)|"
        r"termin zum vorstellungsgespraech|vereinbarten termin|erinnerung an den termin|"
        r"terminvorschlag|(moechten|wuerden) (wir )?(sie|dich) .{0,40}einladen|invite you (to|for) (an?|your|the) (interview|call|conversation)|"
        r"interview (invitation|request)|schedule (an?|your) (interview|call)|calendly|doodle\.com|vorstellungsgespraech am"
    ),
    "rejection": (
        r"leider (muessen|koennen|kein|nicht|haben wir|aber|ist|mitteilen)|regret to inform|we regret|wir bedauern|"
        r"nach (sorgfaeltiger|eingehender|genauer|gruendlicher) pruefung.{0,80}(muessen|koennen|leider)|"
        r"unfortunately|absage|won'?t be moving forward|"
        r"not (be )?moving forward|cannot (offer|include)|nicht (weiter )?beruecksichtig|andere(n)? (bewerber|kandidat)|"
        r"other candidates|pause recruitment|paus(e|ing) (the )?recruit|profiles we are seeking|decided (not|against)|"
        r"nicht in die engere|position (has been |was )?filled|anderweitig (besetzt|vergeben)|zurueckziehen|shortlist"
    ),
    "next_step": (
        r"take-home|challenge|assessment|fragebogen|questionnaire|the next step is|naechster schritt ist|"
        r"bitte (bestaetige|registrier|lade|sende|uebersende|vervollstaendige)|please (confirm|register|complete|upload|submit|provide)|"
        r"recruiting-portal|hochladen|fehlende unterlagen|unterlagen nachreichen|verify your|"
        r"noch nicht (ganz )?komplett|invitation to finish|einladung zur registrierung|arbeitserlaubnis|aktualisiere"
    ),
    "acknowledgment": (
        r"eingegangen|erhalten|received|thank you for (your |the )?(application|applying)|thanks for applying|"
        r"vielen dank fuer (ihre|deine) bewerbung|danke fuer (ihre|deine) bewerbung|herzlichen dank fuer (ihre|deine) bewerbung|"
        r"eingangsbestaetigung|acknowledg|we.ve received|application was sent|your application (is in|at|for)|"
        r"successfully submitted|bewerbung (bei uns )?eingegangen|wir freuen uns ueber (ihr|dein) interesse|bedanken uns fuer ihre bewerbung"
    ),
}
_COMPILED = {k: re.compile(v) for k, v in INTENT_PATTERNS.items()}
_ALERT = re.compile("|".join(re.escape(p) for p in ALERT_SUBJECT_PHRASES))
_PROMO = re.compile(r"newsletter|unsubscribe|abmelden|rabatt|sale\b|% off|angebot|webinar|hireready|tips? (for|to)|tipps")
_APPLICATION = re.compile(r"\bbewerbung|\bapplication\b|\bapplying\b|vorstellungsgespraech|interview|absage")
# Applications for housing / university places are not job mail ("Studentenwerk", "accommodation application").
_NONJOB = re.compile(r"studentenwerk|studierendenwerk|accommodation|wohnheim|wohnen|wohnung|zulassung|admission|visum|visa\b|aufenthalt")
_JOBWORDS = re.compile(r"werkstudent|praktikum|internship|working student|stelle\b|position|\bjob|minijob|aushilfe")
_HIRED = re.compile(r"willkommen im team|welcome to the team|vertragsangebot|job offer|arbeitsvertrag|zusage")


def _norm(x) -> str:
    return normalize_umlauts("" if x is None or (isinstance(x, float) and x != x) else str(x)).lower()


_CONDITIONAL = re.compile(r"\b(may|might|could|if|falls|sollte|sollten|wenn|whether|ob)\b")


def _real_invitation(blob: str) -> bool:
    """True if at least one interview phrase is not introduced by conditional wording ("we may invite you...")."""
    for m in _COMPILED["interview"].finditer(blob):
        if not _CONDITIONAL.search(blob[max(0, m.start() - 35): m.start()]):
            return True
    return False


def weak_intent(subject, text) -> tuple[str | None, str]:
    """Return (intent or None, confidence 'high' | 'low' | 'none')."""
    blob = f"{_norm(subject)} {_norm(text)}"
    hits = [name for name, rx in _COMPILED.items() if rx.search(blob)]
    if "interview" in hits and not _real_invitation(blob):
        hits.remove("interview")
    if not hits:
        return None, "none"
    # an acknowledgment phrase often appears inside rejections ("thank you for your application ... unfortunately")
    if len(hits) > 1 and "acknowledgment" in hits:
        hits.remove("acknowledgment")
        if len(hits) == 1:
            return hits[0], "high" if hits[0] in ("rejection", "interview") else "low"
    for priority in ("interview", "rejection", "next_step", "acknowledgment"):
        if priority in hits:
            return priority, "high" if len(hits) == 1 else "low"
    return None, "none"


def weak_category(sender, subject, text) -> tuple[str | None, str]:
    """Stage-1 label from sender/subject signals, or (None, 'none') when it cannot tell."""
    dom, sub, body = sender_domain(sender), _norm(subject), _norm(text)
    if _NONJOB.search(sub + " " + body[:300]) and not _JOBWORDS.search(sub + " " + body[:300]):
        return "non_job", "high"
    if _HIRED.search(sub):
        return "job_related", "high"
    if _ALERT.search(sub) or (any(p in dom for p in PORTAL_DOMAINS) and not _APPLICATION.search(sub)):
        return "job_agency_spam", "high"
    if _PROMO.search(sub) and not _APPLICATION.search(sub):
        return "promotional", "low"
    if _APPLICATION.search(sub) and (any(a in dom for a in ATS_DOMAINS) or _APPLICATION.search(body)):
        return "job_related", "high"
    if _APPLICATION.search(sub):
        return "job_related", "low"
    return None, "none"


def weak_label_row(sender, subject, text) -> dict:
    cat, cat_conf = weak_category(sender, subject, text)
    auth = authenticity(sender, subject, text) if cat == "job_related" else ""
    intent, intent_conf = (weak_intent(subject, text) if auth == "real_job_application" else (None, "none"))
    return {"weak_category": cat, "weak_category_conf": cat_conf, "weak_authenticity": auth,
            "weak_intent": intent, "weak_intent_conf": intent_conf}
