"""Stage 1.5: rule-based authenticity filter for emails the model called `job_related`.

Returns one of: real_job_application, job_alert, personal, academic,
account_notification, commerce, other.

Differences from the original notebook:
* The default is `other` (the old version returned `real_job_application` for anything it
  did not recognise, so nothing was ever filtered: the saved "filtered" file was empty).
* Alert subjects are checked first, so a newsletter that merely contains "bewerbung" or
  "interview" no longer overrides the alert rule.
* Keywords are matched on word boundaries ("order" no longer matches "border").
* A single weak word such as "leider" no longer qualifies an email as an application.
"""
from __future__ import annotations

import re

from .preprocess import normalize_umlauts, sender_domain

APPLICATION_PHRASES = [
    "ihre bewerbung", "deine bewerbung", "ihrer bewerbung", "deiner bewerbung",
    "bewerbung als", "bewerbung bei", "bewerbung auf", "your application",
    "application for", "application at", "application as", "subject: application",
    "bewerbung ueber", "bewerbungsprozess", "beworben", "application update", "update on your",
    "eingangsbestaetigung", "career application", "bewerbung an uns",
    "application status", "thank you for applying", "thanks for applying",
    "vorstellungsgespraech", "interview", "absage", "rejection",
    "we regret", "we have received your application", "eingang ihrer bewerbung",
]
ALERT_SUBJECT_PHRASES = [
    "jobs der naechsten woche", "job alert", "jobalert", "jobagent", "neue jobs",
    "new jobs", "jobs for you", "stellenangebote", "empfohlene jobs", "newsletter",
    "ungelesene nachrichten", "unread messages", "jobs in der naehe",
]
PORTAL_DOMAINS = [
    "linkedin.com", "xing.com", "stepstone", "indeed.com", "jooble", "jobvector",
    "jobs2web", "hotelcareer", "sportfive", "monster.", "glassdoor", "kimeta", "jobware",
]
ATS_DOMAINS = [
    "concludis", "successfactors", "workday", "myworkday", "recruitee", "smartrecruiters",
    "greenhouse", "lever.co", "icims", "personio", "join.com", "teamtailor", "softgarden",
    "rexx-systems", "hr4you", "jobs.", "recruiting", "karriere", "bewerbung",
]
FREEMAIL_DOMAINS = ["gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "yahoo.com", "gmx.de", "gmx.net", "web.de"]

ACADEMIC_TERMS = ["university", "universitaet", "campus", "msc", "bsc", "admission", "semester"]
COMMERCE_TERMS = ["order", "delivery", "invoice", "tracking", "payment", "bestellung", "rechnung", "versand"]
ACCOUNT_TERMS = ["password", "passwort", "login", "verify your", "account deleted", "konto geloescht"]


def _norm(x) -> str:
    return normalize_umlauts("" if x is None or (isinstance(x, float) and x != x) else str(x)).lower()


def _phrase_regex(phrases: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + ")")


_APPLICATION = _phrase_regex(APPLICATION_PHRASES)
_ALERT = _phrase_regex(ALERT_SUBJECT_PHRASES)
_ACADEMIC = _phrase_regex(ACADEMIC_TERMS)
_COMMERCE = re.compile(r"\b(?:" + "|".join(COMMERCE_TERMS) + r")\b")
_ACCOUNT = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ACCOUNT_TERMS) + r")\b")


def _domain_in(domain: str, fragments: list[str]) -> bool:
    return any(f in domain for f in fragments)


def _distinct_hits(pattern: re.Pattern, text: str) -> int:
    return len(set(pattern.findall(text)))


def authenticity(sender, subject, text) -> str:
    domain = sender_domain(sender)
    sub, body = _norm(subject), _norm(text)
    app_in_subject = bool(_APPLICATION.search(sub))
    app_hits_body = _distinct_hits(_APPLICATION, body)

    if _ALERT.search(sub):
        return "job_alert"
    if _domain_in(domain, ATS_DOMAINS) and (app_in_subject or app_hits_body >= 1):
        return "real_job_application"
    if _domain_in(domain, PORTAL_DOMAINS):
        # Portals relay genuine application updates ("Neuigkeiten zu Ihrer Bewerbung") but
        # mostly send recommendations.
        return "real_job_application" if app_in_subject else "job_alert"
    if domain in FREEMAIL_DOMAINS and not app_in_subject:
        return "personal"
    if app_in_subject or app_hits_body >= 2:
        return "real_job_application"
    if _distinct_hits(_ACADEMIC, body) >= 2:
        return "academic"
    if _ACCOUNT.search(sub) or _ACCOUNT.search(body):
        return "account_notification"
    if _COMMERCE.search(sub) or _COMMERCE.search(body):
        return "commerce"
    return "other"
