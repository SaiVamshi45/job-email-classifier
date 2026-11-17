"""Gmail access (read-only). Credentials are read from git-ignored paths in config.py."""
from __future__ import annotations

import base64
import json
import logging
import re
import socket
import ssl
import time
from email.utils import parsedate_to_datetime

import httplib2
import pandas as pd

from . import config
from .preprocess import strip_html

log = logging.getLogger(__name__)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="ignore")


def _walk(part: dict, plain: list[str], html: list[str]) -> None:
    data = (part.get("body") or {}).get("data")
    mime = part.get("mimeType", "")
    if data and mime == "text/plain":
        plain.append(_decode(data))
    elif data and mime == "text/html":
        html.append(_decode(data))
    for sub in part.get("parts") or []:
        _walk(sub, plain, html)


def extract_body(payload: dict, snippet: str = "", max_chars: int = config.MAX_BODY_CHARS) -> str:
    """Prefer text/plain, then HTML (tags stripped), then Gmail's snippet. Handles nested multiparts."""
    plain: list[str] = []
    html: list[str] = []
    _walk(payload, plain, html)
    if plain:
        text = "\n".join(plain)
    elif html:
        text = strip_html("\n".join(html))
    else:
        text = snippet or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


HTTP_TIMEOUT_SECONDS = 30


def get_service():
    """Authenticate with a local browser flow (the old out-of-band flow is no longer supported by Google)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if config.GMAIL_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(config.GMAIL_TOKEN), config.GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not config.GMAIL_CREDENTIALS.exists():
                raise FileNotFoundError(
                    f"Put your OAuth client file at {config.GMAIL_CREDENTIALS} (never commit it)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(config.GMAIL_CREDENTIALS), config.GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        config.GMAIL_TOKEN.parent.mkdir(parents=True, exist_ok=True)
        config.GMAIL_TOKEN.write_text(creds.to_json(), encoding="utf-8")
    import google_auth_httplib2
    import httplib2

    # Without a timeout a stalled connection blocks forever (the hang seen in ssl.py recv_into).
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=HTTP_TIMEOUT_SECONDS))
    return build("gmail", "v1", http=http, cache_discovery=False)


def list_message_ids(service, query: str, max_messages: int | None = None) -> list[str]:
    ids: list[str] = []
    token = None
    while True:
        resp = service.users().messages().list(userId="me", q=query, maxResults=100, pageToken=token).execute()
        ids.extend(m["id"] for m in resp.get("messages", []))
        token = resp.get("nextPageToken")
        if not token or (max_messages and len(ids) >= max_messages):
            break
    return ids[:max_messages] if max_messages else ids


def _is_rate_limit(err) -> bool:
    """Gmail reports per-minute quota problems as HTTP 403 (rateLimitExceeded) as well as 429/5xx."""
    status = getattr(err.resp, "status", 0)
    text = str(err)
    return status in (429, 500, 502, 503) or (
        status == 403 and any(k in text for k in ("rateLimitExceeded", "userRateLimitExceeded", "Quota exceeded"))
    )


def _get_with_retry(service, msg_id: str, attempts: int = 7) -> dict:
    """Fetch one message; on quota errors wait 2, 4, 8, ... 64 seconds and try again."""
    from googleapiclient.errors import HttpError

    for n in range(attempts):
        try:
            return service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        except HttpError as err:
            if _is_rate_limit(err) and n < attempts - 1:
                wait = 2 ** (n + 1)
                print(f"  rate limited, waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
        except (OSError, ssl.SSLError, httplib2.HttpLib2Error) as err:  # timeout / dropped connection
            if n < attempts - 1:
                print(f"  network problem ({type(err).__name__}), retrying in 3s ...", flush=True)
                time.sleep(3)
                continue
            raise


def _header(headers: list[dict], name: str, default: str) -> str:
    return next((h["value"] for h in headers if h["name"].lower() == name), default)


def message_to_row(msg: dict) -> dict:
    headers = msg["payload"].get("headers", [])
    date_raw = _header(headers, "date", "")
    try:
        date = parsedate_to_datetime(date_raw).isoformat()
    except (TypeError, ValueError):
        date = date_raw
    return {
        "Id": msg.get("id", ""),
        "Date": date,
        "Sender": _header(headers, "from", ""),
        "Subject": _header(headers, "subject", ""),
        "Email Text": extract_body(msg["payload"], msg.get("snippet", "")),
    }


CACHE_PATH = config.SECRETS_DIR / "mail_cache.jsonl"   # secrets/ is git-ignored; the cache holds email text
REQUEST_PAUSE_SECONDS = 0.25                              # about 4 mails per second, well inside Gmail's quota


def _load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["Id"]] = row
    return cache


def fetch_emails(query: str, max_messages: int | None = 1000) -> pd.DataFrame:
    """Fetch mails matching `query`. Already fetched mails are read from the cache, so re-running is cheap
    and an interrupted run resumes where it stopped."""
    service = get_service()
    print("logged in, listing messages ...", flush=True)
    ids = list_message_ids(service, query, max_messages)
    cache = _load_cache()
    todo = [i for i in ids if i not in cache]
    print(f"{len(ids)} mails matched, {len(ids) - len(todo)} already cached, {len(todo)} to fetch", flush=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        for i, mid in enumerate(todo, 1):
            try:
                row = message_to_row(_get_with_retry(service, mid))
                cache[mid] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            except Exception as exc:  # log instead of silently skipping
                failed += 1
                log.warning("could not fetch message %s: %s", mid, str(exc)[:120])
            time.sleep(REQUEST_PAUSE_SECONDS)
            if i % 50 == 0:
                print(f"fetched {i}/{len(todo)}", flush=True)
    if failed:
        log.warning("%d of %d messages could not be fetched; run the command again to retry them", failed, len(todo))
    rows = [cache[i] for i in ids if i in cache]
    return pd.DataFrame(rows, columns=["Id", "Date", "Sender", "Subject", "Email Text"])