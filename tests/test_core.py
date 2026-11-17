import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobtracker import config, gmail, pipeline
from jobtracker.classify import classify
from jobtracker.data import clean_raw, make_splits
from jobtracker.preprocess import build_model_text, clean_text, normalize_subject, sender_domain
from jobtracker.rules import authenticity
from jobtracker.tokenizer import WordTokenizer


# ---------------------------------------------------------------- preprocessing
def test_clean_text_masks_urls_emails_and_umlauts():
    out = clean_text("Hallo <b>Müller</b>, siehe https://x.de/a?b=1 oder mail an a.b@firma.de!")
    assert "url" in out and "email" in out
    assert "mueller" in out and "<b>" not in out and "http" not in out


def test_clean_text_handles_none_and_nan():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_subject_normalisation_and_domain():
    assert normalize_subject("AW: Re: Ihre Bewerbung 12345") == "ihre bewerbung 0"
    assert sender_domain('"HR" <jobs@Firma.DE>') == "firma.de"
    assert build_model_text("Betreff", "Text") == "betreff text"


# ---------------------------------------------------------------- tokenizer (train/inference consistency)
def test_tokenizer_pads_and_truncates_at_the_end():
    tok = WordTokenizer(100).fit(["alpha beta gamma", "alpha beta"])
    enc = tok.encode(["alpha beta", "alpha beta gamma delta epsilon"], maxlen=4)
    assert enc.shape == (2, 4)
    assert enc[0, 0] != 0 and enc[0, 2] == 0 and enc[0, 3] == 0   # post-padding
    assert (enc[1] != 0).all()                                     # post-truncation keeps the start
    assert enc[1, 3] == 1                                          # unseen word -> OOV id


def test_tokenizer_roundtrip(tmp_path):
    tok = WordTokenizer(50).fit(["eins zwei drei"])
    tok.save(tmp_path / "t.json")
    again = WordTokenizer.load(tmp_path / "t.json")
    assert (tok.encode(["eins drei"], 5) == again.encode(["eins drei"], 5)).all()


# ---------------------------------------------------------------- stage 1.5 rules
@pytest.mark.parametrize(
    "sender,subject,text,expected",
    [
        ('"KFC" <no-reply-kfc@concludis.de>', "Deine Bewerbung bei KFC", "vielen dank", "real_job_application"),
        ("SPORTFIVE <events@jobs.sportfive.com>", "Jobs der nächsten Woche", "bewerbung interview absage", "job_alert"),
        ("LinkedIn <jobs@linkedin.com>", "New jobs for you", "apply now", "job_alert"),
        ("Indeed <noreply@indeed.com>", "Neuigkeiten zu Ihrer Bewerbung von Bäckerei X", "", "real_job_application"),
        ("Indeed <alert@match.indeed.com>", "Software Engineer in Berlin", "viele jobs", "job_alert"),
        ("Vivek <v@gmail.com>", "Re: Studentisches Wohnen - fehlende Unterlagen", "leider fehlen unterlagen", "personal"),
        ("Shop <x@shop.de>", "Your order has shipped", "tracking delivery", "commerce"),
        ("Someone <x@random.org>", "Hello", "nothing relevant here", "other"),
        ("Fa <hr@firma.de>", "Rückmeldung", "ihre bewerbung als dev. leider absage", "real_job_application"),
    ],
)
def test_authenticity_rules(sender, subject, text, expected):
    assert authenticity(sender, subject, text) == expected


def test_unknown_mail_is_not_passed_through_by_default():
    assert authenticity("a@b.de", "Hi", "kein bezug") != "real_job_application"


def test_word_boundaries_in_commerce_rule():
    assert authenticity("a@b.de", "border crossing", "the border is open") == "other"


# ---------------------------------------------------------------- weak labellers
from jobtracker.weak_labels import weak_category, weak_intent


@pytest.mark.parametrize(
    "subject,text,expected",
    [
        ("Ihre Bewerbung", "leider müssen wir Ihnen mitteilen dass wir Sie nicht berücksichtigen", "rejection"),
        ("Application update", "after reviewing your CV we won't be moving forward", "rejection"),
        ("We received your application", "thank you for applying, we will review it", "acknowledgment"),
        ("Re: Application", "The next step is our take-home challenge", "next_step"),
        ("Termin zum Vorstellungsgespräch", "bitte wählen Sie einen Termin", "interview"),
        ("Willkommen", "wir würden Sie gern zum Interview einladen: invite you to an interview", "interview"),
    ],
)
def test_weak_intent(subject, text, expected):
    assert weak_intent(subject, text)[0] == expected


def test_weak_intent_mention_of_interview_is_not_an_invitation():
    assert weak_intent("Thank you for your application", "we received it; we may invite you for a conversation later")[0] == "acknowledgment"


def test_weak_category_signals():
    assert weak_category("a@jobagent.stepstone.de", "New job opportunities for you", "")[0] == "job_agency_spam"
    assert weak_category("a@firma.de", "Ihre Bewerbung bei uns", "vielen dank für ihre bewerbung")[0] == "job_related"
    assert weak_category("a@studentenwerk.de", "Bewerbung Wohnheim", "ihr antrag ist eingegangen")[0] == "non_job"
    assert weak_category("a@x.de", "Hallo", "nichts")[0] is None


# ---------------------------------------------------------------- data cleaning and splitting
def _raw(n=60):
    rows = []
    for i in range(n):
        label = ["a", "b", "c"][i % 3]
        tag = "abcdefghij"[(i // 2) % 10] + "klmnopqrst"[(i // 20) % 10]   # digit-free, shared by pairs of rows
        rows.append({"Sender": f"x{i}@d{i % 5}.de", "Subject": f"Betreff vorlage {tag}",
                     "preprocessed_text": f"text {label} wort{i} " + "füllung " * 4, "category": label})
    rows.append(rows[0].copy())                                  # exact duplicate
    rows.append({**rows[1], "category": "b" if rows[1]["category"] != "b" else "c"})  # conflicting label
    rows.append({**rows[2], "preprocessed_text": "kurz", "Subject": "s"})            # too short
    rows.append({**rows[3], "category": "non_nob", "preprocessed_text": "text non_job drei vier fuenf"})
    return pd.DataFrame(rows)


def test_clean_raw_removes_duplicates_conflicts_and_fixes_labels():
    clean, stats = clean_raw(_raw(), "preprocessed_text", "category")
    assert clean["text"].is_unique
    assert "non_nob" not in set(clean["label"]) and "non_job" in set(clean["label"])
    assert stats["conflicting_label_texts_dropped"] == 1
    assert stats["exact_duplicates_dropped"] >= 1
    assert not clean["sender_domain"].str.contains("@").any()


def test_splits_have_no_group_overlap():
    clean, _ = clean_raw(_raw(120), "preprocessed_text", "category")
    train, val, test = make_splits(clean, seed=1)
    for a, b in [(train, test), (val, test), (train, val)]:
        assert not set(a["group"]) & set(b["group"])
    assert len(train) + len(val) + len(test) == len(clean)


# ---------------------------------------------------------------- gmail body extraction
def _b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def test_extract_body_nested_multipart_prefers_plain():
    payload = {"mimeType": "multipart/mixed", "parts": [
        {"mimeType": "multipart/alternative", "parts": [
            {"mimeType": "text/html", "body": {"data": _b64("<p>html version</p>")}},
            {"mimeType": "text/plain", "body": {"data": _b64("plain version")}},
        ]}]}
    assert gmail.extract_body(payload) == "plain version"


def test_extract_body_html_only_and_snippet_fallback():
    html_only = {"mimeType": "multipart/alternative", "parts": [
        {"mimeType": "text/html", "body": {"data": _b64("<div>Ihre <b>Bewerbung</b></div>")}}]}
    assert gmail.extract_body(html_only) == "Ihre Bewerbung"
    assert gmail.extract_body({"mimeType": "multipart/mixed", "parts": []}, snippet="nur snippet") == "nur snippet"


def test_extract_body_truncates():
    payload = {"mimeType": "text/plain", "body": {"data": _b64("x" * 5000)}}
    assert len(gmail.extract_body(payload)) == config.MAX_BODY_CHARS


# ---------------------------------------------------------------- classify + pipeline with stub models
class Stub:
    def __init__(self, labels, rule, dtype="float32"):  # float32 like Keras output (regression test)
        self.labels, self.rule, self.dtype = labels, rule, dtype

    def predict_proba(self, texts):
        return np.array([self.rule(t) for t in texts], dtype=self.dtype)


def test_classify_fallback_and_abstain():
    primary = Stub(["x", "y"], lambda t: [0.55, 0.45])
    fallback = Stub(["x", "y"], lambda t: [0.1, 0.9])
    out = classify(["a", "b"], primary, fallback, fallback_below=0.7)
    assert list(out["label"]) == ["y", "y"] and out.attrs["fallback_rate"] == 1.0
    out = classify(["a"], primary, None, abstain_below=0.6)
    assert out.loc[0, "label"] == "uncertain"


def test_classify_rejects_mismatched_label_sets():
    with pytest.raises(ValueError):
        classify(["a"], Stub(["x", "y"], lambda t: [.5, .5]), Stub(["x", "z"], lambda t: [.5, .5]), 0.9)


def test_pipeline_end_to_end_with_stubs(tmp_path):
    emails = pd.DataFrame({
        "Date": ["d1", "d2", "d3"],
        "Sender": ['"KFC" <no-reply@concludis.de>', "SPORTFIVE <e@jobs.sportfive.com>", "Shop <a@shop.de>"],
        "Subject": ["Deine Bewerbung bei KFC", "Jobs der nächsten Woche", "Sale"],
        "Email Text": ["leider absage", "viele jobs", "rabatt"],
    })
    s1 = Stub(["job_related", "promotional"], lambda t: [0.9, 0.1] if ("bewerbung" in t or "jobs" in t) else [0.1, 0.9])
    s2 = Stub(["acknowledgment", "rejection"], lambda t: [0.1, 0.9])
    res = pipeline.run(emails, s1, s2)
    assert list(res["authenticity"]) == ["real_job_application", "job_alert", ""]
    assert list(res["predicted_intent"]) == ["rejection", "", ""]
    paths = pipeline.save(res, "test", tmp_path)
    saved = pd.read_csv(paths["all"])
    assert "preprocessed_text" not in saved.columns and "Email Text" not in saved.columns
    assert len(pd.read_csv(paths["applications"])) == 1 and len(pd.read_csv(paths["filtered"])) == 1


# ---------------------------------------------------------------- gmail quota handling
def test_rate_limit_403_is_retried(monkeypatch):
    from googleapiclient.errors import HttpError

    class Resp(dict):
        status = 403
        reason = "Forbidden"

    err = HttpError(Resp(), b'{"error": {"errors": [{"reason": "rateLimitExceeded"}]}} Quota exceeded for quota metric')
    assert gmail._is_rate_limit(err)

    calls = {"n": 0}

    class Exec:
        def execute(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise err
            return {"id": "x"}

    class Svc:
        def users(self): return self
        def messages(self): return self
        def get(self, **kw): return Exec()

    monkeypatch.setattr(gmail.time, "sleep", lambda s: None)
    assert gmail._get_with_retry(Svc(), "x") == {"id": "x"} and calls["n"] == 3


def test_other_403_is_not_retried():
    from googleapiclient.errors import HttpError

    class Resp(dict):
        status = 403
        reason = "Forbidden"

    assert not gmail._is_rate_limit(HttpError(Resp(), b"Insufficient Permission"))


def test_cap_groups_limits_each_template_and_label():
    from jobtracker.data import cap_groups

    df = pd.DataFrame({"group": ["a"] * 20 + ["b"] * 3 + ["a"] * 2, "label": ["x"] * 23 + ["y"] * 2,
                       "text": [str(i) for i in range(25)]})
    out = cap_groups(df, 5, seed=1)
    counts = out.groupby(["group", "label"]).size().to_dict()
    assert counts == {("a", "x"): 5, ("a", "y"): 2, ("b", "x"): 3}
    assert len(cap_groups(df, 0)) == len(df)
