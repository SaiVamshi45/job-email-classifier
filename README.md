# JobTracker: job-application email classifier

Reads your Gmail (read-only) and sorts mail in three steps:

1. **Stage 1**: category (`job_related`, `job_agency_spam`, `promotional`, `non_job`). CNN, with XLM-R as optional fallback when the CNN is unsure.
2. **Stage 1.5**: rules separate genuine application mail from alerts, personal, commerce, etc.
3. **Stage 2**: intent of genuine applications (`acknowledgment`, `interview`, `next_step`, `rejection`), or `uncertain`.

See [FLAWS.md](FLAWS.md) for what was wrong in the original notebook and what changed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # baseline + Gmail + tests
pip install -r requirements-neural.txt     # optional: CNN and XLM-R (TensorFlow, PyTorch, Transformers)
pytest
```

## Train

```bash
# 1. clean the raw labelled exports (removes duplicates / conflicts, drops personal addresses)
python scripts/clean_data.py --stage1 raw/Stage1_ideal_balanced_final.csv --stage2 raw/stage2_final_clean.csv

# 2. train and evaluate on a leakage-free split (metrics in reports/*.json)
python scripts/train.py --stage stage1 --backend baseline cnn xlmr
python scripts/train.py --stage stage2 --backend baseline cnn xlmr

# optional: see how much the original protocol inflated scores
python scripts/leakage_check.py --raw raw/Stage1_ideal_balanced_final.csv --label category
```

On Colab use `--backend xlmr` with a GPU runtime (omit `--cpu`).

## Improving the dataset (the biggest lever)

The original data had duplicates, inconsistent label definitions and almost no modern ATS templates (softgarden, Personio, join.com, Ashby, SuccessFactors ...). Tools for fixing that:

| Tool | What it does |
|---|---|
| `data/gold_seed.csv` | 72 recent real mails from your mailbox, hand-labelled (category, Stage-1.5 verdict, intent). Independent test set, **never train on it**. |
| `scripts/eval_gold.py` | Scores rules, weak labellers and trained models against the gold set. |
| `scripts/audit_labels.py` | Lists old training rows that contradict a confident rule, so you can fix or confirm them. |
| `scripts/build_dataset.py` | Pulls your whole mailbox, pre-labels it, and writes a review queue sorted by how much a human is needed. |
| `scripts/merge_reviewed.py` | Merges the reviewed queue into `data/stage*_clean.csv`. |
| `scripts/remove_gold_from_training.py` | One-off cleanup that removes mailbox-reviewed rows whose subject template is in the gold set. `merge_reviewed.py` now excludes gold mails by id and subject, so they cannot leak into training. |
| `LABELING.md` | Exact label definitions to apply, so labels stay consistent. |

## Results (honest numbers)

Trained on the reviewed mailbox data (Stage 1: 5,895 rows, Stage 2: 634 rows; grouped splits so mails from one template never sit in both train and test). The gold set is 72 recent real mails that never enter training.

| | Held-out test | Gold (72 real mails) |
|---|---|---|
| Stage 1, TF-IDF + LogReg | 0.910 accuracy, 0.898 macro-F1 (1,180 mails) | 0.972 accuracy, 0.718 macro-F1 |
| Stage 1, TextCNN | not retrained on the final data | 0.847 accuracy (older data) |
| Stage 2, TF-IDF + LogReg | 0.542 accuracy, 0.553 macro-F1 (131 mails) | 0.641 accuracy |
| Stage 2, TextCNN | 0.557 accuracy, 0.559 macro-F1 | 0.828 accuracy, 0.398 macro-F1 |

Read these with care. About 41% of the Stage 1 labels come from the keyword labeller, so the held-out test partly measures agreement with it. The gold set has 50 of 64 intents as acknowledgment, 4 next steps and 1 interview, so Stage 2 is **not** reliable on `next_step` and `interview` (0 correct on gold) and its scores move by about 0.1 between retrains. The original 94.2% figure came from a split that leaked duplicates and should not be quoted.

## Run

```bash
# Gmail: put your OAuth client file at secrets/credentials.json first (never commit it)
python scripts/run_pipeline.py --gmail-query "after:2025/11/01" --backend cnn --fallback xlmr

# or any CSV with Subject and "Email Text" columns
python scripts/run_pipeline.py --input-csv emails.csv --backend baseline
```

Results go to `outputs/` (labels only, no email bodies).

## Serve it and CI (all free)

```bash
pip install -r requirements-api.txt
PYTHONPATH=src uvicorn jobtracker.api:app --port 8000      # Windows PowerShell: $env:PYTHONPATH="src"
curl -X POST localhost:8000/classify -H "Content-Type: application/json" \
  -d '{"emails":[{"subject":"Ihre Bewerbung","body":"leider müssen wir absagen","sender":"jobs@firma.de"}]}'
```

* `GET /health` says whether the models are loaded; `POST /classify` takes up to 50 mails and returns category, authenticity and intent with scores. It never logs mail content and never touches Gmail.
* `JOBTRACKER_BACKEND=baseline|cnn|xlmr` picks the model (the baseline needs no TensorFlow). Set `JOBTRACKER_API_KEY` to require an `X-API-Key` header.
* Docker: `docker build -t jobtracker-api .` then `docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" jobtracker-api`. The models are mounted, not baked into the image, because they are derived from private mail.
* CI: `.github/workflows/ci.yml` runs the tests and builds the image on every push. GitHub Actions is free for public repositories.

## Layout

```
src/jobtracker/  config, preprocess, data (cleaning + grouped split), tokenizer, predictors,
                 classify, rules (Stage 1.5), gmail, pipeline, evaluate
scripts/         clean_data, train, run_pipeline, leakage_check
tests/           24 unit tests (no model downloads needed)
secrets/ data/ models/ outputs/   git-ignored
```

## Before you publish this on GitHub

Do not commit `secrets/`, `data/`, `raw/`, `outputs/` or notebooks with real email output. They contain personal data. Rotate the old Google OAuth client secret, which was part of the original zip.
