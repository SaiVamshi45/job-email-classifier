# Flaws found in the original JobTrackerAPP, and what changed

Severity: **Critical** = wrong results or a security problem, **High** = misleading or fragile, **Medium** = quality / maintainability.

## Security and privacy

| # | Severity | Flaw | Fix |
|---|---|---|---|
| 1 | Critical | `credentials.json` (real OAuth `client_id` and `client_secret`) was inside the project folder. | Not shipped. Paths come from `secrets/` or env vars and are git-ignored. **Rotate the secret in Google Cloud Console** (APIs & Services → Credentials → reset secret) because it has been in a zip. |
| 2 | High | Tokens were stored with `pickle` (`token.pickle`). Unpickling a tampered file executes code. | `token.json` via `Credentials.to_json()`. |
| 3 | High | Result CSVs hold real senders, subjects and message content, including **another person's mailbox** (`ResultwithVivekemail`). Pushing this to GitHub would be a GDPR problem. | Outputs now contain only Date / Sender / Subject / labels, never bodies. Training data keeps only the sender *domain*. `data/`, `outputs/`, `models/`, `raw/` are git-ignored. |
| 4 | High | Gmail login used the out-of-band flow (`urn:ietf:wg:oauth:2.0:oob`) and `run_console()`. Google blocked OOB in 2022-23, so auth fails on a fresh setup. | `InstalledAppFlow.run_local_server()`. |

## Model and evaluation bugs

| # | Severity | Flaw | Evidence | Fix |
|---|---|---|---|---|
| 5 | Critical | **Train/inference mismatch in the CNN.** Training used `padding='post'`; inference called `pad_sequences(..., maxlen=300)` with the default `padding='pre'` (cells 42 and 47). The model sees differently shaped input than it was trained on. | code | One tokenizer `encode()` (post-pad, post-truncate) shared by training and inference (`tokenizer.py`), plus a unit test. Predictors are reloaded from disk and evaluated in `train.py`. |
| 6 | Critical | **Data leakage.** 19% of Stage-1 rows (about 1,150 of 5,956) and 9% of Stage-2 rows (36 of 400) are exact duplicates. A random split put copies of the same email in train and test. 4 texts carry conflicting labels. | `scripts/leakage_check.py`: Stage 1 macro-F1 0.94 → 0.87, Stage 2 0.94 → 0.55 with the same simple model once duplicates are removed and templates are kept together. | `clean_raw()` removes duplicates and conflicts. `make_splits()` is stratified and grouped by subject template, with asserts that no group crosses a split. |
| 7 | Critical | **Stage 1.5 filter never filtered.** `authenticity()` ended with `return "real_job_application"`, and a hard override fired on any mail containing "bewerbung", "interview" or "leider" *before* the alert check. The saved `stage1_5_filtered` file for the second mailbox is empty. Newsletters ("Jobs der nächsten Woche"), "Du hast seit 24h ungelesene Nachrichten" and a private housing mail all reached Stage 2 (48 of 154 rows there were alert or personal senders). | result CSVs | Rewritten in `rules.py`: alert subjects first, default `other`, word-boundary matching, weak words removed. On those 154 mails, 57 are now routed out. It is still a heuristic: see "Still open". |
| 8 | High | **Labels lost.** The Stage-1 XLM-R model had no `id2label`, so the pipeline returned `LABEL_0…LABEL_4`; the saved results contain those (up to 63 rows) and a `non_nob` typo (33 rows). Cell 42 silently dropped them, cell 47 kept them. | result CSVs | Labels are stored in `labels.json` and applied by the code. `id2label` is set on the model. Label fallbacks refuse mismatched label sets. |
| 9 | High | Stage-2 report (cell 36) uses `label_names` from **Stage 1** (same count of 4 classes, so no error), so intents are printed as `job_related`, `non_job`… | code | Labels always come from the stage's own data. |
| 10 | High | Stage 2 was forced to pick an intent for every mail, with no "not an application" or "unsure" outcome. | code | `STAGE2_ABSTAIN_BELOW` returns `uncertain` for low-confidence mails. |
| 11 | High | Stage-2 XLM-R trained with `MAX_LEN=128` but inference used the pipeline default (up to 512); training had no evaluation or best-model selection, no seed, `no_cuda=True` (deprecated) while comments said GPU. | code | Same `max_len` in `labels.json` for training and inference, per-epoch evaluation, best model by macro-F1, seeds, `use_cpu` flag. |
| 12 | Medium | Class imbalance ignored (Stage 2: rejection 41%, interview 13%) and only accuracy reported in places. | data | Class-weighted loss, macro-F1 and per-class metrics. |
| 13 | Medium | `valid_idx = y_test < num_classes` (cell 14) hid a label mismatch instead of fixing it. | code | Removed. |
| 14 | Medium | CNN fallback threshold of 0.7 is arbitrary and the fallback rate is never reported. | code | Threshold in config; fallback rate logged. |

## Engineering

| # | Severity | Flaw | Fix |
|---|---|---|---|
| 15 | Medium | One 50-cell notebook with hard-coded `/content/...` paths, Colab-only `files.download`, repeated pip installs, duplicated functions and globals overwritten between cells (`le`, `df`, `label_names`). | `src/jobtracker/` package, 3 scripts, one config file. |
| 16 | Medium | Bare `except: continue` dropped emails silently; no retry on Gmail 429/5xx; HTML-only mails fell back to the snippet; nested multipart bodies were missed. | `gmail.py` logs failures, retries, handles nested multipart and HTML. |
| 17 | Medium | Deprecated or fragile APIs: `Tokenizer`, `.h5`, `input_length`, `XLMRobertaTokenizer`, `tokenizer=` in `Trainer`. | Own tokenizer, `.keras`, `AutoTokenizer`, custom collator. |
| 18 | Medium | No tests, no requirements file, no README, no `.gitignore`. | Added (34 tests). |

## Dataset work done in round 2

* **Stage 2 labels were inconsistent.** 44 of the 81 original `next_step` rows were plain "we received your application" acknowledgments, and 5 `rejection` rows were requests to update a document. These 49 labels were corrected (backup of the originals kept locally in `data/_backup_original/`). Definitions are now fixed in `LABELING.md`.
* **Gold set.** 72 real, recent mails from your mailbox, labelled by hand: `data/gold_seed.csv`. Before / after correcting Stage 2 labels, accuracy on it: TF-IDF 0.25 → 0.50, CNN 0.52 → 0.80. Its intent mix is 50 acknowledgments, 9 rejections, 4 next steps and 1 interview, so macro-F1 (0.24 / 0.35) is noisy and rare classes are still weak.
* **Weak labellers** (`weak_labels.py`) agree with your original labels on 80% of the rows they are confident about for Stage 2 (74% coverage) and 90% for Stage 1 (45% coverage). On the gold set they score 96-100%, but I tuned them while looking at that set, so ignore that figure. Use them for pre-labelling and review, not as ground truth.
* **Stage 1 audit**: 206 old rows disagree with a confident rule (`data/label_audit_stage1.csv`). In my 14-row spot check about 2 in 3 flags were genuine errors (Indeed / Hotelcareer application confirmations filed as spam, job newsletters filed as job-related), so these need your review and were **not** applied automatically.

## Round 3: mailbox run

* Fetched 3,000 mails (Gmail quota errors fixed: retry with backoff, 30 s timeout, resumable cache) and reviewed them into the training files.
* **Gold-set leakage found and closed.** The gold mails come from the same mailbox as the review queue, so nothing stopped them from entering training. `merge_reviewed.py` now excludes gold mails by id and subject template, and `remove_gold_from_training.py` removed 85 Stage 1 and 25 Stage 2 rows that had already been merged. After the cleanup the Stage 2 CNN moved from 0.859 to 0.828 on gold, so the earlier figure was probably a little optimistic.
* **Near-duplicate templates dominated the test set.** One DHL auto-reply (54 copies) was 54 of the 55 acknowledgment rows in the Stage 2 test, so a model that had never seen it scored an acknowledgment F1 of about 0.03. Training and testing now cap each subject template at 12 mails per label (`GROUP_CAP`, `--group-cap`). Stage 2 CNN test macro-F1 went from 0.56 to 0.83 on a fairer split.
* Final numbers are in the README. Stage 1 is usable. Stage 2 is still weak on `next_step` and `interview`.

## Still open (needs your input, not just code)

* **More interviews and next steps.** Even after the mailbox run there are about 50 interviews and 43 next steps in Stage 2, and only 1 and 4 in the gold set. Search the mailbox for "Einladung", "Vorstellungsgespräch", "interview", "nächste Schritte", label them, and add some to `data/gold_seed.csv`.
* **Stage 1 labels from the weak labeller.** About 41% of the Stage 1 rows were accepted without review (`--accept-unreviewed`); `job_agency_spam` is probably over-called.
* **Stage 1.5 rules are heuristics.** Label ~200 mails as `application / not application` and replace the rules with a model, or at least measure them.
* **Stage 1 labels look noisy.** 89 `job_agency_spam` rows contain "ihre bewerbung" (43 from indeed.com), which are probably genuine application updates.
* XLM-R code could not be trained here (no access to the Hugging Face hub from the sandbox). The CNN and TF-IDF paths were run end to end; run `python scripts/train.py --stage stage1 --backend xlmr` on your GPU / Colab.