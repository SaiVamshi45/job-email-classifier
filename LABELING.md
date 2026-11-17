# Labelling guide

Inconsistent definitions were the main reason the original data was unreliable (for example, plain "we received your application" mails labelled `next_step`). Use these definitions for every new label.

## Stage 1: category of the mail

| Label | Use when |
|---|---|
| `job_related` | Mail about **your own application or a hiring process**: confirmation, update, rejection, interview invitation, request for documents, offer, welcome message, and the account mails of an applicant portal. Includes confirmations relayed by LinkedIn / Indeed / Hotelcareer ("your application was sent", "Bewerbung über Indeed"). |
| `job_agency_spam` | Job alerts, recommendations and newsletters from portals and agencies ("New jobs for you", "Neue Stellenausschreibungen", Stepstone / Xing / Glassdoor digests), recruiter mass mail. |
| `promotional` | Marketing that is not about a specific job: sales, tips newsletters (Jobscan), webinars, courses. |
| `non_job` | Everything else: housing, university admission, visa, orders, banking, private mail. A housing or university "application" is `non_job`. |

## Stage 2: intent of a real application mail

| Label | Use when |
|---|---|
| `acknowledgment` | The employer or portal confirms the application was received or is being reviewed. No decision, and nothing for you to do. |
| `rejection` | A decision against you: "unfortunately", "we will not move forward", "position filled", "we paused recruiting". |
| `interview` | An invitation to, scheduling of, reminder of or confirmation of a conversation or interview. |
| `next_step` | **You** must do something: take-home challenge, assessment, upload or update a document, complete the application, register in a portal, confirm your e-mail. |

Rules of thumb:
* If a mail both thanks you and rejects you, it is `rejection`.
* "We will contact you with next steps" is `acknowledgment`: it asks nothing of you.
* If you cannot decide, leave the row empty in the review queue. A skipped row is better than a wrong one.

## Workflow that scales

1. `python scripts/build_dataset.py --gmail-query "after:2025/08/01 -in:sent" --max-messages 3000`
2. Open `data/review_queue.csv`; work through the rows with `review_priority` 1, then 2. Write the correct label into `human_category` / `human_intent`.
3. `python scripts/merge_reviewed.py --queue data/review_queue.csv --accept-unreviewed`
4. `python scripts/audit_labels.py` to list old rows that contradict a confident rule, fix them in `data/label_audit_stage*.csv`, then `--apply`.
5. `python scripts/train.py --stage stage2 --backend baseline cnn` and `python scripts/eval_gold.py --backend baseline cnn`.

Keep `data/gold_seed.csv` out of training. It is your independent test set, and you should extend it by hand whenever you can.
