"""Central configuration: paths, thresholds and per-stage settings.

Nothing here is secret. Gmail credentials are read from the paths below, which are
git-ignored, or from the GMAIL_CREDENTIALS / GMAIL_TOKEN environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

SECRETS_DIR = Path(os.environ.get("JOBTRACKER_SECRETS", ROOT / "secrets"))
GMAIL_CREDENTIALS = Path(os.environ.get("GMAIL_CREDENTIALS", SECRETS_DIR / "credentials.json"))
GMAIL_TOKEN = Path(os.environ.get("GMAIL_TOKEN", SECRETS_DIR / "token.json"))
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

SEED = 42

# Mislabelled values found in the original data / results.
LABEL_FIXES = {"non_nob": "non_job"}

# The training data was built from the first 1500 characters of each email, so inference
# must truncate in the same way, otherwise the input distribution shifts.
MAX_BODY_CHARS = 1500

# Stage 1: the CNN decides; XLM-R is only consulted when the CNN is unsure.
CNN_FALLBACK_BELOW = 0.70

# Stage 2: below this probability the email is reported as "uncertain" instead of being
# forced into one of the four intents.
STAGE2_ABSTAIN_BELOW = 0.50

STAGES = {
    "stage1": {
        "data_file": "stage1_clean.csv",
        "cnn_maxlen": 300,
        "cnn_max_words": 30000,
        "xlmr_maxlen": 256,
        "xlmr_epochs": 3,
    },
    "stage2": {
        "data_file": "stage2_clean.csv",
        "cnn_maxlen": 200,
        "cnn_max_words": 20000,
        "xlmr_maxlen": 256,
        "xlmr_epochs": 6,
    },
}
