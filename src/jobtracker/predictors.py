"""Model back-ends behind one interface: `.labels` and `.predict_proba(texts) -> (n, k) array`.

Heavy libraries (tensorflow, torch, transformers) are imported lazily so the rest of the
project, tests and the TF-IDF baseline run without them.

Every model directory contains `labels.json` ({"labels": [...], "max_len": N}). Predictions
are mapped back to label names through that file, never through the generic "LABEL_0..."
names that Hugging Face pipelines emit when id2label is not set (the original notebook's
Stage-1 results contained LABEL_0 ... LABEL_4 rows for that reason).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .data import class_weights
from .tokenizer import WordTokenizer


def _write_meta(out_dir: Path, labels: list[str], max_len: int | None) -> None:
    (out_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "max_len": max_len}, ensure_ascii=False), encoding="utf-8"
    )


def _read_meta(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "labels.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ TF-IDF baseline
class BaselinePredictor:
    def __init__(self, model_dir):
        import joblib

        self.dir = Path(model_dir)
        self.labels = _read_meta(self.dir)["labels"]
        self.pipe = joblib.load(self.dir / "model.joblib")
        if list(self.pipe.classes_) != list(range(len(self.labels))):
            raise ValueError("class order in model.joblib does not match labels.json")

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipe.predict_proba([str(t) for t in texts])


def train_baseline(train_df, labels: list[str], out_dir) -> None:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = train_df["label"].map({l: i for i, l in enumerate(labels)})
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    pipe.fit(train_df["text"].astype(str), y)
    joblib.dump(pipe, out_dir / "model.joblib")
    _write_meta(out_dir, labels, None)


# ------------------------------------------------------------------ Keras CNN
def build_cnn(vocab_size: int, maxlen: int, n_classes: int, embed_dim: int = 128):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = layers.Input(shape=(maxlen,), dtype="int32")
    x = layers.Embedding(vocab_size, embed_dim)(inp)
    pooled = [layers.GlobalMaxPooling1D()(layers.Conv1D(128, k, activation="relu")(x)) for k in (3, 4, 5)]
    x = layers.concatenate(pooled)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(train_df, val_df, labels, out_dir, maxlen, max_words, epochs=20, batch_size=32, seed=config.SEED):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l2i = {l: i for i, l in enumerate(labels)}
    tok = WordTokenizer(max_words).fit(train_df["text"])
    x_tr, x_va = tok.encode(train_df["text"], maxlen), tok.encode(val_df["text"], maxlen)
    y_tr, y_va = train_df["label"].map(l2i).to_numpy(), val_df["label"].map(l2i).to_numpy()
    model = build_cnn(tok.vocab_size, maxlen, len(labels))
    model.fit(
        x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
        class_weight=class_weights(y_tr, len(labels)), verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    model.save(out_dir / "model.keras")
    tok.save(out_dir / "tokenizer.json")
    _write_meta(out_dir, labels, maxlen)


class CnnPredictor:
    def __init__(self, model_dir):
        from tensorflow import keras

        self.dir = Path(model_dir)
        meta = _read_meta(self.dir)
        self.labels, self.max_len = meta["labels"], meta["max_len"]
        self.tok = WordTokenizer.load(self.dir / "tokenizer.json")
        self.model = keras.models.load_model(self.dir / "model.keras")

    def predict_proba(self, texts) -> np.ndarray:
        return self.model.predict(self.tok.encode(texts, self.max_len), verbose=0)


# ------------------------------------------------------------------ XLM-R
def train_xlmr(train_df, val_df, labels, out_dir, max_len, epochs=3, lr=2e-5, batch_size=8,
               seed=config.SEED, use_cpu=False, base="xlm-roberta-base"):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l2i = {l: i for i, l in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(base)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tokenizer(df["text"].astype(str).tolist(), truncation=True, max_length=max_len)
            self.y = df["label"].map(l2i).tolist()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    tr, va = DS(train_df), DS(val_df)
    cw = torch.tensor(list(class_weights(np.array(tr.y), len(labels)).values()), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, y, weight=cw.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, pred),
                "f1_macro": f1_score(p.label_ids, pred, average="macro")}

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=l2i,
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=16, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True, save_total_limit=1,
        seed=seed, report_to="none", use_cpu=use_cpu, logging_steps=25,
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    _write_meta(out_dir, labels, max_len)


class XlmrPredictor:
    def __init__(self, model_dir, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dir = Path(model_dir)
        meta = _read_meta(self.dir)
        self.labels, self.max_len, self.batch_size = meta["labels"], meta["max_len"], batch_size
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.dir)).to(self.device).eval()

    def predict_proba(self, texts) -> np.ndarray:
        texts = [str(t) for t in texts]
        chunks = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(texts[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                chunks.append(self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, len(self.labels)))


BACKENDS = {"baseline": BaselinePredictor, "cnn": CnnPredictor, "xlmr": XlmrPredictor}


def model_dir(stage: str, backend: str) -> Path:
    return config.MODELS_DIR / stage / backend


def load_predictor(stage: str, backend: str):
    return BACKENDS[backend](model_dir(stage, backend))
