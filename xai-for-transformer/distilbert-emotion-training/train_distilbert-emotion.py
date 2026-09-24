"""Fine-tune distilbert-base-uncased on dair-ai/emotion (6-class emotion classification).

Produces the checkpoint used by the XAI transformer tutorials.
Set SMOKE=1 for a tiny end-to-end sanity run.
"""
import os, inspect
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

SMOKE = os.environ.get("SMOKE") == "1"
CKPT = "distilbert-base-uncased"
OUTDIR = os.environ.get("OUTDIR", "./distilbert-emotion" if not SMOKE else "./smoke-out")
set_seed(42)

print(f">> loading dataset dair-ai/emotion (split)  [SMOKE={SMOKE}]", flush=True)
ds = load_dataset("dair-ai/emotion", "split")
labels = ds["train"].features["label"].names
num_labels = len(labels)
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}
print(">> labels:", labels, flush=True)

tok = AutoTokenizer.from_pretrained(CKPT)
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)
collator = DataCollatorWithPadding(tokenizer=tok)

train_ds, eval_ds, test_ds = ds["train"], ds["validation"], ds["test"]
if SMOKE:
    train_ds = train_ds.select(range(200))
    eval_ds = eval_ds.select(range(100))
    test_ds = test_ds.select(range(100))

model = AutoModelForSequenceClassification.from_pretrained(
    CKPT, num_labels=num_labels, id2label=id2label, label2id=label2id)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted")}

ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
ta_kwargs = dict(
    output_dir="./_trainer_out",
    num_train_epochs=1 if SMOKE else 3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)
for k in ("eval_strategy", "evaluation_strategy"):   # arg renamed across versions
    if k in ta_params:
        ta_kwargs[k] = "epoch"
        break
args = TrainingArguments(**{k: v for k, v in ta_kwargs.items() if k in ta_params})

tr_params = set(inspect.signature(Trainer.__init__).parameters)      # tokenizer -> processing_class
tok_kw = {"processing_class": tok} if "processing_class" in tr_params else {"tokenizer": tok}
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, compute_metrics=compute_metrics, **tok_kw)

print(f">> device={args.device}  epochs={args.num_train_epochs}  train={len(train_ds)}", flush=True)
trainer.train()
test_metrics = trainer.evaluate(test_ds)
print(">> TEST metrics:", test_metrics, flush=True)

model.save_pretrained(OUTDIR)
tok.save_pretrained(OUTDIR)

import json
val_hist = [h for h in trainer.state.log_history if "eval_accuracy" in h]
metrics = {
    "model": CKPT,
    "dataset": "dair-ai/emotion (split)",
    "labels": labels,
    "hyperparameters": {"epochs": args.num_train_epochs, "batch_size": 16,
                        "learning_rate": 2e-5, "max_length": 128, "seed": 42, "device": str(args.device)},
    "validation_per_epoch": [
        {"epoch": h.get("epoch"), "accuracy": h.get("eval_accuracy"),
         "f1_weighted": h.get("eval_f1"), "loss": h.get("eval_loss")} for h in val_hist],
    "test": {"accuracy": test_metrics.get("eval_accuracy"),
             "f1_weighted": test_metrics.get("eval_f1"), "loss": test_metrics.get("eval_loss")},
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(">> saved model to", os.path.abspath(OUTDIR), "| metrics -> metrics.json", flush=True)
