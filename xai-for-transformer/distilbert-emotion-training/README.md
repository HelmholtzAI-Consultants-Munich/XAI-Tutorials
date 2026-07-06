# DistilBERT emotion — training

This folder documents how the **DistilBERT emotion classifier** used in the transformer
tutorials was produced. The trained weights themselves are **not** stored here (~268 MB) —
they are published as a GitHub **Release asset** (`distilbert_emotion_weights.zip`) and
downloaded by the tutorial notebook at runtime.

## What's here

| File | Description |
|---|---|
| `train_distilbert-emotion.py` | The fine-tuning script (`distilbert-base-uncased` → `dair-ai/emotion`). |
| `requirements.txt` | Exact package versions used for training (Python 3.11.7). |
| `metrics.json` | Machine-readable config + validation/test metrics. |
| `TRAINING_SUMMARY.md` | Human-readable summary of the setup and results. |
| `train.log` | Training console output (tqdm progress bars stripped for readability). |

## The model

- **Base:** `distilbert-base-uncased`
- **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (`split`) — 6 classes: sadness, joy, love, anger, fear, surprise
- **Result:** test accuracy **92.95%**, weighted F1 **0.929**

See [`TRAINING_SUMMARY.md`](./TRAINING_SUMMARY.md) for the full breakdown.

## Reproduce

```bash
pip install -r requirements.txt
python train_distilbert-emotion.py          # full run (~21 min on Apple Silicon / MPS)
SMOKE=1 python train_distilbert-emotion.py  # tiny end-to-end sanity run
```

The script writes the fine-tuned checkpoint to `./distilbert-emotion/` (a standard
`save_pretrained` folder: `config.json` + `model.safetensors` + tokenizer) and the
metrics to `metrics.json`.

## Using the weights

After downloading and unzipping the Release asset, load it like any Hugging Face model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("<unzipped-folder>")
tokenizer = AutoTokenizer.from_pretrained("<unzipped-folder>")
```
