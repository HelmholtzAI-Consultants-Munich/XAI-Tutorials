# DistilBERT fine-tuned on the Emotion dataset

Checkpoint used by the XAI transformer tutorials.

## Setup
- **Base model:** `distilbert-base-uncased`
- **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (`split` config) — 16,000 train / 2,000 validation / 2,000 test
- **Labels (6):** sadness, joy, love, anger, fear, surprise
- **Hyperparameters:** 3 epochs · batch size 16 · lr 2e-5 · weight decay 0.01 · max_length 128 · seed 42
- **Hardware:** Apple Silicon GPU (MPS), fp32 · training time ≈ 21 min

## Results

| Split | Accuracy | F1 (weighted) | Loss |
|---|---|---|---|
| Validation — epoch 1 | 0.9265 | 0.9265 | 0.1906 |
| Validation — epoch 2 | 0.9355 | 0.9349 | 0.1678 |
| Validation — epoch 3 | 0.9390 | 0.9391 | 0.1552 |
| **Test (final)** | **0.9295** | **0.9290** | 0.1871 |

Final training loss: 0.239.

## Artifact
- `model.safetensors` ≈ 268 MB (255 MiB), fp32
- Load with `AutoModelForSequenceClassification.from_pretrained("<path>")`

## Reproduce
```bash
pip install -r requirements.txt
python train_distilbert-emotion.py          # full run
SMOKE=1 python train_distilbert-emotion.py  # tiny sanity run
```
