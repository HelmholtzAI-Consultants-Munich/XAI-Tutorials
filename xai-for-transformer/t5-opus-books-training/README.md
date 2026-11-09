# T5-small translation — training

This folder contains the fine-tuning script for **T5-small English-to-German
translation**, complementing the optional translation section of the
Transformer tutorial.

Previously published weights are available as a GitHub Release asset:
[`t5_small_weights.zip`](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/releases/download/v1.0.0/t5_small_weights.zip)
(~671 MB). The weights are not stored in this folder.

## What's here

| File | Description |
|---|---|
| `train_t5-small.py` | Fine-tuning script for `google-t5/t5-small` on the English–German `opus_books` dataset. |
| `README.md` | Model overview, script configuration, and instructions for using the published weights. |

## The model

- **Base:** [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small)
- **Dataset:** [`opus_books`](https://huggingface.co/datasets/opus_books), configuration `de-en`
- **Translation direction:** English → German
- **Task prefix:** `"Translate English to German: "`
- **Evaluation metric:** SacreBLEU

## Training configuration

The script defines the following settings:

| Setting | Value |
|---|---|
| Training/evaluation split | 80% / 20% |
| Dataset usage | Full training and evaluation splits |
| Maximum tokenized length | 128 tokens for inputs and targets |
| Learning rate | `2e-3` |
| Training batch size per device | 128 |
| Evaluation batch size per device | 128 |
| Training epochs | 40 |
| Weight decay | `0.01` |
| Evaluation strategy | Every epoch |
| Generate translations during evaluation | Yes |
| FP16 | Enabled |
| Checkpoint directory | `./t5-english-to-german/` |
| Maximum retained checkpoints | 3 |

The dataset split does not specify a fixed seed. Exact reproduction of
the original split is therefore not guaranteed.

These settings describe the script currently provided. The original
training environment, evaluation results, and exact provenance of the
published weights have not been documented here.

## Running the script

The script performs a full fine-tuning run and can require substantial
time and GPU memory. It is not necessary to run it to complete the tutorial.

The configuration retains the original batch size and FP16 setting.
Running it requires hardware that supports these settings and has
sufficient memory.

Run from this folder:

```bash
python -m pip install -r ../../requirements_xai-for-transformer.txt
python train_t5-small.py
```

During training, checkpoints are saved under `./t5-english-to-german/`.
After training, the final model and tokenizer are saved together in
`./t5-small-english-german/`.

All output paths are relative to the directory from which the script is run.

## Using the published weights

Download and extract
[`t5_small_weights.zip`](https://github.com/HelmholtzAI-Consultants-Munich/XAI-Tutorials/releases/download/v1.0.0/t5_small_weights.zip).

Locate the checkpoint folder containing the model configuration and weights,
then load it with:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("<checkpoint-folder>")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
```

The tokenizer above matches the base checkpoint used by the training script.

The published weights can be used without repeating fine-tuning.