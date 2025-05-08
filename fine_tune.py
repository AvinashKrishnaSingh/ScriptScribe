#!/usr/bin/env python3
"""
fine_tune_itihasa.py

Fine-tunes a Sanskrit→English translation model on the rahular/itihasa corpus.
"""

import os
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# 1. Configuration
MODEL_NAME    = "facebook/m2m100_418M"      # or "ai4bharat/indictrans2-indic-en-1B"
OUTPUT_DIR    = "./itihasa-finetuned"
MAX_LENGTH    = 128
BATCH_SIZE    = 8
NUM_EPOCHS    = 4
LR            = 3e-5
FP16          = torch.cuda.is_available()

# 2. Load dataset
print("Loading dataset…")
ds = load_dataset("rahular/itihasa")
print(ds)

# 3. Preprocess: flatten translation dict → src/tgt
def preprocess_examples(ex):
    return {
        "src": ex["translation"]["sn"],
        "tgt": ex["translation"]["en"]
    }

ds = ds.map(preprocess_examples, remove_columns=["translation"])

# 4. Load tokenizer & tokenize
print(f"Loading tokenizer for {MODEL_NAME}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    src_tok = tokenizer(
        batch["src"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tgt_tok = tokenizer(
        batch["tgt"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    return {
        "input_ids":      src_tok["input_ids"],
        "attention_mask": src_tok["attention_mask"],
        "labels":         tgt_tok["input_ids"],
    }

print("Tokenizing dataset…")
tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["src","tgt"])

# 5. Load model
print(f"Loading model {MODEL_NAME}…")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model = model.to("cuda")

# 6. Prepare metrics
metric = load_metric("sacrebleu")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # SacreBLEU expects list of references per example
    refs = [[lbl] for lbl in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=refs)
    return {"bleu": result["score"]}

# 7. TrainingArguments and Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    fp16=FP16,
    logging_steps=100,
    load_best_model_at_end=True,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Train!
if __name__ == "__main__":
    print("Starting training…")
    trainer.train()
    print("Evaluating on test set…")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"Test metrics: {test_metrics}")
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
