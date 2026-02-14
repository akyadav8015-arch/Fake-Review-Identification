import pandas as pd
import numpy as np
import torch
import os

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
data = pd.read_csv("data/yelp_reviews.csv")

# Rename columns to HuggingFace format
data = data.rename(columns={"review": "text", "label": "labels"})

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"], data["labels"],
    test_size=0.2,
    random_state=42,
    stratify=data["labels"]
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_dict({
    "text": train_texts.tolist(),
    "labels": train_labels.tolist()
})

test_dataset = Dataset.from_dict({
    "text": test_texts.tolist(),
    "labels": test_labels.tolist()
})

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/distilbert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)

# Save model
os.makedirs("models/distilbert_model", exist_ok=True)
trainer.save_model("models/distilbert_model")
tokenizer.save_pretrained("models/distilbert_model")

print("DistilBERT model saved successfully!")
