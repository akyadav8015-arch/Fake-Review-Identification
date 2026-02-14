import torch
import numpy as np

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "models/distilbert_model"

# Load tokenizer and model once (important for performance)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

def predict_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    prediction = np.argmax(probabilities)
    confidence = float(np.max(probabilities))

    if prediction == 1:
        label = "Genuine"
    else:
        label = "Fake"

    return label, confidence
