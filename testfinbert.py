from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

text = "Market is bullish"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
probs = F.softmax(outputs.logits, dim=1)
labels = ["negative", "neutral", "positive"]
for i, p in enumerate(probs[0]):
    print(f"{labels[i]}: {p.item():.4f}")
