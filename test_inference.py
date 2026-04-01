from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

base_model_name = "google-bert/bert-base-cased"
lora_model_name = "gouwsxander/slop-detector-bert"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, lora_model_name)

model.eval()

inputs = ["This is a sample text to classify.", "Another example sentence."]

# Tokenize
tokenized = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=512, padding=True)

with torch.no_grad():
    outputs = model(**tokenized)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(probabilities[:, 1])  # Probability of AI-generated