#!/usr/bin/env python3
"""
AI Slop Detector - Usage Script

This script demonstrates how to use the AI detector model to classify text as human-written or AI-generated.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

def load_detector():
    """Load the AI detector model and tokenizer."""
    base_model_name = "google-bert/bert-base-cased"
    lora_model_name = "gouwsxander/slop-detector-bert"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_name)

    model.eval()
    return model, tokenizer

def classify_text(text: str, model, tokenizer) -> float:
    """Classify a single text and return probability of being AI-generated."""
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    ai_probability = probabilities[0, 1].item()
    return ai_probability

def main():
    import sys

    # Load the model
    model, tokenizer = load_detector()

    texts = []
    if len(sys.argv) > 1:
        # Read from file
        filename = sys.argv[1]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                texts = [content]  # Treat whole file as one text
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return
    else:
        # Interactive input
        print("Enter text to classify (press Enter twice to finish):")
        while True:
            line = input()
            if line == "":
                break
            texts.append(line)

    if not texts:
        print("No text entered. Using default examples.")
        texts = [
            "This is a sample human-written text.",
            "The quick brown fox jumps over the lazy dog.",
            "In a groundbreaking discovery, scientists have found evidence of water on Mars. This finding could have significant implications for future space exploration and the search for extraterrestrial life.",
        ]

    print("\nClassifying texts:")
    print("-" * 50)

    for text in texts:
        prob = classify_text(text, model, tokenizer)
        label = "AI-generated" if prob > 0.5 else "Human-written"
        print(f"Text: {text[:50]}...")
        print(f"Probability: {prob:.4f} ({label})")
        print()

if __name__ == "__main__":
    main()