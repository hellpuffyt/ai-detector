#!/usr/bin/env python3
"""
AI Slop Detector - Usage Script

This script demonstrates how to use the AI detector model to classify text as human-written or AI-generated.

Usage:
    python run_detector.py                # type text, finish with an empty line (or Ctrl-D)
    python run_detector.py your_text.txt  # classify the whole file as one text
    cat text.txt | python run_detector.py # classify piped stdin as one text
"""

import sys
from typing import TextIO

BASE_MODEL_NAME = "google-bert/bert-base-cased"
LORA_MODEL_NAME = "gouwsxander/slop-detector-bert"
MAX_LENGTH = 512
THRESHOLD = 0.5

DEFAULT_EXAMPLES = [
    "This is a sample human-written text.",
    "The quick brown fox jumps over the lazy dog.",
    "In a groundbreaking discovery, scientists have found evidence of water on Mars. This finding could have significant implications for future space exploration and the search for extraterrestrial life.",
]


def load_detector(base_model_name: str = BASE_MODEL_NAME, lora_model_name: str = LORA_MODEL_NAME):
    """Load the AI detector model and tokenizer."""
    # Heavy imports are deferred so input errors are reported before any model download.
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    import torch

    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    ai_probability = probabilities[0, 1].item()
    return ai_probability


def label_for(probability: float) -> str:
    """Texts with probability > THRESHOLD are estimated to be AI-generated."""
    return "AI-generated" if probability > THRESHOLD else "Human-written"


def read_texts(argv: list[str], stdin: TextIO) -> list[str]:
    """Return the texts to classify, or raise SystemExit with a message on bad input.

    A file or piped stdin is treated as ONE text. Interactive input is read until an
    empty line (or EOF) and the lines are joined into ONE text. If nothing is entered
    interactively, the built-in examples are used.
    """
    if len(argv) > 1:
        filename = argv[1]
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            raise SystemExit(f"File {filename} not found.")
        except (OSError, UnicodeDecodeError) as e:
            raise SystemExit(f"Could not read {filename}: {e}")
        if not content.strip():
            raise SystemExit(f"File {filename} is empty.")
        return [content]

    if not stdin.isatty():
        content = stdin.read()
        if not content.strip():
            raise SystemExit("No text received on stdin.")
        return [content]

    print("Enter text to classify (press Enter twice to finish):")
    lines: list[str] = []
    while True:
        line = stdin.readline()
        if line.strip() == "":  # empty line or EOF ("")
            break
        lines.append(line.rstrip("\n"))

    if not lines:
        print("No text entered. Using default examples.")
        return list(DEFAULT_EXAMPLES)
    return ["\n".join(lines)]


def main(argv: list[str] | None = None, stdin: TextIO | None = None) -> int:
    argv = sys.argv if argv is None else argv
    stdin = sys.stdin if stdin is None else stdin

    # Validate input before loading (and possibly downloading) the model.
    texts = read_texts(argv, stdin)

    model, tokenizer = load_detector()

    print("\nClassifying texts:")
    print("-" * 50)

    for text in texts:
        prob = classify_text(text, model, tokenizer)
        print(f"Text: {text[:50]}...")
        print(f"Probability: {prob:.4f} ({label_for(prob)})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
