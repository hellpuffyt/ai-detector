"""Inference pipeline test with a TINY STAND-IN model (not the real detector).

The real detector (bert-base-cased + the published LoRA adapter, ~446 MB) is NOT
downloaded here. Instead a ~1 MB randomly initialised BERT from Hugging Face's
test fixtures gets a fresh LoRA adapter, so this only checks that the code path
(tokenise -> PEFT model -> softmax -> probability) is wired correctly. It says
nothing about detection quality. Needs network access to the Hugging Face Hub.
"""

import math

import pytest

from conftest import load_module

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

rd = load_module("run_detector.py", "run_detector_pipeline")
agg = load_module("raid/src/aggregators.py", "raid_aggregators_pipeline")

TINY = "hf-internal-testing/tiny-random-BertForSequenceClassification"


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TINY)
    base = AutoModelForSequenceClassification.from_pretrained(TINY, num_labels=2)
    config = LoraConfig(r=4, target_modules=["query", "value"], modules_to_save=["classifier"])
    model = get_peft_model(base, config)
    model.eval()
    return model, tokenizer


def test_classify_text_returns_probability(tiny_model):
    model, tokenizer = tiny_model
    p = rd.classify_text("A short paragraph to classify.", model, tokenizer)
    assert 0.0 <= p <= 1.0
    assert rd.classify_text("A short paragraph to classify.", model, tokenizer) == p


def test_long_text_is_truncated_not_rejected(tiny_model):
    model, tokenizer = tiny_model
    p = rd.classify_text("word " * 5000, model, tokenizer)
    assert 0.0 <= p <= 1.0 and not math.isnan(p)
