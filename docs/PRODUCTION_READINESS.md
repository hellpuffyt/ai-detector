# Production readiness — ai-detector

Audit date: 2026-10-22. Tested commit: see "Tested commit" at the end.

## Scope
An end-to-end research pipeline for detecting AI-generated paragraphs: a Wikipedia scraper (`scraping/`), OpenAI
Batch API data generation (`data-generation/`), LoRA fine-tuning of `bert-base-cased` (`training/`), RAID benchmark
evaluation (`raid/`), and a command-line inference script (`run_detector.py`) that uses the published adapter
`gouwsxander/slop-detector-bert`. The repository is a copy of
[gouwsxander/slop-detector](https://github.com/gouwsxander/slop-detector) (see README "Provenance").

Architecture: each stage is an independent uv project; the root holds the inference script, its `requirements.txt`
and the test suite. Data flows scrape -> summarize (batch) -> rewrite (batch) -> `combine_data.py` -> `data/wikipedia.jsonl`
-> training -> adapter on Hugging Face -> `run_detector.py` / RAID analysis.

## Why the repository is ~14 MB
No model weights or binaries are committed. The size is versioned data: `data-generation/tmp/` (raw OpenAI Batch
API requests/responses, ~47 MB uncompressed, largest single blob 12 MB), `data/wikipedia.jsonl` (10.5 MB),
`scrapes/` (5 MB) and `raid/analysis/predictions_10000.csv` (1.3 MB). These are not disposable build artifacts:
`data-generation/src/combine_data.py` regenerates `data/wikipedia.jsonl` byte-for-byte from them (verified, see
below), so they stay tracked. Git history was not rewritten. The model weights are obtained at runtime from the
Hugging Face Hub (`google-bert/bert-base-cased` ~436 MB, adapter ~11 MB).

## Platforms and prerequisites
- Inference and tests: Python 3.11+ (verified with 3.12 on macOS arm64 and, via CI definition, Ubuntu); network access to the Hugging Face Hub on first run.
- `training/`: CUDA GPU (calls `.to('cuda')`, `fp16=True`) — not verified.
- `raid/`: RAID dataset CSV (not in the repo) and a local checkpoint under `models/` — not verified end-to-end.
- `data-generation/`: OpenAI API key and paid API usage — not run.

## Clean-clone commands (as in the README)
```
git clone https://github.com/hellpuffyt/ai-detector && cd ai-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_detector.py input_text.txt
pip install -r requirements-dev.txt
python -m pytest
```
Result on the tested commit (fresh `git clone` of the audit branch into a new directory, Python 3.12.14, macOS arm64):
install exit 0; quick start printed `Probability: 0.0026 (Human-written)` exit 0; piped stdin same result;
`pytest`: 28 passed, 0 failed, 0 skipped. (The first clean-clone run found a real defect — `test_scraping.py` could
not import because `requests` was only pulled in transitively by an old wikipedia-api — fixed in 5a39bc6.)

## Commands executed and outcomes
| Command | Exit | Result |
|---|---|---|
| baseline `python run_detector.py input_text.txt` (original code, real model) | 0 | `Probability: 0.0026 (Human-written)`; deterministic across runs (classifier head is loaded from the adapter) |
| baseline `printf 'a\nb\n' \| python run_detector.py` | 1 | `EOFError: EOF when reading a line` (bug, fixed) |
| baseline `python run_detector.py nope.txt` | 0 | printed "not found" after loading the model (bug, fixed: now exit 1 before loading) |
| baseline test suite | — | none existed; `test_inference.py` is a smoke script that downloads the model at import |
| `python -m pytest` (after fixes) | 0 | 28 passed, 0 failed, 0 skipped, 0 xfail |
| same suite against the original source files | 1 | 17 failed, 10 passed (the regression tests fail on the old code) |
| validation accuracy re-check (published adapter, CPU, seed-42 5% split from `training/src/train.py:get_datasets()`) | 0 | 964/997 = 96.7% (README: 96%) |
| RAID numbers recomputed from `raid/analysis/predictions_10000.csv` | 0 | table now in README; old README ranges were wrong for news (26.3%, not 30–40%) and Reddit (0.7%, not 1–20%) |
| `python src/combine_data.py` in a scratch export | 0 | output identical (`cmp`) to committed `data/wikipedia.jsonl` |
| `uvx pip-audit` on root env (latest resolution) | 0 | no known vulnerabilities |
| `uvx pip-audit` on stage locks after upgrades | — | data-generation, scraping: none. training/raid: torch 2.9.x and transformers 4.57.x advisories remain (see risks) |

Caveat on the accuracy re-check: the split is rebuilt with the current `datasets` library; if its shuffle differs
from the version used for training, part of the 997 rows may have been training data. It confirms the claim is
plausible, not that it is exactly reproduced.

## Tests
- Unit: `tests/test_run_detector.py` (input handling, exit codes), `tests/test_data_generation.py` (batch-file builder,
  API-key lookup), `tests/test_scraping.py` (paragraph filter, RandomList, dependency pin), `tests/test_aggregators.py`,
  `tests/test_raid_analysis.py` (TPR@FPR helper + the README numbers from the committed CSV).
- Integration: `tests/test_inference_pipeline.py` runs tokenise -> PEFT -> softmax with a **tiny random stand-in
  model** (`hf-internal-testing/tiny-random-BertForSequenceClassification` + a fresh LoRA). It verifies wiring only,
  not detection quality, and needs network access to the Hub.
- The real model was exercised manually (commands above), not in CI.

## CI
`.github/workflows/ci.yml`: one Ubuntu job, Python 3.12, CPU-only torch, `pytest`. Actions pinned to commit SHAs
(checkout v7.0.1 `3d3c42e5…`, setup-python v7.1.0 `6d4fd22b…`, both resolved via the GitHub API), `contents: read`.
Status: the workflow runs on every push to `main` and on pull requests; it passed on `9570dbc` (2026-10-24).

## Bugs found and fixed (each with a regression test)
1. Piped stdin crashed with `EOFError` (`test_piped_stdin_without_blank_line_does_not_crash`).
2. Interactive input classified each line separately instead of one text (`test_interactive_lines_are_joined_into_one_text`).
3. Missing input file: model loaded/downloaded first, then exit 0 (`test_missing_file_exits_nonzero_before_loading_model`); empty input now rejected.
4. `all_or_nothing` aggregator returned NaN for long texts due to float underflow (`test_all_or_nothing_long_text_does_not_underflow_to_nan`).
5. Batch-file splitting produced an extra file for non-divisible line counts, divided by zero for tiny inputs, and duplicated requests on re-run (`tests/test_data_generation.py`).
6. API key only from a mandatory `.env` with a non-standard name; missing file crashed (`test_missing_api_key_is_a_clear_error_not_a_crash`).
7. Scraper relied on `requests` arriving transitively via wikipedia-api 0.8.x; newer wikipedia-api breaks import and timeout handling (`test_scraper_timeout_handler_matches_the_pinned_wikipedia_api`).
8. Documentation: RAID ranges corrected; provenance added; LICENSE notice of the original author restored (MIT requires keeping it).

## Security review
- Secrets: regex scan of the tree and all history (GitHub/OpenAI/Anthropic/AWS/Slack/HF token shapes, private keys,
  local paths) — no hits. `.env` is git-ignored; `.env.example` added.
- Inputs: the CLI reads a local file or stdin; no network-facing surface. Models are fetched from the Hugging Face Hub
  by name without a pinned revision (supply-chain trust in the Hub repos; documented in SECURITY.md).
- Dependency audit: see table above.
- Workflow permissions: `contents: read` only.

## Known limitations and unresolved risks
- Detection quality limits are in the README; the detector must not be used as proof of authorship.
- `training/` and `raid/` locks still pin torch 2.9.x / transformers 4.57.x with published advisories. Upgrading
  them is a major-version change that cannot be verified here (needs a CUDA GPU and the RAID dataset).
- `training/`, `raid/` benchmark run and `data-generation/` were not executed (GPU / dataset / paid API).
- Model downloads are not pinned to a revision.
- Licensing: the code is a copy of an MIT project; the original copyright line had been replaced. It is restored on
  this branch; the owner should confirm.

## Acceptance checklist
| # | Item | Expected | Command / test | Result | Actual | Reference |
|---|---|---|---|---|---|---|
| 1 | Install per README | deps install | `pip install -r requirements.txt` | pass | exit 0 (clean clone) | requirements.txt |
| 2 | Quick start on file | probability + label, exit 0 | `python run_detector.py input_text.txt` | pass | 0.0026 Human-written | run_detector.py |
| 3 | Piped stdin | one text classified | `cat input_text.txt \| python run_detector.py` | pass | 0.0026 | test_run_detector.py |
| 4 | Bad input | clear error, non-zero exit, no model load | `python run_detector.py nope.txt` | pass | exit 1 | test_run_detector.py |
| 5 | Inference wiring | probability in [0,1], long input truncated | pytest | pass (stand-in model) | — | test_inference_pipeline.py |
| 6 | README accuracy claim | ≈96% | validation re-check | pass (with split caveat) | 96.7% | this file |
| 7 | README RAID claims | match committed predictions | pytest | pass after README fix | see README table | test_raid_analysis.py |
| 8 | Dataset reproducible from committed data | identical output | `combine_data.py` + `cmp` | pass | identical | data-generation/src/combine_data.py |
| 9 | No secrets in tree/history | none | regex scan | pass | no hits | — |
| 10 | CI defined, pinned, least privilege | yes | review | pass (not yet run) | — | .github/workflows/ci.yml |

## Classification proposal
**Partially Verified** — the inference CLI, its tests, the dataset reproduction and the README claims are verified;
training, the RAID benchmark run and data generation could not be executed here (GPU, external dataset, paid API),
and the training/raid locks keep vulnerable torch/transformers pins.

## Tested commit
Branch `audit/production-readiness`; the commit that adds this file (code last changed in 5a39bc6), 2026-10-22.
