# Changelog

All notable changes to this project are documented here (format: Keep a Changelog).

## [Unreleased]
### Security
- `training/uv.lock` and `raid/uv.lock` pinned torch 2.9, transformers 4.57, aiohttp 3.13.2, datasets 4.4,
  accelerate, fsspec, fonttools and multidict versions with published advisories; relocked to fixed releases
  (torch 2.14.1, transformers 5.20.2). The default Linux torch wheels are now CUDA 13 builds, which need a recent
  NVIDIA driver. `requirements.txt` floors raised to `transformers>=5.10`, `torch>=2.13`.
### Fixed
- `run_detector.py`: piped stdin without a trailing empty line crashed with `EOFError`; stdin is now read as one text.
- `run_detector.py`: interactive input was classified line by line; the lines are now joined into one text, as the prompt implies.
- `run_detector.py`: a missing input file loaded (and possibly downloaded) the model first and then exited 0; input is now validated first and errors exit non-zero. Empty input is rejected.
- `raid/src/aggregators.py`: `all_or_nothing` returned NaN for long texts (float underflow of both products); now computed in log space.
- `data-generation/src/utils.py`: splitting a batch file whose line count is not divisible by the split count produced an extra file; more splits than lines divided by zero; re-running appended duplicate requests (duplicate `custom_id`s) to existing batch files.
- `data-generation/src/utils.py`: the API key can now come from `OPENAI_API_KEY` in the environment; a missing key gives a clear error instead of a `FileNotFoundError`/`KeyError`.
- `scraping/`: the scraper imports `requests` and catches `requests.exceptions.Timeout`, but neither `requests` nor an upper bound on `wikipedia-api` was declared; wikipedia-api >= 0.9 switched to httpx, so a fresh resolve broke the import. Now `requests` is declared and `wikipedia-api` is pinned to `<0.9`.
### Added
- Test suite (`tests/`), `requirements.txt` / `requirements-dev.txt`, GitHub Actions CI, `SECURITY.md`, `CONTRIBUTING.md`, `.editorconfig`, `data-generation/.env.example`, `docs/PRODUCTION_READINESS.md`.
### Changed
- README: provenance note, RAID numbers recomputed from the committed predictions (the previous ranges were wrong for news and Reddit), repository-size explanation, testing instructions.
- Stage lock files: upgraded transitive packages with known advisories (urllib3, requests, certifi, idna, setuptools, anyio, pillow, nltk, pyarrow, filelock, fsspec).
