# Changelog

All notable changes to this project are documented here (format: Keep a Changelog).

## [Unreleased]
### Fixed
- `run_detector.py`: piped stdin without a trailing empty line crashed with `EOFError`; stdin is now read as one text.
- `run_detector.py`: interactive input was classified line by line; the lines are now joined into one text, as the prompt implies.
- `run_detector.py`: a missing input file loaded (and possibly downloaded) the model first and then exited 0; input is now validated first and errors exit non-zero. Empty input is rejected.
- `raid/src/aggregators.py`: `all_or_nothing` returned NaN for long texts (float underflow of both products); now computed in log space.
- `data-generation/src/utils.py`: splitting a batch file whose line count is not divisible by the split count produced an extra file; more splits than lines divided by zero; re-running appended duplicate requests (duplicate `custom_id`s) to existing batch files.
- `data-generation/src/utils.py`: the API key can now come from `OPENAI_API_KEY` in the environment; a missing key gives a clear error instead of a `FileNotFoundError`/`KeyError`.
### Added
- Test suite (`tests/`), `requirements.txt` / `requirements-dev.txt`, GitHub Actions CI, `SECURITY.md`, `CONTRIBUTING.md`, `.editorconfig`, `data-generation/.env.example`, `docs/PRODUCTION_READINESS.md`.
### Changed
- README: provenance note, RAID numbers recomputed from the committed predictions (the previous ranges were wrong for news and Reddit), repository-size explanation, testing instructions.
- Stage lock files: upgraded transitive packages with known advisories (urllib3, requests, certifi, idna, setuptools, anyio, pillow, nltk, pyarrow, filelock, fsspec).
