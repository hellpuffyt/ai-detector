# Contributing

1. Create a virtual environment and install the test dependencies:
   `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`
   (on Linux without a GPU, install CPU torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
2. Run the suite: `python -m pytest`. It must pass before a pull request is merged; CI runs the same command.
3. Every bug fix needs a regression test in `tests/`.
4. The per-stage projects (`scraping/`, `data-generation/`, `training/`, `raid/`) keep their own `uv.lock`;
   if you change a stage's dependencies, update its lock with `uv lock` inside that directory.
5. Do not commit API keys, `.env` files, model checkpoints (`models/`) or the RAID dataset.
6. Claims in the README (accuracy, benchmark numbers) must come from a command you actually ran; say how.
