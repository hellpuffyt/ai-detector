# Security Policy

## Reporting a vulnerability
Please report vulnerabilities privately through GitHub:
https://github.com/hellpuffyt/ai-detector/security/advisories/new

Do not open a public issue for security problems. Include the affected file/script, steps to reproduce and the impact.

## Scope notes
- `run_detector.py` downloads `google-bert/bert-base-cased` and the `gouwsxander/slop-detector-bert` adapter from the
  Hugging Face Hub. Only load models you trust; pin a revision if you need reproducibility.
- The data-generation scripts read an OpenAI API key from `OPENAI_API_KEY` or a git-ignored `.env` file. Never commit keys.
- This is research code without a supported release line; fixes land on `main`.
