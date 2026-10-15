# Contributing

Issues and pull requests are welcome.

```bash
pip install -e ".[dev]"
make lint test        # ruff + fast tests
make test-all         # includes slow model fits
```

Changes that alter model hyper-parameters, seeds, the cross-validation protocol or the statistics change the
published numbers. Please open an issue first and label such a change clearly in the pull request.
