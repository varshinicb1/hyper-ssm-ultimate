# Contributing

Thanks for your interest in ICM! Here's how to contribute:

## Development Setup

```bash
git clone https://github.com/varshinicb1/hyper-ssm-ultimate
cd hyper-ssm-ultimate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

All 81 tests must pass before submitting a PR.

## Code Style

- Python 3.10+ with type annotations
- Follow existing patterns in the codebase
- No unnecessary comments
- No emojis in code (unless asked)

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description

## Adding a New Memory Backend

1. Create a class similar to `HyperbolicMemoryTree` or `InfiniteContextMemory`
2. Implement `remember(embedding, content)` and `recall(embedding, top_k)`
3. Add tests in `tests/test_icm.py`
4. Integrate into `IcmLlm` in `hyper_ssm/llm_integration.py`
5. Add `--memory-backend` support in CLI and server
