# Contributing to Infinite Context Memory

Thanks for your interest! Here's how to contribute effectively.

## Getting Started

```bash
git clone https://github.com/varshinicb1/hyper-ssm-ultimate.git
cd hyper-ssm-ultimate
pip install -r requirements.txt
pip install fastapi uvicorn sentence-transformers pytest pytest-asyncio
python -m pytest tests/ -v
```

## Development Workflow

1. **Fork** the repo
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Write tests** for your changes
4. **Run the full suite**: `python -m pytest tests/ -v`
5. **Push** and open a Pull Request

## Code Style

- Follow existing patterns in the codebase
- No comments unless absolutely necessary
- Type hints on all public functions
- 88 character line limit (black-compatible)

## Testing

All features must have tests. We use pytest with markers:

```bash
pytest tests/ -v                    # all tests
pytest tests/ -m unit              # unit tests only
pytest tests/ -m integration       # integration tests only
```

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] No new warnings
- [ ] Tests added for new features
- [ ] Type hints added
- [ ] README updated if API changed

## Questions?

Open a [Discussion](https://github.com/varshinicb1/hyper-ssm-ultimate/discussions) or [Issue](https://github.com/varshinicb1/hyper-ssm-ultimate/issues/new).
