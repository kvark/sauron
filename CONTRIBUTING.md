# Contributing

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

Python 3.12 is required (`uni2ts` does not yet support 3.13+). `uv` will download it automatically:

```bash
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Linting

```bash
ruff check .
ruff format --check .
```
