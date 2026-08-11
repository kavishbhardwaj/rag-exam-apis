# Grounded RAG & Knowledge-Graph API

A FastAPI service for grounded question answering, vector similarity search, and lightweight knowledge-graph extraction/querying.

**This repository consolidates and updates related work for easier maintenance and reference.**

## Capabilities

- **Grounded answering** — answer from supplied context chunks with citations, confidence, and answerability metadata
- **Vector search** — cosine-similarity ranking
- **Knowledge-graph extraction** — entities and relations from source text
- **Graph querying** — structured relationship queries
- **Community summaries** — higher-level graph summaries
- **Input validation** — Pydantic request/response models
- **Health endpoint** — deployment-friendly service check

## API surface

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/grounded-answer` | Context-grounded Q&A with citations |
| `POST` | `/vector-search` | Similarity-based retrieval |
| `POST` | `/extract-graph` | Entity/relation graph extraction |
| `POST` | `/graph-query` | Structured graph queries |
| `POST` | `/community-summary` | Community-level graph summaries |
| `GET` | `/health` | Service health check |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

## Tests

```bash
pytest -q
```

## Design

Several retrieval and scoring steps are kept deterministic so behaviour is easy to inspect and test.

## Deployment

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Use `/health` as the service health-check path.
