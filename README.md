# Grounded RAG & Knowledge-Graph API

A FastAPI service for grounded question answering, vector similarity search, and lightweight knowledge-graph extraction/querying. The project combines deterministic retrieval utilities with graph-oriented endpoints behind a single deployable API.

## Capabilities

- **Grounded answering** — answer questions from supplied context chunks and return supporting citations, confidence, and answerability metadata.
- **Vector search** — rank candidate items using cosine similarity.
- **Knowledge-graph extraction** — identify entities and relations from source text.
- **Graph querying** — query extracted relationships using structured requests.
- **Community summaries** — produce higher-level summaries over graph communities.
- **Input validation** — Pydantic models and validation for API contracts.
- **Health endpoint** — deployment-friendly service health check.

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI interface.

## Design notes

The implementation deliberately keeps several retrieval and scoring steps deterministic. This makes behaviour easier to inspect and test while still demonstrating core RAG concepts such as relevance scoring, grounded-answer selection, citations, vector similarity, entity/relation extraction, and graph traversal.

## Deployment

The application is compatible with standard Python ASGI hosting. A typical start command is:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Use `/health` as the service health-check path.

## Portfolio focus

This repository is retained as an applied RAG/knowledge-graph engineering project. The public-facing documentation focuses on the system design and API behaviour rather than the original exercise context in which the implementation was developed.
