# RAG Exam APIs — Questions 3, 4 and 5

This single FastAPI application provides all required endpoints:

- `POST /grounded-answer` — Question 3
- `POST /vector-search` — Question 4
- `POST /extract-graph` — Question 5
- `POST /graph-query` — Question 5
- `POST /community-summary` — Question 5
- `GET /health` — deployment health check

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the endpoints.

## Render settings

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

After deployment, submit:

- Q3: `https://YOUR-SERVICE.onrender.com/grounded-answer`
- Q4: `https://YOUR-SERVICE.onrender.com/vector-search`
- Q5: `https://YOUR-SERVICE.onrender.com`
