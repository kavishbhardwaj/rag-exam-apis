from fastapi.testclient import TestClient
from main import app, EMBEDDINGS

client = TestClient(app)


def test_grounded_answer():
    response = client.post(
        "/grounded-answer",
        json={
            "question": "What year was FAISS released?",
            "chunks": [
                {
                    "chunk_id": "C1",
                    "text": "FAISS was developed by Facebook AI Research and open-sourced in 2017.",
                },
                {
                    "chunk_id": "C2",
                    "text": "Qdrant was released in 2021.",
                },
            ],
        },
    )
    print(response.json())


def test_vector_search():
    response = client.post(
        "/vector-search",
        json={
            "query_id": "Q001",
            "query_vector": EMBEDDINGS["D001"],
            "top_k": 10,
            "rerank_top_n": 3,
            "filter": {
                "department": "legal",
                "year": {"gte": 2020},
                "region": {"in": ["europe"]},
            },
        },
    )
    print(response.json())


def test_graphrag():
    graph = client.post(
        "/extract-graph",
        json={
            "chunk_id": "C001",
            "text": "LangChain was created by Harrison Chase and integrates with OpenAI.",
        },
    ).json()
    print(graph)
    print(
        client.post(
            "/graph-query",
            json={
                "question": "Who created the framework that integrates with OpenAI?",
                "graph": graph,
            },
        ).json()
    )


if __name__ == "__main__":
    test_grounded_answer()
    test_vector_search()
    test_graphrag()
