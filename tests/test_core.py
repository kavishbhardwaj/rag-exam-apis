import math

from main import cosine, root


def test_cosine_identical_vectors():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)


def test_cosine_orthogonal_vectors():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_root_lists_core_endpoints():
    response = root()
    assert response["status"] == "ok"
    assert "/grounded-answer" in response["endpoints"]
    assert "/vector-search" in response["endpoints"]
    assert "/extract-graph" in response["endpoints"]
