from embeddingkit import cosine, normalize


def test_normalize_and_cosine():
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
    assert round(cosine([1.0, 0.0], [1.0, 0.0]), 6) == 1.0
