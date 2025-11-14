from graph_rag import _postprocess_cypher

def test_lowercase_comparisons():
    inp = "MATCH (s) WHERE s.gender = 'FEMALE' RETURN s"
    out = _postprocess_cypher(inp)
    assert "toLower(s.gender) = 'female'" in out

def test_contains_lowercase():
    inp = "MATCH (s)-[:AFFILIATED_WITH]->(i) WHERE i.name CONTAINS 'Cambridge'"
    out = _postprocess_cypher(inp)
    assert "toLower(i.name) CONTAINS 'cambridge'" in out

def test_strip_semicolon_and_whitespace():
    inp = " MATCH   (s)   RETURN  s ; "
    out = _postprocess_cypher(inp)
    assert out == "MATCH (s) RETURN s"

if __name__ == "__main__":
    test_lowercase_comparisons()
    test_contains_lowercase()
    test_strip_semicolon_and_whitespace()
    print("All tests passed!")