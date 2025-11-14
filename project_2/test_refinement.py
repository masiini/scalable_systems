from graph_rag import _postprocess_cypher, validate_cypher, GraphRAG, Query
import types

class FakeConn:
    """Simulates a Kuzu connection for testing EXPLAIN."""
    def __init__(self, valid_queries: set[str]):
        self.valid_queries = valid_queries

    def execute(self, q: str):
        # Only allow EXPLAIN queries to pass if q is in valid set
        if q.startswith("EXPLAIN "):
            inner = q[len("EXPLAIN "):]
            if inner in self.valid_queries:
                return []   # no error
            raise RuntimeError("Syntax error near token")
        return []


def test_self_refinement_loop():
    rag = GraphRAG(max_attempts=3)

    # Text2cypher to return an invalid query
    bad_query = Query(query="MATCH (broken")
    rag.text2cypher = lambda **kw: types.SimpleNamespace(query=bad_query)

    # Repair to return a valid query
    fixed_query = Query(query="MATCH (s:Scholar) RETURN s")
    rag.repair = lambda **kw: types.SimpleNamespace(repaired=fixed_query)

    # Fake schema and connection
    schema = "{}"
    conn = FakeConn(valid_queries={"MATCH (s:Scholar) RETURN s"})

    # Run
    out = rag.get_cypher_query("dummy question", schema, conn)

    assert out.query == "MATCH (s:Scholar) RETURN s"
    print("Self-refinement loop test passed!")

if __name__ == "__main__":
    test_self_refinement_loop()
