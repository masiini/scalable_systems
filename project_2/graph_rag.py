from __future__ import annotations

import os
import hashlib
import re
from dataclasses import dataclass
from typing import Any, List

import dspy # type: ignore
import kuzu  # type: ignore
import marimo
import numpy as np
from dotenv import load_dotenv
from dspy.adapters.json_adapter import JSONAdapter # type: ignore
from pydantic import BaseModel, Field

load_dotenv()

__generated_with = "0.14.17"
app = marimo.App(width="medium")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# Optional SBERT
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

    _SBERT_AVAILABLE = True
except Exception:
    _SBERT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Graph schema models
# ---------------------------------------------------------------------------


class Query(BaseModel):
    query: str = Field(description="Valid Cypher query with no newlines")


class Property(BaseModel):
    name: str
    type: str = Field(description="Data type of the property")


class Node(BaseModel):
    label: str
    properties: list[Property] | None = None


class Edge(BaseModel):
    label: str = Field(description="Relationship label")
    from_: Node = Field(alias="from", description="Source node label")
    to: Node = Field(alias="to", description="Target node label")
    properties: list[Property] | None = None


class GraphSchema(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


# ---------------------------------------------------------------------------
# Exemplars and embedding utils
# ---------------------------------------------------------------------------


@dataclass
class Exemplar:
    question: str
    cypher: str


EXEMPLARS: List[Exemplar] = [
    Exemplar(
        question="Which scholars won Nobel Prizes in Physics?",
        cypher=(
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'physics' "
            "RETURN s.knownName AS scholar, p.awardYear AS year "
            "ORDER BY year"
        ),
    ),
    Exemplar(
        question="Which scholars were affiliated with the University of Cambridge?",
        cypher=(
            "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution) "
            "WHERE toLower(i.name) CONTAINS 'cambridge' "
            "RETURN s.knownName AS scholar, i.name AS institution"
        ),
    ),
    Exemplar(
        question="List female Chemistry laureates.",
        cypher=(
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "WHERE toLower(p.category) = 'chemistry' "
            "AND toLower(s.gender) = 'female' "
            "RETURN s.knownName AS laureate, p.awardYear AS year"
        ),
    ),
    Exemplar(
        question="How many laureates won prizes in each category?",
        cypher=(
            "MATCH (s:Scholar)-[:WON]->(p:Prize) "
            "RETURN toLower(p.category) AS category, "
            "COUNT(DISTINCT s) AS total "
            "ORDER BY total DESC"
        ),
    ),
]


print(f"SBERT: {_SBERT_AVAILABLE}")
if _SBERT_AVAILABLE:
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    # SBERT-based embedding
    def _embed(texts: list[str]) -> np.ndarray:
        return np.array(_sbert.encode(texts, normalize_embeddings=True))

else:
    # Simple hash-based embedding fallback
    def _hash_vec(t: str, dim: int = 256) -> np.ndarray:
        h = hashlib.sha1(t.encode()).digest()
        v = (h * ((dim // len(h)) + 1))[:dim]
        arr = np.frombuffer(v, dtype=np.uint8).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        return arr / (np.linalg.norm(arr) + 1e-9)

    def _embed(texts: list[str]) -> np.ndarray:
        return np.vstack([_hash_vec(t) for t in texts])


_EX_Q_EMB = _embed([e.question for e in EXEMPLARS])


def select_exemplars(question: str, k: int = 2) -> list[Exemplar]:
    qv = _embed([question])[0]
    sims = _EX_Q_EMB @ qv # cosine similarity
    idx = np.argsort(-sims)[:k]
    return [EXEMPLARS[i] for i in idx]


def render_exemplars_block(exs: list[Exemplar]) -> str:
    out: list[str] = []
    for e in exs:
        cy = re.sub(r"\s+", " ", e.cypher).strip().rstrip(";")
        out.append(f"Q: {e.question}\nA (Cypher): {cy}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Post-process and validate Cypher
# ---------------------------------------------------------------------------


def _postprocess_cypher(cypher: str) -> str:
    """Normalize whitespace and enforce lowercase string comparisons."""
    q = re.sub(r"\s+", " ", cypher).strip().rstrip(";")

    def _norm(m: re.Match) -> str:
        left, op, quote, val = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"toLower({left}) {op} {quote}{val.lower()}{quote}"

    q = re.sub(
        r"\b([a-zA-Z0-9_.]+)\s*(=|CONTAINS)\s*(['\"])([^'\"]+)\3",
        _norm,
        q,
        flags=re.IGNORECASE,
    )
    return q


def validate_cypher(conn: kuzu.Connection, cypher: str) -> tuple[bool, str]:
    try:
        conn.execute(f"EXPLAIN {cypher}")
        return True, ""
    except RuntimeError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class RepairText2Cypher(dspy.Signature):
    """
    Repair an invalid Cypher query using schema + EXPLAIN error.
    Keep intent; output ONE LINE, no trailing semicolon.
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    previous_query: str = dspy.InputField()
    error_message: str = dspy.InputField()
    repaired: Query = dspy.OutputField()


class PruneSchema(dspy.Signature):
    """
    Understand the graph schema and return ONLY the relevant subset.
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    pruned_schema: GraphSchema = dspy.OutputField()


class Text2Cypher(dspy.Signature):
    """
    Translate the question into a valid Cypher query that respects the graph schema.

    <SYNTAX>
    - When matching on Scholar names, ALWAYS match on `knownName`.
    - For countries, cities, continents and institutions, match on `name`.
    - Use short alphanumeric variable names (`a1`, `r1`, ...).
    - ALWAYS lowercase strings in comparisons and use CONTAINS in WHERE.
    - DO NOT use APOC.
    </SYNTAX>
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    exemplars: str = dspy.InputField(description="Relevant Q→Cypher examples")
    query: Query = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """
    Use the provided question, Cypher query and context to answer the question.
    If context is empty, say you don't have enough information.
    """

    question: str = dspy.InputField()
    cypher_query: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# DSPy / LM configuration
# ---------------------------------------------------------------------------

lm = dspy.LM(
    model="gemini/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
)
dspy.configure(lm=lm, adapter=JSONAdapter())

# ---------------------------------------------------------------------------
# Kuzu
# ---------------------------------------------------------------------------


class KuzuDatabaseManager:
    """Manages Kuzu database connection and schema retrieval."""

    def __init__(self, db_path: str = "nobel.kuzu"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)

    @property
    def get_schema_dict(self) -> dict[str, list[dict]]:
        # Nodes
        response = self.conn.execute(
            "CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;"
        )
        nodes = [row[1] for row in response]  # type: ignore

        # Relationship tables
        response = self.conn.execute(
            "CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;"
        )
        rel_tables = [row[1] for row in response]  # type: ignore

        relationships = []
        for tbl_name in rel_tables:
            response = self.conn.execute(
                f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;"
            )
            for row in response:
                relationships.append(
                    {"name": tbl_name, "from": row[0], "to": row[1]}  # type: ignore
                )

        schema: dict[str, list[dict]] = {"nodes": [], "edges": []}

        # Node properties
        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(
                f"CALL TABLE_INFO('{node}') RETURN *;"
            )
            for row in node_properties:  # type: ignore
                node_schema["properties"].append(
                    {"name": row[1], "type": row[2]}  # type: ignore
                )
            schema["nodes"].append(node_schema)

        # Edge properties
        for rel in relationships:
            edge = {
                "label": rel["name"],
                "from": rel["from"],
                "to": rel["to"],
                "properties": [],
            }
            rel_properties = self.conn.execute(
                f"CALL TABLE_INFO('{rel['name']}') RETURN *;"
            )
            for row in rel_properties:  # type: ignore
                edge["properties"].append(
                    {"name": row[1], "type": row[2]}  # type: ignore
                )
            schema["edges"].append(edge)

        return schema


# ---------------------------------------------------------------------------
# GraphRAG
# ---------------------------------------------------------------------------


class GraphRAG(dspy.Module):
    def __init__(self, max_attempts: int = 3):
        super().__init__()
        self.max_attempts = max_attempts
        self.prune = dspy.Predict(PruneSchema)
        self.text2cypher = dspy.ChainOfThought(Text2Cypher)
        self.repair = dspy.ChainOfThought(RepairText2Cypher)
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
        self._last_debug: dict[str, Any] = {}

    def get_cypher_query(
        self, question: str, input_schema: str, conn: kuzu.Connection
    ) -> Query:
        # Prune schema
        pruned = self.prune(
            question=question,
            input_schema=input_schema,
        ).pruned_schema

        # Dynamic few-shot
        exs = select_exemplars(question, k=2)
        ex_block = render_exemplars_block(exs)

        # Initial generation
        gen = self.text2cypher(
            question=question,
            input_schema=str(pruned),
            exemplars=ex_block,
        )
        cy = _postprocess_cypher(gen.query.query)

        # Validate / repair loop
        attempts_log: list[dict[str, Any]] = [
            {"stage": "gen", "query": cy, "error": None}
        ]
        ok, err = validate_cypher(conn, cy)
        attempt = 1
        while not ok and attempt < self.max_attempts:
            rep = self.repair(
                question=question,
                input_schema=str(pruned),
                previous_query=cy,
                error_message=err[:800],
            )
            cy = _postprocess_cypher(rep.repaired.query)
            ok, err = validate_cypher(conn, cy)
            attempts_log.append(
                {
                    "stage": f"repair_{attempt}",
                    "query": cy,
                    "error": None if ok else err,
                }
            )
            attempt += 1

        # Keep debug snapshot
        self._last_debug = {
            "picked_exemplars": [e.question for e in exs],
            "exemplars_block": ex_block,
            "attempts": attempts_log,
            "explain_ok": ok,
            "last_error": None if ok else err,
        }
        return Query(query=cy)

    def run_query(
        self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
    ) -> tuple[str, str]:
        conn = db_manager.conn
        q = self.get_cypher_query(question, input_schema, conn)

        # Execute the Cypher query and serialize results as context
        result = conn.execute(q.query)
        cols = result.get_column_names() # type: ignore
        rows = [dict(zip(cols, row)) for row in result]
        context = repr(rows)

        return q.query, context

    def forward(
        self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
    ) -> dict[str, Any]:
        final_query, final_context = self.run_query(db_manager, question, input_schema)
        if not final_context:
            print(
                "Empty results obtained from the graph database. "
                "Please retry with a different question."
            )
            return {}

        answer = self.generate_answer(
            question=question,
            cypher_query=final_query,
            context=str(final_context),
        )
        return {
            "question": question,
            "query": final_query,
            "answer": answer,
            "debug": self._last_debug,
        }

    async def aforward(
        self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
    ) -> dict[str, Any]:
        # Simple async wrapper around forward
        return self.forward(
            db_manager=db_manager,
            question=question,
            input_schema=input_schema,
        )


def run_graph_rag(
    questions: list[str],
    db_manager: KuzuDatabaseManager,
) -> list[Any]:
    schema = str(db_manager.get_schema_dict)
    rag = GraphRAG()
    results: list[Any] = []
    for q in questions:
        results.append(rag(db_manager=db_manager, question=q, input_schema=schema))
    return results


# ---------------------------------------------------------------------------
# marimo UI cells
# ---------------------------------------------------------------------------


@app.cell
def _(mo=marimo):
    mo.md(
        r"""
# Graph RAG using Text2Cypher

This is a demo app in marimo that allows you to query the Nobel laureate graph (managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

> - Powered by Kuzu, DSPy and marimo -
"""
    )
    return


@app.cell
def _(mo=marimo):
    text_ui = mo.ui.text(
        value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        full_width=True,
    )
    return (text_ui,)


@app.cell
def _(text_ui):
    # display the text input widget
    text_ui
    return


@app.cell
def _(
    KuzuDatabaseManager=KuzuDatabaseManager,
    mo=marimo,
    run_graph_rag=run_graph_rag,
    text_ui=None,
):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value if text_ui is not None else ""

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

    query = result["query"]
    answer = result["answer"].response
    return answer, query


@app.cell
def _(answer, mo=marimo, query=None):
    mo.hstack(
        [
            mo.md(f"### Query\n```{query}```"),
            mo.md(f"### Answer\n{answer}"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
