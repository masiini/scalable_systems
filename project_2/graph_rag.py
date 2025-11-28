import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value
    query = ""
    answer = ""

    # Only run if the question is not empty
    if question:
        with mo.status.spinner(title="Generating answer...") as _spinner:
            result = run_graph_rag([question], db_manager)[0]

        if result:
            query = result.get('query', "")
            answer_obj = result.get('answer')
            if answer_obj:
                answer = answer_obj.response
    return answer, query


@app.cell
def _(answer, mo, query):
    mo.hstack([mo.md(f"""### Query\n```{query}```"""), mo.md(f"""### Answer\n{answer}""")])
    return


# # Evaluation cell for Task 1
# @app.cell
# def _(
#     db_manager,
#     examples_test_1,
#     mo,
#     pd,
#     postprocess_cypher,
#     run_graph_rag,
#     time,
# ):
#     # 10 req/min
#     PER_EXAMPLE_DELAY_S = 7.0
# 
#     def is_rate_limit_error(e: Exception) -> bool:
#         msg = str(e).lower()
#         return (
#             "429" in msg
#             or "rate limit" in msg
#             or "resource_exhausted" in msg
#             or "quota" in msg
#         )
# 
#     def eval_task1(testset, db_manager):
#         rows = []
# 
#         def normalize_for_eval(q: str) -> str:
#             if not q:
#                 return ""
#             q = postprocess_cypher(q)
#             # Canonicalize variable names: (s1:Scholar) -> (v:Scholar)
#             q = re.sub(r"\([a-zA-Z_][a-zA-Z0-9_]*\s*:", "(v:", q)
#             q = re.sub(r"\s+", " ", q).strip().rstrip(";")
#             return q
# 
#         # ---- run ALL questions in one GraphRAG instantiation ----
#         questions = [ex.question for ex in testset]
# 
#         t0_total = time.perf_counter()
#         try:
#             outputs = run_graph_rag(questions, db_manager)  # one SentenceTransformer load
#             batch_err = None
#         except Exception as e:
#             outputs = []
#             batch_err = str(e)
# 
#         t1_total = time.perf_counter()
#         total_latency = t1_total - t0_total
#         per_example_latency = total_latency / len(testset) if testset else 0.0
# 
#         # ---- build rows, stop early on rate limit if needed ----
#         for ex, out in zip(testset, outputs):
#             question = ex.question
#             gold_query = ex.query.query
# 
#             pred_query = out.get("query", "") if out else ""
#             err = out.get("error") if isinstance(out, dict) else None
# 
#             norm_gold = normalize_for_eval(gold_query)
#             norm_pred = normalize_for_eval(pred_query)
# 
#             rows.append(
#                 {
#                     "question": question,
#                     "gold_query": gold_query,
#                     "pred_query": pred_query,
#                     "normalized_match": norm_gold == norm_pred,
#                     # latency is now an estimate (batch/len) since we run as one batch
#                     "latency_s": per_example_latency,
#                     "error": err,
#                 }
#             )
# 
#         # If the whole batch failed, still return a single row explaining why
#         if batch_err and not rows:
#             rows.append(
#                 {
#                     "question": "(batch)",
#                     "gold_query": "",
#                     "pred_query": "",
#                     "normalized_match": False,
#                     "latency_s": total_latency,
#                     "error": batch_err,
#                 }
#             )
# 
#         return pd.DataFrame(rows)
# 
#     with mo.status.spinner(title="Evaluating Task 1..."):
#         df = eval_task1(examples_test_1, db_manager)
# 
#     # Summarize partial/complete run
#     correct = int(df["normalized_match"].sum())
#     total_done = len(df)
#     total_all = len(examples_test_1)
#     acc = correct / total_done if total_done else 0.0
#     mean_lat = float(df["latency_s"].mean()) if total_done else 0.0
# 
#     stopped_early = total_done < total_all
#     stop_note = " (stopped early due to rate limit)" if stopped_early else ""
# 
#     mo.vstack(
#         [
#             mo.md(
#                 f"""
#                     {stop_note}
# 
#                     - **Completed:** {total_done}/{total_all}
#                     - **Accuracy (on completed):** {correct}/{total_done} = **{acc:.1%}**
#                     - **Mean latency:** **{mean_lat:.3f}s**
#                 """
#             ),
#             df,
#         ]
#     )
# 
#     return df


@app.cell
def _(GraphSchema, Query, dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: GraphSchema = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        Translate the question into a valid Cypher query that respects the graph schema.

        <SYNTAX>
        - When matching on Scholar names, ALWAYS match on the `knownName` property
        - For countries, cities, continents and institutions, you can match on the `name` property
        - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
        - Always strive to respect the relationship direction (FROM/TO) using the schema information.
        - When comparing string properties, ALWAYS do the following:
            - Lowercase the property values before comparison
            - Use the WHERE clause
            - Use the CONTAINS operator to check for presence of one substring in the other
        - DO NOT use APOC as the database does not support it.
        </SYNTAX>

        <RETURN_RESULTS>
        - If the result is an integer, return it as an integer (not a string).
        - When returning results, return property values rather than the entire node or relationship.
        - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
        - NO Cypher keywords should be returned by your query.
        </RETURN_RESULTS>
        """

        question: str = dspy.InputField()
        context: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        query: Query = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()

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
    return AnswerQuestion, PruneSchema, RepairText2Cypher, Text2Cypher


@app.cell
def _(BAMLAdapter, GOOGLE_API_KEY, dspy):
    # Using OpenRouter. Switch to another LLM provider as needed
    lm = dspy.LM(
        model="gemini/gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
    )
    dspy.configure(lm=lm, adapter=BAMLAdapter())
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]  # type: ignore
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]  # type: ignore
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:  # type: ignore
                    node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:  # type: ignore
                    edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: str = Field(alias="from", description="Source node label")
        to: str = Field(alias="to", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[str]
        edges: list[Edge]
    return GraphSchema, Query


@app.cell
def _(Query, dspy):
    # Few-shot examples

    examples = [
        dspy.Example(
            question="Which scholars won Nobel Prizes in Physics?",
            input_schema=[
                {
                    "label": "Scholar", 
                    "properties": [{"name": "knownName", "type": "STRING"}]
                }, 
                {
                    "label": "Prize", 
                    "properties": [{"name": "category", "type": "STRING"}, {"name": "awardYear", "type": "INT64"}]
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ],
            query=Query(
                query="MATCH (s:Scholar)-[:WON]->(p:Prize) "
                "WHERE toLower(p.category) = 'physics' "
                "RETURN s.knownName AS scholar, p.awardYear AS year "
                "ORDER BY year"
            ),
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question="Which scholars were affiliated with the University of Cambridge?",
            input_schema=[
                {
                    "label": "Scholar", 
                    "properties": [{"name": "knownName", "type": "STRING"}]
                }, 
                {
                    "label": "Institution",
                    "properties": [{"name": "name", "type": "STRING"}]
                },
                {
                    "label": "AFFILIATED_WITH",
                    "properties": []
                }
            ],
            query=Query(
                query="MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution) "
                "WHERE toLower(i.name) CONTAINS 'cambridge' "
                "RETURN s.knownName AS scholar, i.name AS institution"
            ),
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question="List female Chemistry laureates.",
            input_schema= [
                {
                    "label": "Scholar", 
                    "properties": [{"name": "knownName", "type": "STRING"}]
                }, 
                {
                    "label": "Prize", 
                    "properties": [{"name": "category", "type": "STRING"}, {"name": "awardYear", "type": "INT64"}]
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ],
            query=Query(
                query="MATCH (s:Scholar)-[:WON]->(p:Prize) "
                "WHERE toLower(p.category) = 'chemistry' "
                "AND toLower(s.gender) = 'female' "
                "RETURN s.knownName AS laureate, p.awardYear AS year"
            ),
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question="How many laureates won prizes in each category?",
            input_schema=[
                {
                    "label": "Scholar", 
                    "properties": []
                }, 
                {
                    "label": "Prize", 
                    "properties": [{"name": "category", "type": "STRING"}]
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ],
            query=Query(
                query="MATCH (s:Scholar)-[:WON]->(p:Prize) "
                "RETURN toLower(p.category) AS category, "
                "COUNT(DISTINCT s) AS total "
                "ORDER BY total DESC"
            ),
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "Which scholars have not won any prize?",
            input_schema =  [
                {
                    "label": "Scholar", 
                    "properties": [{"name": "fullName", "type": "STRING"}]
                }, 
                {
                    "label": "Prize", 
                    "properties": []
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ], 
            query =  Query(query="MATCH (s:Scholar) WHERE NOT (s)-[:WON]->(:Prize) RETURN s.fullName")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "Find scholars who died before 1950 and won a prize.",
            input_schema =  [
                {
                    "label": "Scholar", 
                     "properties": [{"name": "fullName", "type": "STRING"}, {"name": "deathDate", "type": "STRING"}]
                },
                {
                    "label": "Prize", 
                     "properties": []
                },
                {
                    "label": "WON",
                    "properties": []
                }
            ],
            query =  Query(query="MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.deathDate < '1950-01-01' RETURN DISTINCT s.fullName")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "Which scholars have won more than one prize?", 
            input_schema =  [
                {
                    "label": "Scholar",
                    "properties": [{"name": "fullName", "type": "STRING"}]
                }, 
                {
                    "label": "Prize", 
                    "properties": []
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ], 
            query =  Query(query="MATCH (s:Scholar)-[w:WON]->(p:Prize) WITH s, COUNT(p) AS prizeCount WHERE prizeCount > 1 RETURN s.fullName, prizeCount")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "How many prizes were awarded in total after the year 2015?", 
            input_schema =  [
                {
                    "label": "Prize",
                    "properties": [{"name": "prize_id", "type": "STRING"}, {"name": "awardYear", "type": "INT64"}]
                }
            ], 
            query =  Query(query="MATCH (p:Prize) WHERE p.awardYear > 2015 RETURN COUNT(p)")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "Find all prize categories awarded in 1901.", 
            input_schema =  [
                {
                    "label": "Prize", 
                    "properties": [{"name": "awardYear", "type": "INT64"}, {"name": "category", "type": "STRING"}]
                }
            ], 
            query =  Query(query="MATCH (p:Prize) WHERE p.awardYear = 1901 RETURN DISTINCT p.category")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "What was the motivation for the 'Peace' prize in 2014?", 
            input_schema =  [
                {
                    "label": "Prize", 
                    "properties": [{"name": "awardYear", "type": "INT64"}, {"name": "category", "type": "STRING"}, {"name": "motivation", "type": "STRING"}]
                }
            ],
            query =  Query(query="MATCH (p:Prize) WHERE p.awardYear = 2014 AND p.category = 'Peace' RETURN p.motivation")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "How many scholars won a prize in 'Literature'?", 
            input_schema =  [
                {
                    "label": "Scholar", 
                    "properties": []
                }, 
                {
                    "label": "Prize", 
                    "properties": [{"name": "category", "type": "STRING"}]
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ], 
            query =  Query(query="MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE p.category = 'Literature' RETURN COUNT(DISTINCT s)")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "When was the scholar 'Max Planck' born?", 
            input_schema =  [
                {
                    "label": "Scholar", 
                    "properties": [{"name": "knownName", "type": "STRING"}, {"name": "birthDate", "type": "STRING"}]
                }
            ], 
            query =  Query(query="MATCH (s:Scholar) WHERE s.knownName = 'Max Planck' RETURN s.birthDate")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "Find the total adjusted prize amount awarded for 'Chemistry'.", 
            input_schema =  [
                {
                    "label": "Prize", 
                    "properties": [{"name": "category", "type": "STRING"}, {"name": "prizeAmountAdjusted", "type": "INT64"}]
                }
            ], 
            query =  Query(query="MATCH (p:Prize) WHERE p.category = 'Chemistry' RETURN SUM(p.prizeAmountAdjusted)")
        ).with_inputs("question", "input_schema"),
        dspy.Example(
            question =  "List all prizes won by 'Marie Curie'.", 
            input_schema =  [
                {
                    "label": "Scholar", 
                    "properties": [{"name": "knownName", "type": "STRING"}]}, 
                {
                    "label": "Prize",
                    "properties": [{"name": "category", "type": "STRING"}, {"name": "awardYear", "type": "INT64"}]
                }, 
                {
                    "label": "WON", 
                    "properties": []
                }
            ], 
            query =  Query(query="MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.knownName = 'Marie Curie' RETURN p.category, p.awardYear")
        ).with_inputs("question", "input_schema")
    ]

    examples_test_1 = [
        # 1. Basic category filter
        dspy.Example(
            question="Which scholars won Nobel Prizes in Physics?",
            input_schema=[{"label":"Scholar","properties":[{"name":"knownName","type":"STRING"}]},
                        {"label":"Prize","properties":[{"name":"category","type":"STRING"},{"name":"awardYear","type":"INT64"}]},
                        {"label":"WON","properties":[]}],
            query=Query(query=
                "MATCH (s:Scholar)-[:WON]->(p:Prize) "
                "WHERE toLower(p.category) = 'physics' "
                "RETURN s.knownName AS scholar, p.awardYear AS year "
                "ORDER BY year"
            ),
        ).with_inputs("question","input_schema"),

        # 2. Institution string match
        dspy.Example(
            question="Which scholars were affiliated with the University of Cambridge?",
            input_schema=[{"label":"Scholar","properties":[{"name":"knownName","type":"STRING"}]},
                        {"label":"Institution","properties":[{"name":"name","type":"STRING"}]},
                        {"label":"AFFILIATED_WITH","properties":[]}],
            query=Query(query=
                "MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution) "
                "WHERE toLower(i.name) CONTAINS 'cambridge' "
                "RETURN DISTINCT s.knownName AS scholar, i.name AS institution"
            ),
        ).with_inputs("question","input_schema"),

        # 3. Two constraints (category + institution)
        dspy.Example(
            question="Which Physics laureates were affiliated with University of Cambridge?",
            input_schema=[{"label":"Scholar","properties":[{"name":"knownName","type":"STRING"}]},
                        {"label":"Prize","properties":[{"name":"category","type":"STRING"},{"name":"awardYear","type":"INT64"}]},
                        {"label":"Institution","properties":[{"name":"name","type":"STRING"}]},
                        {"label":"WON","properties":[]},
                        {"label":"AFFILIATED_WITH","properties":[]}],
            query=Query(query=
                "MATCH (s:Scholar)-[:WON]->(p:Prize), (s)-[:AFFILIATED_WITH]->(i:Institution) "
                "WHERE toLower(p.category) = 'physics' "
                "AND toLower(i.name) CONTAINS 'cambridge' "
                "RETURN DISTINCT s.knownName AS scholar, p.awardYear AS year "
                "ORDER BY year"
            ),
        ).with_inputs("question","input_schema"),

        # 4. Aggregation / count per category
        dspy.Example(
            question="How many laureates won prizes in each category?",
            input_schema=[{"label":"Scholar","properties":[]},
                        {"label":"Prize","properties":[{"name":"category","type":"STRING"}]},
                        {"label":"WON","properties":[]}],
            query=Query(query=
                "MATCH (s:Scholar)-[:WON]->(p:Prize) "
                "RETURN toLower(p.category) AS category, COUNT(DISTINCT s) AS total "
                "ORDER BY total DESC"
            ),
        ).with_inputs("question","input_schema"),

        # 5. Simple temporal count
        dspy.Example(
            question="How many prizes were awarded after 2015?",
            input_schema=[{"label":"Prize","properties":[{"name":"awardYear","type":"INT64"}]}],
            query=Query(query=
                "MATCH (p:Prize) WHERE p.awardYear > 2015 RETURN COUNT(p) AS total"
            ),
        ).with_inputs("question","input_schema"),

        # 6. Property lookup by name
        dspy.Example(
            question="When was Max Planck born?",
            input_schema=[{"label":"Scholar","properties":[{"name":"knownName","type":"STRING"},{"name":"birthDate","type":"STRING"}]}],
            query=Query(query=
                "MATCH (s:Scholar) WHERE s.knownName = 'Max Planck' RETURN s.birthDate AS birthDate"
            ),
        ).with_inputs("question","input_schema"),
    ]
    return (examples, examples_test_1)


@app.cell
def _(examples):
    [str(e.toDict()) for e in examples]
    return


@app.cell
def _(kuzu, re):
    def postprocess_cypher(cypher: str) -> str:
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
    return postprocess_cypher, validate_cypher


@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    RepairText2Cypher,
    SentenceTransformer,
    Text2Cypher,
    cachetools,
    dspy,
    examples,
    json,
    postprocess_cypher,
    validate_cypher,
):
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self, sentence_model, top_k):
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.ChainOfThought(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            self.repair = dspy.ChainOfThought(RepairText2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            self.use_dynamic_shots = True
            self.use_repair = True
            self.use_postprocess = True

            def encode_batch(texts):
                if isinstance(texts, str):
                    texts = [texts]
                embs = sentence_model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                return embs.tolist()

            self.embedder = dspy.Embedder(encode_batch)
            def _render_exemplar(e):
                return f"Q: {e.question}\nA (Cypher): {e.query.query}"

            self.search = dspy.retrievers.Embeddings(
                embedder=self.embedder,
                corpus=[_render_exemplar(e) for e in examples],
                k=top_k
            )

            self._last_debug: dict[str, Any] = {}

            self.lru_cache = cachetools.LRUCache(maxsize=128)

        def get_cypher_query(self, question: str, input_schema: str) -> Query:
            schema_str = json.dumps(input_schema)
            cache_key = hash(f"{question}|{schema_str}")

            if cache_key in self.lru_cache:
                print(f"CACHE HIT: {question} | {input_schema}")
                query = self.lru_cache[cache_key]
                return query, ["Cache hit."]
            else:
                prune_result = self.prune(question=question, input_schema=input_schema)
                schema = prune_result.pruned_schema
                if self.use_dynamic_shots:
                    context = self.search(question).passages
                else:
                    context = []

                text2cypher_result = self.text2cypher(
                    question=question,
                    context=context,
                    input_schema=schema
                )
                cypher_query = text2cypher_result.query
                self.lru_cache[cache_key] = cypher_query
                return cypher_query, context

        def run_query(
            self, db_manager: KuzuDatabaseManager, question: str, input_schema: str
        ) -> tuple[str, list[Any] | None]:
            """
            Run a query synchronously on the database.
            """
            result, context = self.get_cypher_query(question=question, input_schema=input_schema)
            print(f'DEBUG RESULT: {result}')
            query = result.query
            try:
                attempts_log: list[dict[str, Any]] = []

                if self.use_repair:
                    valid = False
                    max_attempts = 5
                    attempts = 0

                    valid, e_msg = validate_cypher(conn=db_manager.conn, cypher=query)
                    attempts_log.append(
                        {"stage": "gen", "query": query, "valid": valid, "error": e_msg}
                    )

                    while not valid and attempts < max_attempts:
                        repair_result = self.repair(
                            question=question,
                            input_schema=input_schema,
                            previous_query=query,
                            error_message=e_msg
                        )
                        query = repair_result.repaired.query
                        valid, e_msg = validate_cypher(conn=db_manager.conn, cypher=query)
                        attempts += 1
                        attempts_log.append(
                            {"stage": "repair", "query": query, "valid": valid, "error": e_msg}
                        )
                else:
                    # baseline
                    valid, e_msg = True, ""
                    attempts_log.append({"stage": "gen", "query": query, "valid": True, "error": ""})

                processed_query = (
                    postprocess_cypher(cypher=query)
                    if self.use_postprocess
                    else query
                )

                # Run the query on the database
                result = db_manager.conn.execute(processed_query)
                cols = result.get_column_names()
                results = [dict(zip(cols, row)) for row in result]

            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None

            self._last_debug = {
                "picked_exemplars": context,
                "attempts": attempts_log,
                "explain_ok": valid,
                "last_error": None if valid else e_msg,
            }
            return query, results

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response

        async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            final_query, final_context = self.run_query(db_manager, question, input_schema)
            if final_context is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                return {}
            else:
                answer = self.generate_answer(
                    question=question, cypher_query=final_query, context=str(final_context)
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        """Runs the GraphRAG pipeline for a list of questions and returns results."""
        schema = str(db_manager.get_schema_dict)
        sentence_model = SentenceTransformer("google/embeddinggemma-300m") # "sentence-transformers/all-MiniLM-L6-v2"
        rag = GraphRAG(sentence_model, top_k=2)
        # Run pipeline
        results = []
        for question in questions:
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            results.append(response)
        return results

    return (run_graph_rag,)


@app.cell
def _(dspy):
    dspy.inspect_history(n=2)
    return


@app.cell
def _():
    import marimo as mo
    import os, re
    from textwrap import dedent
    from typing import Any
    import time
    import pandas as pd

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field

    from sentence_transformers import SentenceTransformer
    import cachetools
    import json

    from phoenix.otel import register

    load_dotenv()

    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    tracer_provider = register(
      project_name="my-graph-rag-app",
      auto_instrument=True
    )
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        GOOGLE_API_KEY,
        SentenceTransformer,
        cachetools,
        dspy,
        json,
        kuzu,
        mo,
        re,
        time,
        pd,
    )


if __name__ == "__main__":
    app.run()
