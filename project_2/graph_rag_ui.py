from __future__ import annotations

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


# Top cell: import marimo as mo and share it
@app.cell
def _():
    import marimo as mo
    from graph_rag import KuzuDatabaseManager, run_graph_rag

    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    mo.md(
        r"""
This is a demo app that lets you query the Nobel laureate graph
(stored in Kuzu) using natural language.

- Your question → Cypher via a DSPy Text2Cypher pipeline  
- Cypher runs on Kuzu  
- Results are fed back to the LLM to generate an answer  

> Powered by Kuzu, DSPy, and marimo.
"""
    )
    return mo, run_graph_rag, db_manager


# Inputs: question + run button
@app.cell
def _(mo):
    question_input = mo.ui.text(
        value=(
            "Which scholars won prizes in Physics and were "
            "affiliated with University of Cambridge?"
        ),
        label="Question",
    )
    run_btn = mo.ui.run_button(label="Run")

    mo.vstack([question_input, run_btn])

    return question_input, run_btn


# Run GraphRAG when the button is clicked, with debug info
@app.cell
def _(mo, question_input, run_btn, run_graph_rag, db_manager):
    answer_text = ""
    query = ""
    error_msg = ""
    debug_info = {
        "run_btn_value": run_btn.value,
        "question": None,
        "result_type": None,
        "result_repr": None,
        "parsed_keys": None,
    }

    if run_btn.value:
        question = (question_input.value or "").strip()
        debug_info["question"] = question

        if not question:
            error_msg = "Please enter a question."
        else:
            try:
                with mo.status.spinner("Running GraphRAG..."):
                    result = run_graph_rag([question], db_manager)

                debug_info["result_type"] = type(result).__name__
                debug_info["result_repr"] = repr(result)[:1000]

                try:
                    first = result[0]
                except Exception as e:
                    error_msg = f"Result is not indexable: {type(e).__name__}: {e}"
                else:
                    if isinstance(first, dict):
                        debug_info["parsed_keys"] = list(first.keys())
                        query = first.get("query", "")
                        answer_obj = first.get("answer")
                        answer_text = (
                            getattr(answer_obj, "response", "")
                            if answer_obj is not None
                            else ""
                        )
                        if not query and not answer_text and not error_msg:
                            error_msg = "GraphRAG returned no query or answer."
                    else:
                        error_msg = (
                            "Unexpected first element type from GraphRAG: "
                            f"{type(first).__name__}"
                        )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"

    return answer_text, query, error_msg, debug_info


# Display the results + debug
@app.cell
def _(mo, answer_text, query, error_msg, debug_info):
    blocks = []

    if error_msg:
        blocks.append(mo.md(f"**Error:** {error_msg}"))
    elif not query and not answer_text:
        blocks.append(mo.md("> Enter a question above and press **Run**."))
    else:
        blocks.append(mo.md(f"### Query\n```cypher\n{query}\n```"))
        blocks.append(mo.md(f"### Answer\n\n{answer_text}"))

    # Always show debug info at the bottom
    blocks.append(
        mo.md(
            "### Debug\n"
            "```python\n"
            f"{debug_info!r}\n"
            "```"
        )
    )

    mo.vstack(blocks)


if __name__ == "__main__":
    app.run()
