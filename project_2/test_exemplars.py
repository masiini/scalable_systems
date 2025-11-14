from graph_rag  import select_exemplars, EXEMPLARS

def debug_exemplars():
    test_questions = [
        "Who won Nobel Prizes in Physics?",
        "Which laureates are affiliated with Cambridge?",
        "List female Chemistry laureates after 1950.",
        "How many laureates per category are there?",
    ]

    for q in test_questions:
        exs = select_exemplars(q, k=2)
        print(f"\nQ: {q}")
        for i, ex in enumerate(exs, 1):
            print(f"  exemplar {i}: {ex.question}")

if __name__ == "__main__":
    debug_exemplars()