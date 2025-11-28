examples = [
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
