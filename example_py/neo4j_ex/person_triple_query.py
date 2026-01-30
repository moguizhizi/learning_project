from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))


def query_person_triples(tx, name):
    cypher = """
    MATCH (p:Person {姓名: $name})
    UNWIND keys(p) AS key
    WITH p, key
    WHERE key <> "姓名"
    RETURN
        p.姓名 AS head,
        key    AS predicate,
        p[key] AS tail
    """
    result = tx.run(cypher, name=name)
    return [(r["head"], r["predicate"], r["tail"]) for r in result]


with driver.session() as session:
    triples = session.execute_read(query_person_triples, "张三")

for t in triples:
    print(t)

driver.close()
