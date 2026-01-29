from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)


def write_visit(tx, person_name, visit_date, disease_name):
    tx.run(
        """
        MERGE (p:Person {name: $person_name})

        CREATE (v:Visit {
            visit_id: randomUUID(),
            date: $visit_date
        })

        MERGE (d:Disease {name: $disease_name})

        MERGE (p)-[:HAS_VISIT]->(v)
        MERGE (v)-[:DIAGNOSED_AS]->(d)
        """,
        person_name=person_name,
        visit_date=visit_date,
        disease_name=disease_name,
    )


if __name__ == "__main__":
    with driver.session() as session:
        # 张三
        session.execute_write(write_visit, "张三", "2025-01", "感冒")
        session.execute_write(write_visit, "张三", "2025-02", "癌症")

        # 李四
        session.execute_write(write_visit, "李四", "2025-01", "感冒")
        session.execute_write(write_visit, "李四", "2025-02", "癌症")

    driver.close()
