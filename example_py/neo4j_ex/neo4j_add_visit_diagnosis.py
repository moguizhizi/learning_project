from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

def add_diagnosis(tx, visit_id, disease_name):
    """
    给已有 visit 增加一个诊断
    """
    tx.run(
        """
        MATCH (v:Visit {visit_id: $visit_id})

        MERGE (d:Disease {name: $disease_name})

        MERGE (v)-[:DIAGNOSED_AS]->(d)
        """,
        visit_id=visit_id,
        disease_name=disease_name,
    )


if __name__ == "__main__":
    with driver.session() as session:
        session.execute_write(
            add_diagnosis,
            visit_id="ZS_202501",
            disease_name="肺病",
        )

    driver.close()
    print("✅ 已为 ZS_202501 增加诊断：肺病")