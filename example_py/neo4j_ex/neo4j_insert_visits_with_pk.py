from neo4j import GraphDatabase

# =========================
# Neo4j 连接配置
# =========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)

# =========================
# 写入 Visit（visit_id 为主键）
# =========================
def write_visit(tx, person_name, visit_id, visit_date, disease_name):
    """
    (Person)-[:HAS_VISIT]->(Visit)-[:DIAGNOSED_AS]->(Disease)

    visit_id 是 Visit 的业务主键
    """

    tx.run(
        """
        MERGE (p:Person {name: $person_name})

        MERGE (v:Visit {visit_id: $visit_id})
        SET v.date = $visit_date

        MERGE (d:Disease {name: $disease_name})

        MERGE (p)-[:HAS_VISIT]->(v)
        MERGE (v)-[:DIAGNOSED_AS]->(d)
        """,
        person_name=person_name,
        visit_id=visit_id,
        visit_date=visit_date,
        disease_name=disease_name,
    )


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    with driver.session() as session:

        # ===== 张三 =====
        session.execute_write(
            write_visit,
            person_name="张三",
            visit_id="ZS_202501",
            visit_date="2025-01",
            disease_name="感冒",
        )

        session.execute_write(
            write_visit,
            person_name="张三",
            visit_id="ZS_202502",
            visit_date="2025-02",
            disease_name="癌症",
        )

        # ===== 李四 =====
        session.execute_write(
            write_visit,
            person_name="李四",
            visit_id="LS_202501",
            visit_date="2025-01",
            disease_name="感冒",
        )

        session.execute_write(
            write_visit,
            person_name="李四",
            visit_id="LS_202502",
            visit_date="2025-02",
            disease_name="癌症",
        )

    driver.close()
    print("✅ 数据写入完成")
