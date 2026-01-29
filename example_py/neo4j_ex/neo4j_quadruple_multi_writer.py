from neo4j import GraphDatabase

# =========================
# Neo4j 连接配置
# =========================
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

# =========================
# 写入四元组
# (h, r, t, year)
# =========================
def write_quadruple(tx, h_name, t_name, year):
    tx.run(
        """
        MERGE (h:Entity {name: $h_name})
        MERGE (t:Entity {name: $t_name})
        MERGE (h)-[r:居住于 {year: $year}]->(t)
        """,
        h_name=h_name,
        t_name=t_name,
        year=year,
    )


if __name__ == "__main__":
    with driver.session() as session:
        # 第一条四元组
        session.execute_write(
            write_quadruple,
            h_name="张三",
            t_name="北京",
            year=2023,
        )

        # 第二条四元组
        session.execute_write(
            write_quadruple,
            h_name="张三",
            t_name="南京",
            year=2024,
        )

    driver.close()
