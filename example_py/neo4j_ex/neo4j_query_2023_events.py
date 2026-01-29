from neo4j import GraphDatabase

# =========================
# Neo4j 连接配置
# =========================
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

# =========================
# 查询 2023 年的居住事件
# =========================
def query_2023_events(tx):
    result = tx.run(
        """
        MATCH (h:Entity)-[r:居住于]->(t:Entity)
        WHERE r.year = 2023
        RETURN h.name AS person, r.year AS year, t.name AS city
        """
    )

    # 转成 Python 可读结构
    records = []
    for record in result:
        records.append(
            {
                "person": record["person"],
                "year": record["year"],
                "city": record["city"],
            }
        )

    return records


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    with driver.session() as session:
        events_2023 = session.execute_read(query_2023_events)

        print("2023 年居住事件：")
        for e in events_2023:
            print(f"{e['person']} 在 {e['year']} 年居住于 {e['city']}")

    driver.close()
