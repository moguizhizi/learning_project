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
# 写入三元组（语义关系作为 type）
# =========================
def write_triple(tx, h_id, h_name, t_id, t_name, prop_id, prop_label):
    """
    (h)-[prop_label]->(t)

    prop_label : 关系语义，如 "居住于"
    prop_id    : 本体 / 规则 ID，如 "AU_P0001"
    """
    cypher = f"""
    MERGE (h:Entity {{entity_id: $h_id}})
    SET h.name = $h_name

    MERGE (t:Entity {{entity_id: $t_id}})
    SET t.name = $t_name

    MERGE (h)-[r:`{prop_label}` {{prop_id: $prop_id}}]->(t)
    """

    tx.run(
        cypher,
        h_id=h_id,
        h_name=h_name,
        t_id=t_id,
        t_name=t_name,
        prop_id=prop_id,
    )


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    with driver.session() as session:
        session.execute_write(
            write_triple,
            h_id=1,
            h_name="李四",
            t_id=2,
            t_name="北京",
            prop_id="AU_P0001",
            prop_label="居住于",
        )

    driver.close()

