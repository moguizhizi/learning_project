from neo4j import GraphDatabase
from typing import List, Tuple

# =========================
# Neo4j 配置
# =========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # 改成你自己的

# =========================
# Cypher 查询
# =========================
CYPHER_QUERY = """
MATCH (s:TaskExecutionSet {id:$task_id})-[r*1..5]-(n)
UNWIND r AS rel
RETURN DISTINCT
    startNode(rel).id AS head_id,
    coalesce(startNode(rel).name, startNode(rel).id) AS head_name,
    type(rel) AS predicate,
    endNode(rel).id AS tail_id,
    coalesce(endNode(rel).name, endNode(rel).id) AS tail_name
"""

# =========================
# 三元组 → 文本模板
# =========================
PREDICATE_TEMPLATES = {
    "训练": "{head_name}（ID:{head_id}）属于训练任务 {tail_name}",
    "包含": "{head_name} 包含 {tail_name}",
    "隶属于": "{head_name} 隶属于类别 {tail_name}",
    "疾病": "{head_name} 的疾病标签为 {tail_name}",
}

# =========================
# Neo4j 查询函数
# =========================
def fetch_triples(task_id: str) -> List[Tuple]:
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    triples = []

    with driver.session() as session:
        result = session.run(CYPHER_QUERY, task_id=task_id)
        for record in result:
            triples.append((
                record["head_id"],
                record["head_name"],
                record["predicate"],
                record["tail_id"],
                record["tail_name"],
            ))

    driver.close()
    return triples


# =========================
# 三元组 → 中文语义
# =========================
def triple_to_text(triple: Tuple) -> str:
    head_id, head_name, predicate, tail_id, tail_name = triple

    template = PREDICATE_TEMPLATES.get(predicate)
    if template:
        return template.format(
            head_id=head_id,
            head_name=head_name,
            tail_id=tail_id,
            tail_name=tail_name,
        )

    # 兜底（未知关系）
    return f"{head_name}（ID:{head_id}）与 {tail_name}（ID:{tail_id}）存在关系 {predicate}"


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    task_id = "66_20211231"

    triples = fetch_triples(task_id)

    print("====== 原始三元组 ======")
    for t in triples:
        print(t)

    print("\n====== 文本语义 ======")
    for t in triples:
        print(triple_to_text(t))
