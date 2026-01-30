from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_person(tx):
    cypher = """
    CREATE (p:Person {
        姓名: $姓名,
        年龄: $年龄,
        出生日期: $出生日期
    })
    RETURN p
    """
    return tx.run(
        cypher,
        姓名="张三",
        年龄=40,
        出生日期="1985-08-02"
    ).single()

with driver.session() as session:
    record = session.execute_write(create_person)
    print(record["p"])

driver.close()
