from neo4j import GraphDatabase

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "Huawei@123")

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    session.run(
        """
        CREATE (p:Patient {
            id: $id,
            age: $age,
            gender: $gender
        })
        """,
        id="P002",
        age=65,
        gender="male"
    )

driver.close()
print("✅ 数据写入成功")