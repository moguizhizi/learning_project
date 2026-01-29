from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

def write_literal(tx, h_id, h_name, prop_id, prop_label, value):
    tx.run(
        """
        MERGE (h:Entity {entity_id: $h_id})
        SET h.name = $h_name
        SET h[$prop_key] = $value
        """,
        h_id=h_id,
        h_name=h_name,
        prop_id=prop_id,
        prop_key=prop_label,  # e.g. "年龄"
        value=value,
    )

if __name__ == "__main__":
    with driver.session() as session:
        session.execute_write(
            write_literal,
            h_id=1,
            h_name="李四",
            prop_id="AU_P0002",
            prop_label="年龄",
            value=23,
        )

    driver.close()
