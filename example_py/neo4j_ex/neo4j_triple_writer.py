from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

def write_triple(tx, h_id, h_name, t_id, t_name, prop_id, prop_label):
    tx.run(
        """
        MERGE (h:Entity {entity_id: $h_id})
        SET h.name = $h_name

        MERGE (t:Entity {entity_id: $t_id})
        SET t.name = $t_name

        MERGE (h)-[r:RELATION {prop_id: $prop_id}]->(t)
        SET r.label = $prop_label
        """,
        h_id=h_id,
        h_name=h_name,
        t_id=t_id,
        t_name=t_name,
        prop_id=prop_id,
        prop_label=prop_label,
    )

if __name__ == "__main__":
    with driver.session() as session:
        session.execute_write(
            write_triple,
            h_id=1,
            h_name="张三",
            t_id=2,
            t_name="北京",
            prop_id="AU_P0001",
            prop_label="居住于",
        )

    driver.close()
