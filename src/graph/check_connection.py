"""Connection check for the local Neo4j instance."""

from src.graph.neo4j_client import Neo4jClient, Neo4jConfigError


def main() -> int:
    try:
        with Neo4jClient() as client:
            print(f"URI      : {client.uri}")
            print(f"User     : {client.user}")
            print(f"Database : {client.database}")

            details = client.verify_connectivity()

            print("\nConnected.")
            print(f"  server : {details.get('name')} {details.get('version')}")
            print(f"  edition: {details.get('edition')}")

            counts = client.node_counts()

            print("\nCurrent graph contents:")

            for key in ("documents", "chunks", "entities", "from_document", "next_chunk"):
                print(f"  {key:15}: {counts.get(key, 0)}")

    except Neo4jConfigError as error:
        print(f"Configuration error: {error}")
        return 2
    except Exception as error:
        print(f"Could not connect: {type(error).__name__}: {error}")
        print("\nIs Neo4j running? Start it with: docker compose up -d")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
