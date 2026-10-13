"""Minimal Neo4j client for the local provenance graph.

Local Neo4j Community only. Credentials come from .env, which is git-ignored.
"""

import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_DATABASE = "neo4j"

# Driver defaults. Loaders keep these; interactive retrieval overrides them so
# an unreachable database fails in seconds instead of retrying for ~30s.
DEFAULT_CONNECTION_TIMEOUT = 15.0
DEFAULT_MAX_RETRY_TIME = 30.0
FAST_FAIL_CONNECTION_TIMEOUT = 2.0
FAST_FAIL_MAX_RETRY_TIME = 2.0


class Neo4jConfigError(RuntimeError):
    """Raised when required connection settings are missing."""


class Neo4jClient:
    """Thin wrapper around the official driver. Use as a context manager."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        max_transaction_retry_time: float = DEFAULT_MAX_RETRY_TIME,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", DEFAULT_URI)
        self.user = user or os.getenv("NEO4J_USER", DEFAULT_USER)
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE)

        if not self.password:
            raise Neo4jConfigError(
                "NEO4J_PASSWORD is not set. Copy .env.example to .env and set it."
            )

        self._driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=connection_timeout,
            max_transaction_retry_time=max_transaction_retry_time,
        )

    @classmethod
    def fast_fail(cls) -> "Neo4jClient":
        """Client for interactive use: gives up quickly when Neo4j is down."""
        return cls(
            connection_timeout=FAST_FAIL_CONNECTION_TIMEOUT,
            max_transaction_retry_time=FAST_FAIL_MAX_RETRY_TIME,
        )

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exception_details: object) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def run_write(self, query: str, **parameters: object) -> list[dict]:
        with self._driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda transaction: [
                    record.data() for record in transaction.run(query, **parameters)
                ]
            )

        return result

    def run_read(self, query: str, **parameters: object) -> list[dict]:
        with self._driver.session(database=self.database) as session:
            result = session.execute_read(
                lambda transaction: [
                    record.data() for record in transaction.run(query, **parameters)
                ]
            )

        return result

    def verify_connectivity(self) -> dict:
        """Raise if the database is unreachable, otherwise return server details."""
        self._driver.verify_connectivity()

        rows = self.run_read(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name AS name, versions[0] AS version, edition AS edition"
        )

        return rows[0] if rows else {}

    def node_counts(self) -> dict:
        rows = self.run_read(
            """
            CALL () { MATCH (d:Document) RETURN count(d) AS documents }
            CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH ()-[r:FROM_DOCUMENT]->() RETURN count(r) AS from_document }
            CALL () { MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) AS next_chunk }
            RETURN documents, chunks, entities, from_document, next_chunk
            """
        )

        return rows[0] if rows else {}
