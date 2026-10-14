# Healthcare GraphRAG Pipeline

A local, evidence-grounded GraphRAG prototype for retrieving information from clinical guideline documents.

The current corpus contains 10 NICE diabetes-related guideline PDFs, processed into 728 searchable chunks. The system combines local vector search with a Neo4j knowledge graph containing provenance links and human-reviewed clinical triplets.

> **Important:** This is an educational retrieval prototype, not clinical decision support. It does not provide diagnosis, treatment, or dosing instructions.

## Features

- Manifest-driven PDF ingestion and deterministic document chunking
- Local embeddings and cosine-similarity vector retrieval
- Neo4j provenance graph for documents, chunks, entities, and chunk ordering
- Human-reviewed clinical relations with source chunk and page evidence
- Hybrid retrieval that combines reviewed graph evidence with vector matches
- Citation-oriented output with document, page, chunk ID, and retrieval method
- Automated test coverage

## Architecture

```text
NICE guideline PDFs
        |
        v
PDF ingestion and chunking
        |
        +--> document_chunks.json
        |
        +--> Local embeddings and vector index
        |
        +--> Neo4j provenance graph
                  |
                  +--> Reviewed clinical triplets
        |
        v
Hybrid evidence retrieval
        |
        v
Cited retrieval output
```

## Project Structure

```text
data/
  manifests/        Clinical-guideline manifest
  processed/        Chunks, vector index, metadata, ingestion reports
  reviewed/         Human-reviewed clinical triplets
  evaluation/       Manual retrieval evaluation notes

src/
  ingestion/        PDF parsing, manifest loading, batch ingestion
  embeddings/       Local embedding index and vector retrieval
  graph/            Neo4j schema, loaders, graph queries
  answering/        Evidence-grounded hybrid retrieval interface

tests/              Automated tests
docker-compose.yml  Local Neo4j service
requirements.txt    Python dependencies
```

## Prerequisites

- Python 3.11 or newer
- Docker Desktop
- Git

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Start Neo4j:

```bash
docker compose up -d
```

Confirm that the database is available:

```bash
python -m src.graph.check_connection
```

Neo4j Browser is available at:

```text
http://localhost:7474
```

## Build the Corpus

Run ingestion to parse the PDFs and create deterministic chunks:

```bash
python -m src.ingestion.batch_ingest
```

Build the local vector index:

```bash
python -m src.embeddings.build_local_index
```

Load document and chunk provenance into Neo4j:

```bash
python -m src.graph.schema
python -m src.graph.load_provenance_graph
```

Load reviewed clinical triplets:

```bash
python -m src.graph.load_reviewed_triplets
```

## Run Retrieval

Ask an evidence-retrieval question:

```bash
python -m src.answering.ask "What technology is recommended for type 1 diabetes?"
```

Other examples:

```bash
python -m src.answering.ask "What recommendations exist for diabetic foot problems?"
python -m src.answering.ask "What treats type 2 diabetes?"
```

The output identifies reviewed graph evidence and vector matches, along with source filename, page number, chunk ID, and retrieval method.

## Validation

Run the automated tests:

```bash
pytest tests/ -q
```

Run vector retrieval checks:

```bash
python -m src.embeddings.run_retrieval_checks
```

Check graph contents:

```bash
python -m src.graph.check_connection
```

The populated graph currently includes:

- 10 document nodes
- 728 chunk nodes
- 728 document-provenance relationships
- 718 sequential chunk relationships
- 36 entity nodes
- 30 reviewed clinical relationships

## Limitations

- The corpus is limited to 10 diabetes-related NICE documents.
- Source URLs in the current manifest are empty and should be enriched.
- Vector retrieval can return relevant but unreviewed passages.
- Reviewed graph relations are intentionally limited in scope.
- Retrieved evidence must not be interpreted as personalized medical advice or insulin-dosing guidance.
