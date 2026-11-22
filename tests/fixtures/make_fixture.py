#!/usr/bin/env python3
"""Write the test and review fixture, tests/fixtures/graph/, as the metadata export
writes a graph: nodes/edges as KGX TSV (gzipped) and Parquet, and graph.json.

A small dataset: three subjects, three samples, two protocols and four ontology terms,
with samples of a model bound to a Biolink class (categories [biolink:MaterialSample,
pennsieve:sample]) and one of each data-quality problem the app flags:

  - a dangling edge (a sample linked to a record that isn't in the node file)
  - a duplicate node id (sample 2's row twice, the second with another name and category)
  - an empty name (protocol B)
  - a bare-prefix category (MESH, which has no Biolink class)
  - a self-loop (sample 1 derived from itself)
  - dropped edges (2, in graph.json)

Components: the subjects, samples and terms (9 nodes); protocols A and B (2); subject 3
alone (isolated). Run it again after changing the rows: python3 tests/fixtures/make_fixture.py
"""

import gzip
import io
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "graph")
DATASET = "pennsieve:3f0c5a52-8f3e-4a51-9a5e-2d1b7c6e9a10"

S1 = "pennsieve:0b9f6c1e-1d2a-4c3b-8e4f-5a6b7c8d9e01"
S2 = "pennsieve:1c0a7d2f-2e3b-4d4c-9f50-6b7c8d9e0f12"
S3 = "pennsieve:2d1b8e30-3f4c-4e5d-a061-7c8d9e0f1a23"
X1 = "pennsieve:3e2c9f41-405d-4f6e-b172-8d9e0f1a2b34"
X2 = "pennsieve:4f3da052-516e-4071-c283-9e0f1a2b3c45"
X3 = "pennsieve:504eb163-627f-4182-d394-af1a2b3c4d56"
P1 = "pennsieve:615fc274-7380-4293-e4a5-b02b3c4d5e67"
P2 = "pennsieve:7260d385-8491-43a4-f5b6-c13c4d5e6f78"
MISSING = "pennsieve:83710e96-95a2-44b5-a6c7-d24d5e6f7089"
BRAIN, LIVER, MOUSE, MESH_LIVER = "UBERON:0000955", "UBERON:0002107", "NCBITaxon:10090", "MESH:D008099"
# The sample model is bound to a Biolink class.
SAMPLE_CLASS = "biolink:MaterialSample"


def term(curie, category, name, source):
    return (curie, [category], name, [source] if source else [], json.dumps({"ontology": curie.split(":")[0]}))


def record(rid, model, name, fields, biolink=None):
    """A record node; a model bound to a Biolink class has that class first, as the export
    writes it ([biolink:X, pennsieve:<model>])."""
    categories = ([biolink] if biolink else []) + [f"pennsieve:{model}"]
    return (rid, categories, name, [DATASET], json.dumps(fields, separators=(",", ":")))


NODES = [
    record(S1, "subject", "mouse 1", {"subject_id": "mouse 1", "species": MOUSE}),
    record(S2, "subject", "mouse 2", {"subject_id": "mouse 2", "species": MOUSE}),
    record(S3, "subject", "mouse 3", {"subject_id": "mouse 3"}),
    record(X1, "sample", "brain slice 1", {"sample_id": "brain slice 1", "anatomy": BRAIN}, SAMPLE_CLASS),
    record(X2, "sample", "brain slice 2", {"sample_id": "brain slice 2", "anatomy": BRAIN}, SAMPLE_CLASS),
    # The duplicate: a second row for sample 2, with another name and category; the first row counts.
    record(X2, "other", "brain slice 2 (copy)", {}),
    record(X3, "sample", "liver sample", {"sample_id": "liver sample", "anatomy": LIVER, "mesh_term": MESH_LIVER},
           SAMPLE_CLASS),
    record(P1, "protocol", "protocol A", {"title": "protocol A"}),
    record(P2, "protocol", "", {}),
    term(BRAIN, "biolink:AnatomicalEntity", "brain", "infores:uberon"),
    term(LIVER, "biolink:AnatomicalEntity", "liver", "infores:uberon"),
    term(MOUSE, "biolink:OrganismTaxon", "Mus musculus", "infores:ncbi-taxon"),
    term(MESH_LIVER, "MESH", "Liver", "infores:mesh"),
]

# (subject, predicate, object, pennsieve_property)
EDGES = [
    (S1, "pennsieve:has_sample", X1, ""),
    (S1, "pennsieve:has_sample", X2, ""),
    (S2, "pennsieve:has_sample", X3, ""),
    (X1, "biolink:has_attribute", BRAIN, "anatomy"),
    (X2, "biolink:has_attribute", BRAIN, "anatomy"),
    (X3, "biolink:has_attribute", LIVER, "anatomy"),
    (X3, "biolink:has_attribute", MESH_LIVER, "mesh_term"),
    (S1, "biolink:has_attribute", MOUSE, "species"),
    (S2, "biolink:has_attribute", MOUSE, "species"),
    (X1, "pennsieve:derived_from", X1, ""),
    (X2, "pennsieve:has_sample", MISSING, ""),
    (P1, "pennsieve:follows", P2, ""),
]

NODE_COLUMNS = ["id", "category", "name", "provided_by", "properties"]
EDGE_COLUMNS = ["id", "subject", "predicate", "object", "knowledge_level", "agent_type",
                "primary_knowledge_source", "original_predicate", "pennsieve_property"]


def edge_rows():
    for i, (s, p, o, prop) in enumerate(EDGES):
        yield (f"urn:uuid:00000000-0000-4000-8000-{i:012d}", s, p, o, "knowledge_assertion", "manual_agent",
               "infores:pennsieve", "", prop)


def write_tsv(path, columns, table):
    # mtime 0: the same rows give the same bytes.
    with gzip.GzipFile(path, "wb", mtime=0) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for row in table:
            f.write("\t".join("|".join(v) if isinstance(v, list) else v for v in row) + "\n")


def write_parquet(path, columns, table, lists):
    data = {c: [r[i] for r in table] for i, c in enumerate(columns)}
    schema = pa.schema([(c, pa.list_(pa.string()) if c in lists else pa.string()) for c in columns])
    pq.write_table(pa.table(data, schema=schema), path)


def main():
    os.makedirs(OUT, exist_ok=True)
    edges = list(edge_rows())
    write_tsv(os.path.join(OUT, "nodes.tsv.gz"), NODE_COLUMNS, NODES)
    write_tsv(os.path.join(OUT, "edges.tsv.gz"), EDGE_COLUMNS, edges)
    write_parquet(os.path.join(OUT, "nodes.parquet"), NODE_COLUMNS, NODES, {"category", "provided_by"})
    write_parquet(os.path.join(OUT, "edges.parquet"), EDGE_COLUMNS, edges, set())
    graph = {
        "layout": "kgx", "layout_version": "1.0", "conforms_to": "https://schemas.pennsieve.io/structures/kgx",
        "biolink_version": "4.2.2", "biolink": "partial", "canonical": "tsv",
        "files": {"nodes": "nodes.tsv.gz", "edges": "edges.tsv.gz", "summary": "graph_stats.yaml"},
        "source": {"dataset_id": 1, "dataset_node_id": "N:dataset:3f0c5a52-8f3e-4a51-9a5e-2d1b7c6e9a10", "mode": "view"},
        "generated_at": "2026-11-22T12:00:00Z",
        "prefixes": {"pennsieve": "urn:pennsieve:", "biolink": "https://w3id.org/biolink/vocab/",
                     "UBERON": "http://purl.obolibrary.org/obo/UBERON_",
                     "NCBITaxon": "http://purl.obolibrary.org/obo/NCBITaxon_", "MESH": "http://id.nlm.nih.gov/mesh/"},
        "models": [
            {"name": "protocol", "version": 1, "category": ["pennsieve:protocol"], "name_property": "title"},
            {"name": "sample", "version": 1, "category": [SAMPLE_CLASS, "pennsieve:sample"], "name_from": ["sample_id"]},
            {"name": "subject", "version": 1, "category": ["pennsieve:subject"], "name_from": ["subject_id"]},
        ],
        "counts": {"nodes": len(NODES), "edges": len(EDGES), "dropped_edges": 2},
    }
    with open(os.path.join(OUT, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
