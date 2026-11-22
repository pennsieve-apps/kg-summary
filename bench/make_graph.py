#!/usr/bin/env python3
"""Write a synthetic KGX graph the size of Hetionet on Pennsieve (92,543 nodes,
2,295,711 edges) to measure kg-summary's time and memory.

  python3 bench/make_graph.py OUT_DIR [--tsv]

Shaped like the Hetionet export: ~47k record nodes (pennsieve:<uuid>) in 11 models,
~45k term nodes (CURIEs, bare-prefix or Biolink categories) each the value of one
record's has_attribute edge, and ~2.25M relationships between records whose ends are
drawn from a heavy-tailed distribution, so a few hubs have tens of thousands of edges. Writes OUT_DIR/graph/{nodes,edges}.parquet, or with --tsv the .tsv.gz pair,
and graph.json. Random but seeded: the same graph every time.
"""

import gzip
import json
import os
import sys
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RECORDS = 47_031
RELATIONSHIPS = 2_250_197
MODELS = ["anatomy", "biological_process", "cellular_component", "compound", "disease", "gene",
          "molecular_function", "pathway", "pharmacologic_class", "side_effect", "symptom"]
# Share of records per model (Hetionet's), and the ontology of each model's identifier.
SHARES = [402, 11381, 1391, 1552, 137, 20945, 2884, 1822, 345, 5734, 438]
ONTOLOGY = {"anatomy": ("UBERON", "biolink:AnatomicalEntity"), "biological_process": ("GO", "GO"),
            "cellular_component": ("GO", "GO"), "compound": ("DRUGBANK", "DRUGBANK"), "disease": ("DOID", "DOID"),
            "gene": ("NCBIGene", "NCBIGene"), "molecular_function": ("GO", "GO"), "side_effect": ("UMLS", "UMLS"),
            "symptom": ("MESH", "MESH")}
PREDICATES = [f"pennsieve:{p}" for p in ("interacts", "participates", "regulates", "binds", "expresses", "upregulates",
                                          "downregulates", "associates", "treats", "palliates", "causes", "resembles",
                                          "localizes", "presents", "includes", "covaries")]


def uuids(rng, n):
    raw = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    return [str(uuid.UUID(bytes=b.tobytes())) for b in raw]


def main():
    out = os.path.join(sys.argv[1], "graph")
    tsv = "--tsv" in sys.argv
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)

    model_of = np.repeat(np.arange(len(MODELS)), SHARES)
    record_ids = ["pennsieve:" + u for u in uuids(rng, RECORDS)]
    node_id, node_cat, node_name = list(record_ids), [[f"pennsieve:{MODELS[m]}"] for m in model_of], [
        f"{MODELS[m]} {i}" for i, m in enumerate(model_of)]
    attr_s, attr_o, attr_prop = [], [], []
    for i, m in enumerate(model_of):
        if MODELS[m] in ONTOLOGY:
            prefix, category = ONTOLOGY[MODELS[m]]
            curie = f"{prefix}:{i:07d}"
            node_id.append(curie)
            node_cat.append([category])
            node_name.append(curie)
            attr_s.append(record_ids[i])
            attr_o.append(curie)
            attr_prop.append("identifier")
    nodes = pa.table({
        "id": node_id, "category": node_cat, "name": node_name,
        "provided_by": [["pennsieve:00000000-0000-4000-8000-000000000000"]] * len(node_id),
        "properties": ['{"identifier":"x"}'] * len(node_id),
    })

    # Relationship ends: record k (in a shuffled order, so hubs are spread over models) is
    # drawn with weight k^-0.6, which gives Hetionet-like hubs of ~25k edges.
    weights = np.arange(1, RECORDS + 1) ** -0.6
    ends = rng.permutation(RECORDS)[rng.choice(RECORDS, size=(RELATIONSHIPS, 2), p=weights / weights.sum())]
    rec = np.array(record_ids, dtype=object)
    n_edges = RELATIONSHIPS + len(attr_s)
    edges = pa.table({
        "id": ["urn:uuid:" + u for u in uuids(rng, n_edges)],
        "subject": np.concatenate([np.array(attr_s, dtype=object), rec[ends[:, 0]]]),
        "predicate": ["biolink:has_attribute"] * len(attr_s)
                     + list(np.array(PREDICATES, dtype=object)[rng.integers(0, len(PREDICATES), RELATIONSHIPS)]),
        "object": np.concatenate([np.array(attr_o, dtype=object), rec[ends[:, 1]]]),
        "knowledge_level": ["knowledge_assertion"] * n_edges,
        "agent_type": ["manual_agent"] * n_edges,
        "primary_knowledge_source": ["infores:pennsieve"] * n_edges,
        "original_predicate": [""] * n_edges,
        "pennsieve_property": attr_prop + [""] * RELATIONSHIPS,
    })

    if tsv:
        nodes = nodes.set_column(1, "category", pa.array(["|".join(c) for c in node_cat]))
        nodes = nodes.set_column(3, "provided_by", pa.array(["|".join(p) for p in nodes.column("provided_by").to_pylist()]))
        # KGX TSV is unquoted, which pyarrow's CSV writer refuses for JSON values: join by hand.
        for name, table in (("nodes", nodes), ("edges", edges)):
            columns = [table.column(c).to_pylist() for c in table.column_names]
            with gzip.open(os.path.join(out, f"{name}.tsv.gz"), "wt", compresslevel=6, encoding="utf-8") as f:
                f.write("\t".join(table.column_names) + "\n")
                f.writelines("\t".join(row) + "\n" for row in zip(*columns))
    else:
        pq.write_table(nodes, os.path.join(out, "nodes.parquet"))
        pq.write_table(edges, os.path.join(out, "edges.parquet"))
    with open(os.path.join(out, "graph.json"), "w") as f:
        json.dump({"layout": "kgx", "layout_version": "1.0", "biolink": "partial", "biolink_version": "4.2.2",
                   "counts": {"nodes": nodes.num_rows, "edges": edges.num_rows, "dropped_edges": 0}}, f)
    print(f"wrote {nodes.num_rows:,} nodes and {edges.num_rows:,} edges to {out}")


if __name__ == "__main__":
    main()
