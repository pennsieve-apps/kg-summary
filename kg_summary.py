#!/usr/bin/env python3
"""kg-summary: a readable summary of a dataset's knowledge graph.

The graph is a kgx input (conformsTo: kgx, placement: subfolder) in INPUT_DIR/graph,
laid out as the kgx 1.0 structure says (https://schemas.pennsieve.io/structures/kgx):

  nodes.tsv.gz, edges.tsv.gz    KGX TSV, gzipped, unquoted (the canonical files)
  nodes.parquet, edges.parquet  the same rows, typed (read when both are there)
  graph.json                    provenance, prefixes, models and counts
  graph_stats.yaml              kgx graph-summary form (not read)

Nodes have id, category (a list; "|"-joined in TSV), name, provided_by and properties;
edges have id, subject, predicate, object and provenance columns, plus
pennsieve_property on Pennsieve graphs. Only id, subject, predicate and object are
required, so KGX graphs from elsewhere work too.

The summary has six sections: size, schema (the category -> predicate -> category
metagraph), connectivity (weakly connected components, degree distribution), hubs,
ontology use and data-quality flags. It writes report.json, report.md, a few CSVs with
the full tables and a degree histogram (PNG) to OUTPUT_DIR, prints the markdown and a
last line KG_SUMMARY {...} to stdout, and exits 1 when the graph can't be read (the
report then says why).

Scale: DuckDB reads the files and does the grouping and joins; components and degrees
are computed over integer node indexes with NumPy and SciPy, so no Python object is
made per edge. A graph of 100k nodes and 2.3M edges takes seconds and well under a GB.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback

import duckdb
import numpy as np
import pyarrow as pa
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# The app's own parameters (app.yml), with their defaults.
PARAMETERS = {
    "TOP_N": "25",
    "TOP_PER_CATEGORY": "5",
}

GRAPH_PORT = "graph"
SPEC = "https://schemas.pennsieve.io/structures/kgx/v1.0"

# Record nodes are pennsieve:<record uuid>; any other id is an ontology term (a CURIE).
RECORD_PREFIX = "pennsieve:"
# Categories are expected to be Biolink classes or Pennsieve models; anything else (a
# bare ontology prefix such as MESH) is flagged for information.
KNOWN_CATEGORY_PREFIXES = ("biolink:", "pennsieve:")
HAS_ATTRIBUTE = "biolink:has_attribute"
NO_CATEGORY = "(none)"

# Examples kept per data-quality flag.
MAX_EXAMPLES = 10
# Rows in the markdown tables; the CSVs and report.json have more.
MD_ROWS = 15
# Rows of a table in report.json; the CSVs are complete.
MAX_LISTED = 1000
# The TSV writer bounds a node line at 64 MiB (its properties). DuckDB's read buffer
# defaults to 16 times the line bound (1 GiB here), so it is set to twice the bound; edges
# (short lines) keep DuckDB's defaults.
MAX_NODE_LINE_BYTES = 64 << 20

# Output files.
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
HISTOGRAM_PNG = "degree_histogram.png"


class GraphError(Exception):
    """The graph input is missing or can't be read; the step fails with this message."""


def log(message: str) -> None:
    print(f"[kg-summary] {message}", flush=True)


def param(name: str) -> str:
    return os.environ.get(name, PARAMETERS[name]).strip()


def param_int(name: str, low: int, high: int) -> int:
    raw = param(name)
    try:
        value = int(float(raw))
    except (ValueError, OverflowError):
        log(f"{name}={raw!r} is not a number; using {PARAMETERS[name]}")
        value = int(PARAMETERS[name])
    return max(low, min(high, value))


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sql_string(value: str) -> str:
    """A SQL string literal (file paths in read_parquet / read_csv)."""
    return "'" + value.replace("'", "''") + "'"


def rows(con, query: str) -> list[dict]:
    """A query's rows as dicts."""
    result = con.execute(query)
    names = [d[0] for d in result.description]
    return [dict(zip(names, r)) for r in result.fetchall()]


def scalar(con, query: str):
    return con.execute(query).fetchone()[0]


# --- Reading the graph -----------------------------------------------------------------


def find_files(graph_dir: str) -> dict:
    """Which node and edge files to read: the Parquet pair when both are there, else the
    TSV pair (gzipped, or plain as GRAPE users leave it)."""
    if not os.path.isdir(graph_dir):
        raise GraphError(f"no graph input: {graph_dir} is not a folder (link a dataset's graph to the '{GRAPH_PORT}' input)")

    def present(name: str) -> str | None:
        path = os.path.join(graph_dir, name)
        return path if os.path.isfile(path) else None

    if present("nodes.parquet") and present("edges.parquet"):
        return {"format": "parquet", "nodes": present("nodes.parquet"), "edges": present("edges.parquet")}
    for suffix in (".tsv.gz", ".tsv"):
        if present("nodes" + suffix) and present("edges" + suffix):
            return {"format": "tsv", "nodes": present("nodes" + suffix), "edges": present("edges" + suffix)}
    raise GraphError(f"no node and edge files in {graph_dir}: expected nodes.parquet and edges.parquet, or nodes.tsv.gz and edges.tsv.gz")


def read_graph_json(graph_dir: str) -> dict | None:
    path = os.path.join(graph_dir, "graph.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        log(f"graph.json can't be read: {e}")
        return {"error": str(e)[:300]}
    return doc if isinstance(doc, dict) else {"error": "graph.json is not an object"}


def source(files: dict, kind: str) -> str:
    """A table function over the node or edge file. KGX TSV is unquoted: a quote is an
    ordinary character, so quoting and escaping are off; every column is text."""
    path = sql_string(files[kind])
    if files["format"] == "parquet":
        return f"read_parquet({path})"
    line = f", max_line_size={MAX_NODE_LINE_BYTES}, buffer_size={2 * MAX_NODE_LINE_BYTES}" if kind == "nodes" else ""
    return f"read_csv({path}, delim='\t', header=true, quote='', escape='', all_varchar=true{line})"


def column_types(con, table_function: str) -> dict:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM {table_function}").fetchall()}


def open_graph(con, files: dict) -> dict:
    """Define the views nodes_src(id, category list, name) and edges_src(id, subject,
    predicate, object, prop), whatever the file format, and return the columns found."""
    nodes, edges = source(files, "nodes"), source(files, "edges")
    node_cols, edge_cols = column_types(con, nodes), column_types(con, edges)
    for name, cols, required in (("nodes", node_cols, ("id",)),
                                 ("edges", edge_cols, ("subject", "predicate", "object"))):
        missing = [c for c in required if c not in cols]
        if missing:
            raise GraphError(f"the {name} file has no {', '.join(missing)} column (columns: {', '.join(cols)})")

    # category is list<string> in Parquet and "|"-joined text in TSV; empty values go.
    ctype = node_cols.get("category")
    if ctype is None:
        category = "NULL::VARCHAR[]"
    elif ctype.endswith("[]"):
        category = "list_filter(category, lambda c: coalesce(c, '') <> '')"
    else:
        category = "list_filter(string_split(coalesce(category, ''), '|'), lambda c: c <> '')"
    # An empty name is NULL whatever the format (DuckDB reads an empty TSV value as NULL).
    name = "nullif(name::VARCHAR, '')" if "name" in node_cols else "NULL::VARCHAR"
    con.execute(f"CREATE VIEW nodes_src AS SELECT id::VARCHAR AS id, {category} AS category, {name} AS name FROM {nodes}")

    edge_id = "id::VARCHAR" if "id" in edge_cols else "NULL::VARCHAR"
    prop = "nullif(pennsieve_property::VARCHAR, '')" if "pennsieve_property" in edge_cols else "NULL::VARCHAR"
    con.execute(f"""CREATE VIEW edges_src AS SELECT {edge_id} AS id, subject::VARCHAR AS subject,
        predicate::VARCHAR AS predicate, object::VARCHAR AS object, {prop} AS prop FROM {edges}""")
    return {"nodes": sorted(node_cols), "edges": sorted(edge_cols)}


def load(con) -> None:
    """Materialise the graph once, compactly:

    node(idx, id, category, primary, name, copies): one row per distinct id, idx 0..n-1;
      primary is the first category, used wherever a node needs one category.
    edge(s, o, predicate, prop, id_hash, missing_subject, missing_object): endpoints as
      node indexes (NULL when the node isn't in the node file; the id is then kept), and
      a hash of the edge id to find duplicate ids without keeping the ids.

    The edge file is read once here (a TSV is decompressed and parsed only once), except
    to list examples of a problem found.
    """
    con.execute("""CREATE TABLE node AS
        SELECT (row_number() OVER (ORDER BY id) - 1)::INTEGER AS idx, id, category,
               coalesce(category[1], '(none)') AS primary_category, name, copies
        FROM (SELECT id, first(category) AS category, first(name) AS name, count(*) AS copies
              FROM nodes_src WHERE coalesce(id, '') <> '' GROUP BY id)""")
    con.execute("""CREATE TABLE edge AS
        SELECT s.idx AS s, o.idx AS o, e.predicate, e.prop, hash(nullif(e.id, '')) AS id_hash,
               CASE WHEN s.idx IS NULL THEN coalesce(e.subject, '') END AS missing_subject,
               CASE WHEN o.idx IS NULL THEN coalesce(e.object, '') END AS missing_object
        FROM edges_src e
        LEFT JOIN node s ON s.id = e.subject
        LEFT JOIN node o ON o.id = e.object""")


# --- Sections --------------------------------------------------------------------------


def size_section(con, graph_json: dict | None) -> dict:
    nodes = scalar(con, "SELECT count(*) FROM node")
    records = scalar(con, f"SELECT count(*) FROM node WHERE starts_with(id, {sql_string(RECORD_PREFIX)})")
    # A node with several categories counts under each, as KGX summaries count them.
    by_category = rows(con, f"""SELECT coalesce(c, {sql_string(NO_CATEGORY)}) AS category, count(*) AS nodes
        FROM (SELECT unnest(CASE WHEN len(coalesce(category, [])) = 0 THEN [NULL] ELSE category END) AS c FROM node)
        GROUP BY 1 ORDER BY nodes DESC, category""")
    by_predicate = rows(con, """SELECT coalesce(predicate, '(none)') AS predicate, count(*) AS edges
        FROM edge GROUP BY 1 ORDER BY edges DESC, predicate""")
    counts = (graph_json or {}).get("counts") or {}
    return {
        "nodes": nodes,
        "edges": scalar(con, "SELECT count(*) FROM edge"),
        "recordNodes": records,
        "termNodes": nodes - records,
        "categories": len(by_category),
        "predicates": len(by_predicate),
        "nodesByCategory": by_category,
        "edgesByPredicate": by_predicate,
        "biolink": (graph_json or {}).get("biolink"),
        "biolinkVersion": (graph_json or {}).get("biolink_version"),
        "graphJsonCounts": {k: counts.get(k) for k in ("nodes", "edges", "dropped_edges") if k in counts},
    }


def schema_section(con) -> dict:
    """The metagraph: how many edges join each pair of (primary) categories by each
    predicate. Edges with an endpoint missing from the node file are left out."""
    triples = rows(con, """SELECT sn.primary_category AS subject_category, coalesce(e.predicate, '(none)') AS predicate,
            onode.primary_category AS object_category, count(*) AS edges
        FROM edge e JOIN node sn ON sn.idx = e.s JOIN node onode ON onode.idx = e.o
        GROUP BY ALL ORDER BY edges DESC, subject_category, predicate, object_category""")
    return {"tripleCount": len(triples), "triples": triples}


def degree_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"nodes": 0, "min": None, "median": None, "p90": None, "p99": None, "max": None, "mean": None}
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return {"nodes": int(values.size), "min": int(values.min()), "median": float(p50), "p90": float(p90),
            "p99": float(p99), "max": int(values.max()), "mean": round(float(values.mean()), 3)}


def connectivity_section(con) -> tuple[dict, np.ndarray]:
    """Weakly connected components and degrees over integer node indexes. Degree counts
    edges in either direction (a self-loop counts twice); edges with an endpoint missing
    from the node file are left out. Registers node_stats(idx, degree, in_degree,
    out_degree) for the hub queries and returns the degree array for the histogram."""
    n = scalar(con, "SELECT count(*) FROM node")
    ends = con.execute("SELECT s, o FROM edge WHERE s IS NOT NULL AND o IS NOT NULL").fetchnumpy()
    s, o = np.asarray(ends["s"], dtype=np.int64), np.asarray(ends["o"], dtype=np.int64)
    out_degree = np.bincount(s, minlength=n).astype(np.int64)
    in_degree = np.bincount(o, minlength=n).astype(np.int64)
    degree = out_degree + in_degree
    del ends

    if n:
        adjacency = coo_matrix((np.ones(s.size, dtype=np.int8), (s, o)), shape=(n, n)).tocsr()
        count, labels = connected_components(adjacency, directed=True, connection="weak")
        del adjacency
        sizes = np.sort(np.bincount(labels))[::-1]
    else:
        count, sizes = 0, np.array([], dtype=np.int64)
    del s, o

    con.register("node_stats_arrow", pa.table({
        "idx": pa.array(np.arange(n, dtype=np.int32)),
        "degree": pa.array(degree), "in_degree": pa.array(in_degree), "out_degree": pa.array(out_degree),
    }))
    con.execute("CREATE TABLE node_stats AS SELECT * FROM node_stats_arrow")
    con.unregister("node_stats_arrow")

    by_category = rows(con, """SELECT n.primary_category AS category, count(*) AS nodes, min(d.degree) AS min,
            quantile_cont(d.degree, 0.5) AS median, quantile_cont(d.degree, 0.9) AS p90,
            quantile_cont(d.degree, 0.99) AS p99, max(d.degree) AS max,
            round(avg(d.degree), 3) AS mean, count(*) FILTER (WHERE d.degree = 0) AS isolated
        FROM node n JOIN node_stats d USING (idx) GROUP BY 1 ORDER BY nodes DESC, category""")
    isolated = int((degree == 0).sum())
    return {
        "components": int(count),
        "largestComponent": int(sizes[0]) if sizes.size else 0,
        "largestComponentShare": round(float(sizes[0]) / n, 4) if n else None,
        "secondComponent": int(sizes[1]) if sizes.size > 1 else 0,
        "componentsOfTwoOrMore": int((sizes >= 2).sum()),
        "isolatedNodes": isolated,
        "degree": degree_stats(degree),
        "degreeByCategory": by_category,
    }, degree


def hubs_section(con, top_n: int, per_category: int) -> dict:
    top = rows(con, f"""SELECT n.id, n.name, n.primary_category AS category, d.degree, d.in_degree, d.out_degree
        FROM node n JOIN node_stats d USING (idx) WHERE d.degree > 0
        ORDER BY d.degree DESC, n.id LIMIT {top_n}""")
    by_category = rows(con, f"""SELECT category, id, name, degree, in_degree, out_degree FROM (
            SELECT n.primary_category AS category, n.id, n.name, d.degree, d.in_degree, d.out_degree,
                   row_number() OVER (PARTITION BY n.primary_category ORDER BY d.degree DESC, n.id) AS r
            FROM node n JOIN node_stats d USING (idx) WHERE d.degree > 0)
        WHERE r <= {per_category} ORDER BY category, degree DESC, id""")
    return {"top": top, "byCategory": by_category}


def ontology_section(con, top_n: int) -> dict:
    """Term nodes (ids that aren't pennsieve:) by prefix, and how records use them through
    has_attribute edges (a record's coded value)."""
    rec = sql_string(RECORD_PREFIX)
    con.execute(f"""CREATE TABLE term_use AS
        SELECT e.o AS idx, count(*) AS edges, count(DISTINCT e.s) AS records
        FROM edge e JOIN node sn ON sn.idx = e.s JOIN node t ON t.idx = e.o
        WHERE e.predicate = {sql_string(HAS_ATTRIBUTE)} AND starts_with(sn.id, {rec}) AND NOT starts_with(t.id, {rec})
        GROUP BY e.o""")
    by_prefix = rows(con, f"""SELECT CASE WHEN contains(n.id, ':') THEN split_part(n.id, ':', 1) ELSE '(no prefix)' END AS prefix,
            count(*) AS terms, count(u.idx) AS used_terms, coalesce(sum(u.edges), 0)::BIGINT AS edges
        FROM node n LEFT JOIN term_use u USING (idx) WHERE NOT starts_with(n.id, {rec})
        GROUP BY 1 ORDER BY terms DESC, prefix""")
    linked = scalar(con, f"""SELECT count(DISTINCT e.s) FROM edge e JOIN node sn ON sn.idx = e.s JOIN node t ON t.idx = e.o
        WHERE e.predicate = {sql_string(HAS_ATTRIBUTE)} AND starts_with(sn.id, {rec}) AND NOT starts_with(t.id, {rec})""")
    records = scalar(con, f"SELECT count(*) FROM node WHERE starts_with(id, {rec})")
    top_terms = rows(con, f"""SELECT n.id, n.name, n.primary_category AS category, u.edges, u.records
        FROM term_use u JOIN node n USING (idx) ORDER BY u.edges DESC, n.id LIMIT {top_n}""")
    # Which record properties hold the coded values (pennsieve_property, Pennsieve graphs).
    by_property = rows(con, f"""SELECT prop AS property, count(*) AS edges FROM edge
        WHERE predicate = {sql_string(HAS_ATTRIBUTE)} AND prop IS NOT NULL
        GROUP BY 1 ORDER BY edges DESC, property LIMIT {MAX_LISTED}""")
    return {
        "termNodes": sum(r["terms"] for r in by_prefix),
        "usedTerms": scalar(con, "SELECT count(*) FROM term_use"),
        "hasAttributeEdges": scalar(con, f"SELECT count(*) FROM edge WHERE predicate = {sql_string(HAS_ATTRIBUTE)}"),
        "recordNodes": records,
        "recordsLinkedToTerms": linked,
        "recordsLinkedShare": round(linked / records, 4) if records else None,
        "termsByPrefix": by_prefix,
        "topTerms": top_terms,
        "codedProperties": by_property,
    }


def flag(code: str, level: str, count: int, message: str, examples: list | None = None) -> dict:
    return {"code": code, "level": level, "count": int(count), "message": message, "examples": examples or []}


def quality_section(con, graph_json: dict | None, size: dict) -> dict:
    """Data-quality checks. Every check's count is in `checks` (0 included); a check with
    a non-zero count is also a flag with examples. Warnings are likely errors in the
    graph; info flags are worth knowing but expected in some graphs."""
    checks, flags = {}, []

    def add(code: str, level: str, count, message: str, examples_query: str | None = None,
            examples: list | None = None) -> None:
        count = int(count or 0)
        checks[code] = count
        if count:
            if examples_query:
                examples = [r[0] for r in con.execute(examples_query).fetchall()]
            flags.append(flag(code, level, count, message, examples))

    lim = f"LIMIT {MAX_EXAMPLES}"
    dangling = con.execute("""SELECT count(*), count(*) FILTER (WHERE missing_subject IS NOT NULL),
        count(*) FILTER (WHERE missing_object IS NOT NULL) FROM edge
        WHERE missing_subject IS NOT NULL OR missing_object IS NOT NULL""").fetchone()
    add("dangling-edge", "warning", dangling[0],
        f"edges whose subject or object isn't in the node file ({dangling[1]} by subject, {dangling[2]} by object); "
        "they are left out of the schema, connectivity and hubs sections",
        f"""SELECT DISTINCT coalesce(missing_subject, missing_object) AS id FROM edge
            WHERE missing_subject IS NOT NULL OR missing_object IS NOT NULL ORDER BY id {lim}""")
    add("missing-node-id", "warning", scalar(con, "SELECT count(*) FROM nodes_src WHERE coalesce(id, '') = ''"),
        "node rows without an id; they are left out")
    add("duplicate-node-id", "warning", scalar(con, "SELECT count(*) FROM node WHERE copies > 1"),
        "node ids on more than one row; the first row is used",
        f"SELECT id FROM node WHERE copies > 1 ORDER BY copies DESC, id {lim}")
    # By the ids' 64-bit hashes (a false match is vanishingly unlikely); the examples
    # read the ids again.
    add("duplicate-edge-id", "warning",
        scalar(con, "SELECT count(*) FROM (SELECT id_hash FROM edge WHERE id_hash IS NOT NULL GROUP BY 1 HAVING count(*) > 1)"),
        "edge ids on more than one row",
        f"""SELECT id FROM edges_src WHERE coalesce(id, '') <> '' GROUP BY id HAVING count(*) > 1 ORDER BY id {lim}""")
    add("empty-name", "warning", scalar(con, "SELECT count(*) FROM node WHERE trim(coalesce(name, '')) = ''"),
        "nodes without a name", f"SELECT id FROM node WHERE trim(coalesce(name, '')) = '' ORDER BY id {lim}")
    add("missing-category", "warning", scalar(con, "SELECT count(*) FROM node WHERE len(coalesce(category, [])) = 0"),
        "nodes without a category", f"SELECT id FROM node WHERE len(coalesce(category, [])) = 0 ORDER BY id {lim}")
    known = " OR ".join(f"starts_with(c, {sql_string(p)})" for p in KNOWN_CATEGORY_PREFIXES)
    bare = rows(con, f"""SELECT c AS category, count(*) AS nodes FROM (SELECT unnest(category) AS c FROM node)
        WHERE NOT ({known}) GROUP BY 1 ORDER BY nodes DESC, category""")
    add("bare-prefix-category", "info", sum(r["nodes"] for r in bare),
        "nodes with a category that is neither biolink: nor pennsieve:, such as the bare prefix of an "
        "ontology Pennsieve has no Biolink class for",
        examples=[f"{r['category']} ({r['nodes']})" for r in bare[:MAX_EXAMPLES]])
    add("self-loop", "info", scalar(con, "SELECT count(*) FROM edge WHERE s = o"),
        "edges from a node to itself",
        f"SELECT DISTINCT n.id FROM edge e JOIN node n ON n.idx = e.s WHERE e.s = e.o ORDER BY n.id {lim}")

    if graph_json is None:
        add("no-graph-json", "info", 1, "no graph.json beside the graph files")
    else:
        counts = graph_json.get("counts") or {}
        add("dropped-edges", "info", counts.get("dropped_edges"),
            "relationships the export dropped because they lead to records outside the graph (graph.json)")
        stated = {k: counts.get(k) for k in ("nodes", "edges") if isinstance(counts.get(k), int)}
        # graph.json counts rows, so compare with the rows read (duplicates and all).
        read = {"nodes": scalar(con, "SELECT count(*) FROM nodes_src"), "edges": size["edges"]}
        differ = [f"{k}: graph.json {v}, files {read[k]}" for k, v in stated.items() if v != read[k]]
        add("counts-mismatch", "warning", len(differ), "graph.json counts differ from the files", examples=differ)
        if "error" in graph_json:
            add("unreadable-graph-json", "warning", 1, f"graph.json can't be read: {graph_json['error']}")

    return {"checks": checks, "flags": flags,
            "warnings": sum(1 for f in flags if f["level"] == "warning"),
            "infos": sum(1 for f in flags if f["level"] == "info")}


def graph_json_summary(graph_json: dict | None) -> dict | None:
    """What report.json repeats of graph.json: provenance and the models' naming."""
    if graph_json is None or "error" in graph_json:
        return graph_json
    return {
        "layoutVersion": graph_json.get("layout_version"),
        "biolink": graph_json.get("biolink"),
        "biolinkVersion": graph_json.get("biolink_version"),
        "generatedAt": graph_json.get("generated_at"),
        "source": graph_json.get("source"),
        "prefixes": sorted((graph_json.get("prefixes") or {}).keys()),
        "unresolvedPrefixes": graph_json.get("unresolved_prefixes") or [],
        "models": [{"name": m.get("name"), "version": m.get("version"), "nameProperty": m.get("name_property"),
                    "nameFrom": m.get("name_from") or []}
                   for m in graph_json.get("models") or [] if isinstance(m, dict)],
    }


# --- Outputs ---------------------------------------------------------------------------


def write_csv(path: str, header: list[str], table: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in table:
            w.writerow(["" if r.get(h) is None else r.get(h) for h in header])


def write_csvs(con, out: str, report: dict) -> list[str]:
    """The complete tables, beside the report."""
    s, c = report["size"], report["connectivity"]
    tables = {
        "nodes_by_category.csv": (["category", "nodes"], s["nodesByCategory"]),
        "edges_by_predicate.csv": (["predicate", "edges"], s["edgesByPredicate"]),
        "metagraph.csv": (["subject_category", "predicate", "object_category", "edges"], report["schema"]["triples"]),
        "degree_by_category.csv": (["category", "nodes", "min", "median", "p90", "p99", "max", "mean", "isolated"],
                                   c["degreeByCategory"]),
        "hubs.csv": (["scope", "id", "name", "category", "degree", "in_degree", "out_degree"],
                     [dict(r, scope="top") for r in report["hubs"]["top"]]
                     + [dict(r, scope="category") for r in report["hubs"]["byCategory"]]),
        # Every term node with how often records use it (report.json has the top ones).
        "terms.csv": (["id", "name", "category", "edges", "records"],
                      rows(con, f"""SELECT n.id, n.name, n.primary_category AS category,
                              coalesce(u.edges, 0) AS edges, coalesce(u.records, 0) AS records
                          FROM node n LEFT JOIN term_use u USING (idx)
                          WHERE NOT starts_with(n.id, {sql_string(RECORD_PREFIX)}) ORDER BY edges DESC, n.id""")),
    }
    for name, (header, table) in tables.items():
        write_csv(os.path.join(out, name), header, table)
    return sorted(tables)


def matplotlib_config() -> None:
    """Matplotlib needs a writable config folder for its font cache. The image builds the
    cache into MPLCONFIGDIR, which is read-only at run time, so a copy goes to the temp
    folder (building the cache again would take seconds)."""
    baked = os.environ.get("MPLCONFIGDIR")
    if baked and os.path.isdir(baked) and not os.access(baked, os.W_OK):
        copy = os.path.join(tempfile.gettempdir(), "kg-summary-matplotlib")
        shutil.copytree(baked, copy, dirs_exist_ok=True)
        os.environ["MPLCONFIGDIR"] = copy


def write_histogram(path: str, degree: np.ndarray) -> bool:
    """Nodes by degree on log-spaced bins (log-log), one series; False when no node
    has an edge."""
    connected = degree[degree > 0]
    if connected.size == 0:
        return False
    matplotlib_config()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = int(connected.max())
    edges = np.unique(np.round(np.logspace(0, math.log10(top + 1), 30)).astype(np.int64))
    if edges[-1] <= top:
        edges = np.append(edges, top + 1)
    counts, _ = np.histogram(connected, bins=edges)
    surface, ink, muted, series = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    keep = counts > 0
    ax.bar(edges[:-1][keep], counts[keep], width=(edges[1:] - edges[:-1])[keep] * 0.86, align="edge",
           color=series, edgecolor=surface, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (edges in either direction)", color=muted)
    ax.set_ylabel("nodes", color=muted)
    ax.set_title(f"Degree distribution ({connected.size:,} nodes with edges)", color=ink, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return True


def md_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = round(value, 1)
        return f"{int(value):,}" if value.is_integer() else f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(columns: list[tuple[str, str]], table: list[dict], more: str = "") -> list[str]:
    """A markdown table of at most MD_ROWS rows; `more` names where the rest is."""
    if not table:
        return ["(none)", ""]
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "---|" * len(columns)]
    for r in table[:MD_ROWS]:
        lines.append("| " + " | ".join(md_cell(r.get(key)) for key, _ in columns) + " |")
    if len(table) > MD_ROWS:
        lines.append("")
        lines.append(f"{len(table) - MD_ROWS:,} more" + (f" in {more}" if more else "") + ".")
    return lines + [""]


def markdown(report: dict) -> str:
    app = report["app"]
    lines = ["# Knowledge graph summary", ""]
    if app["error"]:
        return "\n".join(lines + [f"**The graph couldn't be read:** {app['error']}", ""])
    s, sc, c, h, o, q = (report[k] for k in ("size", "schema", "connectivity", "hubs", "ontology", "quality"))
    g = report.get("graphJson") or {}
    lines += [f"Read from the {report['input']['format']} files in `{report['input']['graphDir']}`"
              + (f"; graph.json made {g['generatedAt']}" if g.get("generatedAt") else "") + ".", ""]

    lines += ["## Size", "",
              "| | |", "|---|---|",
              f"| Nodes | {s['nodes']:,} ({s['recordNodes']:,} records, {s['termNodes']:,} ontology terms) |",
              f"| Edges | {s['edges']:,} |",
              f"| Categories / predicates | {s['categories']:,} / {s['predicates']:,} |",
              f"| Biolink (graph.json) | {s['biolink'] or 'not stated'}"
              + (f", Biolink {s['biolinkVersion']}" if s.get("biolinkVersion") else "") + " |", ""]
    lines += ["### Nodes by category", ""] + md_table([("category", "Category"), ("nodes", "Nodes")],
                                                      s["nodesByCategory"], "nodes_by_category.csv")
    lines += ["### Edges by predicate", ""] + md_table([("predicate", "Predicate"), ("edges", "Edges")],
                                                       s["edgesByPredicate"], "edges_by_predicate.csv")

    lines += ["## Schema", "", f"{sc['tripleCount']:,} category → predicate → category combinations.", ""]
    lines += md_table([("subject_category", "Subject category"), ("predicate", "Predicate"),
                       ("object_category", "Object category"), ("edges", "Edges")], sc["triples"], "metagraph.csv")

    d = c["degree"]
    share = f" ({c['largestComponentShare']:.1%} of nodes)" if c["largestComponentShare"] is not None else ""
    lines += ["## Connectivity", "",
              f"{c['components']:,} weakly connected components; the largest has {c['largestComponent']:,} nodes{share}, "
              f"the next {c['secondComponent']:,}. Nodes without edges: {c['isolatedNodes']:,}.", "",
              "| Degree | Min | Median | p90 | p99 | Max | Mean |", "|---|---|---|---|---|---|---|",
              "| All nodes | " + " | ".join(md_cell(d[k]) for k in ("min", "median", "p90", "p99", "max", "mean")) + " |", ""]
    lines += md_table([("category", "Category"), ("nodes", "Nodes"), ("min", "Min"), ("median", "Median"),
                       ("p90", "p90"), ("p99", "p99"), ("max", "Max"), ("isolated", "Isolated")],
                      c["degreeByCategory"], "degree_by_category.csv")
    if report["outputs"].get("histogram"):
        lines += [f"![Degree distribution]({HISTOGRAM_PNG})", ""]

    lines += ["## Hubs", "", "The nodes with the most edges.", ""]
    lines += md_table([("name", "Name"), ("category", "Category"), ("degree", "Degree"), ("in_degree", "In"),
                       ("out_degree", "Out"), ("id", "Id")], h["top"], "hubs.csv")
    lines += ["### Top hubs per category", ""]
    by_cat: dict[str, list[str]] = {}
    for r in h["byCategory"]:
        by_cat.setdefault(r["category"], []).append(f"{md_cell(r['name'] or r['id'])} ({r['degree']:,})")
    if by_cat:
        lines += ["| Category | Hubs (degree) |", "|---|---|"]
        lines += [f"| {md_cell(k)} | {', '.join(v)} |" for k, v in list(by_cat.items())[:MD_ROWS]]
        if len(by_cat) > MD_ROWS:
            lines += ["", f"{len(by_cat) - MD_ROWS:,} more categories in hubs.csv."]
    else:
        lines.append("(none)")
    lines.append("")

    lines += ["## Ontology use", ""]
    if o["recordNodes"]:
        lines += [f"{o['recordsLinkedToTerms']:,} of {o['recordNodes']:,} records ({o['recordsLinkedShare']:.1%}) link to "
                  f"ontology terms through {o['hasAttributeEdges']:,} `{HAS_ATTRIBUTE}` edges; "
                  f"{o['usedTerms']:,} of {o['termNodes']:,} term nodes are used.", ""]
    lines += ["### Term nodes by prefix", ""] + md_table(
        [("prefix", "Prefix"), ("terms", "Terms"), ("used_terms", "Used"), ("edges", "Edges")], o["termsByPrefix"])
    lines += ["### Most-used terms", ""] + md_table(
        [("name", "Name"), ("id", "Term"), ("category", "Category"), ("edges", "Edges"), ("records", "Records")],
        o["topTerms"], "terms.csv")
    if o["codedProperties"]:
        lines += ["### Coded properties", ""] + md_table([("property", "Record property"), ("edges", "Edges")],
                                                         o["codedProperties"])

    lines += ["## Data quality", ""]
    if not q["flags"]:
        lines += ["No flags: every check passed.", ""]
    else:
        lines += ["| Level | Check | Count | What | Examples |", "|---|---|---|---|---|"]
        for f in q["flags"]:
            examples = ", ".join(f"`{e}`" for e in f["examples"][:3])
            lines.append(f"| {f['level']} | {f['code']} | {f['count']:,} | {md_cell(f['message'])} | {md_cell(examples)} |")
        lines.append("")
    return "\n".join(lines)


# --- Main ------------------------------------------------------------------------------


def connect() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB that spills to the temp folder, kept within the container's
    memory limit (cgroup v2) with room for the NumPy arrays beside it."""
    con = duckdb.connect()
    con.execute(f"SET temp_directory = {sql_string(os.path.join(tempfile.gettempdir(), 'kg-summary-duckdb'))}")
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="ascii") as f:
            limit = f.read().strip()
        if limit.isdigit():
            con.execute(f"SET memory_limit = '{int(limit) * 6 // 10 >> 20}MB'")
    except OSError:
        pass
    con.execute("SET preserve_insertion_order = false")
    return con


def summarize(graph_dir: str, out: str | None, top_n: int, per_category: int) -> dict:
    files = find_files(graph_dir)
    graph_json = read_graph_json(graph_dir)
    con = connect()
    columns = open_graph(con, files)
    t = time.monotonic()
    load(con)
    log(f"loaded the graph from {files['format']} in {time.monotonic() - t:.1f}s")
    report = {"input": {"graphDir": graph_dir, "format": files["format"],
                        "files": [os.path.basename(files["nodes"]), os.path.basename(files["edges"])],
                        "columns": columns}}
    report["graphJson"] = graph_json_summary(graph_json)
    report["size"] = size_section(con, graph_json)
    report["schema"] = schema_section(con)
    report["connectivity"], degree = connectivity_section(con)
    report["hubs"] = hubs_section(con, top_n, per_category)
    report["ontology"] = ontology_section(con, top_n)
    report["quality"] = quality_section(con, graph_json, report["size"])
    report["outputs"] = {}
    if out:
        os.makedirs(out, exist_ok=True)
        report["outputs"]["csv"] = write_csvs(con, out, report)
        report["outputs"]["histogram"] = HISTOGRAM_PNG if write_histogram(os.path.join(out, HISTOGRAM_PNG), degree) else None
    con.close()
    # report.json lists at most MAX_LISTED rows per table; the CSVs have them all.
    for section, key in (("size", "nodesByCategory"), ("size", "edgesByPredicate"), ("schema", "triples"),
                         ("connectivity", "degreeByCategory"), ("hubs", "byCategory"), ("ontology", "termsByPrefix")):
        report[section][key] = report[section][key][:MAX_LISTED]
    return report


def main() -> int:
    started = time.monotonic()
    top_n = param_int("TOP_N", 1, 1000)
    per_category = param_int("TOP_PER_CATEGORY", 0, 100)
    root = os.environ.get("INPUT_DIR") or "."
    out = os.environ.get("OUTPUT_DIR")
    graph_dir = os.path.join(root, GRAPH_PORT)
    log(f"kg-summary, graph in {graph_dir}, top {top_n} hubs and terms, {per_category} hubs per category")

    report = {"app": {"name": "kg-summary", "spec": SPEC, "startedAt": now(), "topN": top_n,
                      "topPerCategory": per_category, "error": None}}
    try:
        report.update(summarize(graph_dir, out, top_n, per_category))
    except GraphError as e:
        report["app"]["error"] = str(e)
        log(f"error: {e}")
    except Exception as e:  # noqa: BLE001 - always write a report, and say the app failed
        report["app"]["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        report["app"]["traceback"] = traceback.format_exc()[-4000:]
        log(f"error: {type(e).__name__}: {e}")
    exit_code = 1 if report["app"]["error"] else 0
    report["app"]["finishedAt"] = now()
    report["app"]["seconds"] = round(time.monotonic() - started, 2)
    report["app"]["exitCode"] = exit_code

    text = markdown(report)
    print(text, flush=True)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, REPORT_JSON), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        with open(os.path.join(out, REPORT_MD), "w", encoding="utf-8") as f:
            f.write(text)
        log(f"wrote the report to {out}")
    else:
        log("OUTPUT_DIR is not set; the report is in the log only")

    size, quality = report.get("size") or {}, report.get("quality") or {}
    summary = {
        "run": os.environ.get("EXECUTION_RUN_ID"),
        "nodes": size.get("nodes", 0),
        "edges": size.get("edges", 0),
        "components": (report.get("connectivity") or {}).get("components", 0),
        "warnings": quality.get("warnings", 0),
        "infos": quality.get("infos", 0),
        "error": report["app"]["error"],
        "seconds": report["app"]["seconds"],
        "exitCode": exit_code,
    }
    print("KG_SUMMARY " + json.dumps(summary), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
