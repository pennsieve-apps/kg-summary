"""Tests for kg_summary.py. Run with: make test (or python -m pytest tests).

The fixture (tests/fixtures/graph, written by tests/fixtures/make_fixture.py) has 8 record
nodes, 4 term nodes and 12 edges; the samples' model is bound to biolink:MaterialSample.
It has one dangling edge, one duplicate node id (its second row differs), one empty name,
one bare-prefix category (MESH), one self-loop and 2 dropped edges.
"""

import gzip
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "graph")
sys.path.insert(0, ROOT)

import kg_summary  # noqa: E402

S1 = "pennsieve:0b9f6c1e-1d2a-4c3b-8e4f-5a6b7c8d9e01"
X1 = "pennsieve:3e2c9f41-405d-4f6e-b172-8d9e0f1a2b34"
X2 = "pennsieve:4f3da052-516e-4071-c283-9e0f1a2b3c45"
P2 = "pennsieve:7260d385-8491-43a4-f5b6-c13c4d5e6f78"
MISSING = "pennsieve:83710e96-95a2-44b5-a6c7-d24d5e6f7089"


def copy_graph(tmp_path, keep=None):
    """A copy of the fixture graph in tmp_path/input/graph, with only the files in `keep`."""
    graph = tmp_path / "input" / "graph"
    graph.mkdir(parents=True)
    for name in os.listdir(FIXTURE):
        if keep is None or name in keep:
            shutil.copy(os.path.join(FIXTURE, name), graph / name)
    return graph


def summarize(graph, out, top_n=25, per_category=5):
    return kg_summary.summarize(str(graph), str(out), top_n, per_category)


def flags(report):
    return {f["code"]: f for f in report["quality"]["flags"]}


@pytest.fixture(scope="module")
def parquet_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parquet")
    out = tmp / "output"
    return summarize(copy_graph(tmp), out), out


@pytest.fixture(scope="module")
def report(parquet_run):
    return parquet_run[0]


def test_reads_parquet_when_present(report):
    assert report["input"]["format"] == "parquet"
    assert report["input"]["files"] == ["nodes.parquet", "edges.parquet"]


def test_size(report):
    s = report["size"]
    assert (s["nodes"], s["edges"], s["recordNodes"], s["termNodes"]) == (12, 12, 8, 4)
    assert s["biolink"] == "partial" and s["biolinkVersion"] == "4.2.2"
    by_category = {r["category"]: r["nodes"] for r in s["nodesByCategory"]}
    assert by_category == {"pennsieve:subject": 3, "pennsieve:sample": 3, "pennsieve:protocol": 2,
                           "biolink:AnatomicalEntity": 2, "biolink:OrganismTaxon": 1, "MESH": 1}
    # Records count under their model, not the Biolink class bound to it; the duplicate
    # row's category (pennsieve:other) isn't used.
    by_predicate = {r["predicate"]: r["edges"] for r in s["edgesByPredicate"]}
    assert by_predicate == {"biolink:has_attribute": 6, "pennsieve:has_sample": 4,
                            "pennsieve:derived_from": 1, "pennsieve:follows": 1}


def test_biolink_bound_records_count_under_their_model(report):
    """Samples are [biolink:MaterialSample, pennsieve:sample]; every section uses pennsieve:sample."""
    assert "biolink:MaterialSample" not in json.dumps({k: report[k] for k in ("size", "schema", "connectivity", "hubs")})
    assert any(r["category"] == "pennsieve:sample" for r in report["connectivity"]["degreeByCategory"])


def test_duplicate_node_uses_first_row(report):
    x2 = next(r for r in report["hubs"]["top"] if r["id"] == X2)
    assert (x2["name"], x2["category"]) == ("brain slice 2", "pennsieve:sample")


def test_schema_metagraph(report):
    triples = {(t["subject_category"], t["predicate"], t["object_category"]): t["edges"]
               for t in report["schema"]["triples"]}
    assert triples[("pennsieve:subject", "pennsieve:has_sample", "pennsieve:sample")] == 3
    assert triples[("pennsieve:sample", "biolink:has_attribute", "biolink:AnatomicalEntity")] == 3
    assert triples[("pennsieve:sample", "biolink:has_attribute", "MESH")] == 1
    assert triples[("pennsieve:sample", "pennsieve:derived_from", "pennsieve:sample")] == 1
    # The dangling edge (sample 2 -> a missing record) is left out.
    assert sum(triples.values()) == 11


def test_connectivity(report):
    c = report["connectivity"]
    assert c["components"] == 3
    assert c["largestComponent"] == 9 and c["secondComponent"] == 2
    assert c["isolatedNodes"] == 1
    d = c["degree"]
    assert (d["nodes"], d["min"], d["max"]) == (12, 0, 4)
    assert d["median"] == 2
    subjects = next(r for r in c["degreeByCategory"] if r["category"] == "pennsieve:subject")
    assert (subjects["nodes"], subjects["min"], subjects["max"], subjects["isolated"]) == (3, 0, 3, 1)


def test_hubs(report):
    top = report["hubs"]["top"]
    # Sample 1: two edges out (a term, itself), two in (subject 1, itself).
    assert top[0]["id"] == X1 and top[0]["name"] == "brain slice 1"
    assert (top[0]["degree"], top[0]["in_degree"], top[0]["out_degree"]) == (4, 2, 2)
    assert top[1]["id"] == S1 and top[1]["degree"] == 3
    # The isolated subject isn't a hub.
    assert len(top) == 11
    samples = [r["name"] for r in report["hubs"]["byCategory"] if r["category"] == "pennsieve:sample"]
    assert samples == ["brain slice 1", "liver sample", "brain slice 2"]


def test_top_n_limits(tmp_path):
    r = summarize(copy_graph(tmp_path), tmp_path / "out", top_n=2, per_category=1)
    assert len(r["hubs"]["top"]) == 2 and len(r["ontology"]["topTerms"]) == 2
    assert all(sum(1 for h in r["hubs"]["byCategory"] if h["category"] == c) == 1
               for c in {h["category"] for h in r["hubs"]["byCategory"]})


def test_ontology(report):
    o = report["ontology"]
    assert (o["termNodes"], o["usedTerms"], o["hasAttributeEdges"]) == (4, 4, 6)
    assert (o["recordNodes"], o["recordsLinkedToTerms"]) == (8, 5)
    prefixes = {r["prefix"]: (r["terms"], r["used_terms"], r["edges"]) for r in o["termsByPrefix"]}
    assert prefixes == {"UBERON": (2, 2, 3), "NCBITaxon": (1, 1, 2), "MESH": (1, 1, 1)}
    assert [(t["id"], t["name"], t["records"]) for t in o["topTerms"][:2]] == [
        ("NCBITaxon:10090", "Mus musculus", 2), ("UBERON:0000955", "brain", 2)]
    assert {r["property"]: r["edges"] for r in o["codedProperties"]} == {"anatomy": 3, "species": 2, "mesh_term": 1}


def test_quality_flags(report):
    f = flags(report)
    assert f["dangling-edge"]["count"] == 1 and f["dangling-edge"]["examples"] == [MISSING]
    assert f["duplicate-node-id"]["count"] == 1 and f["duplicate-node-id"]["examples"] == [X2]
    assert f["empty-name"]["count"] == 1 and f["empty-name"]["examples"] == [P2]
    assert f["bare-prefix-category"]["count"] == 1 and f["bare-prefix-category"]["examples"] == ["MESH (1)"]
    assert f["self-loop"]["count"] == 1 and f["self-loop"]["examples"] == [X1]
    assert f["dropped-edges"]["count"] == 2
    assert {c: f[c]["level"] for c in f} == {
        "dangling-edge": "warning", "duplicate-node-id": "warning", "empty-name": "warning",
        "bare-prefix-category": "info", "self-loop": "info", "dropped-edges": "info"}
    checks = report["quality"]["checks"]
    assert checks["duplicate-edge-id"] == 0 and checks["missing-category"] == 0 and checks["counts-mismatch"] == 0
    assert (report["quality"]["warnings"], report["quality"]["infos"]) == (3, 3)


def test_outputs(parquet_run):
    _, out = parquet_run
    names = set(os.listdir(out))
    assert {"nodes_by_category.csv", "edges_by_predicate.csv", "metagraph.csv", "degree_by_category.csv",
            "hubs.csv", "terms.csv", "degree_histogram.png"} <= names
    with open(out / "hubs.csv") as f:
        assert f.readline().strip() == "scope,id,name,category,degree,in_degree,out_degree"
    with open(out / "terms.csv") as f:
        assert len(f.readlines()) == 5
    with open(out / "degree_histogram.png", "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_tsv_only(tmp_path, report):
    """Without the Parquet files the TSV pair is read, with the same results."""
    tsv = summarize(copy_graph(tmp_path, {"nodes.tsv.gz", "edges.tsv.gz", "graph.json"}), tmp_path / "out")
    assert tsv["input"]["format"] == "tsv"
    for section in ("size", "schema", "connectivity", "hubs", "ontology", "quality"):
        assert tsv[section] == report[section], section


def test_foreign_kgx(tmp_path):
    """A KGX graph from elsewhere: plain TSV, core columns only, no graph.json."""
    graph = tmp_path / "input" / "graph"
    graph.mkdir(parents=True)
    (graph / "nodes.tsv").write_text("id\tcategory\nMONDO:1\tbiolink:Disease|biolink:NamedThing\nHP:1\tbiolink:PhenotypicFeature\n")
    (graph / "edges.tsv").write_text("subject\tpredicate\tobject\nMONDO:1\tbiolink:has_phenotype\tHP:1\n")
    r = summarize(graph, tmp_path / "out")
    assert (r["size"]["nodes"], r["size"]["edges"], r["size"]["recordNodes"]) == (2, 1, 0)
    # A term with several categories counts under its first.
    assert {c["category"] for c in r["size"]["nodesByCategory"]} == {"biolink:Disease", "biolink:PhenotypicFeature"}
    assert r["schema"]["triples"][0]["subject_category"] == "biolink:Disease"
    f = flags(r)
    assert f["empty-name"]["count"] == 2 and "no-graph-json" in f
    assert r["size"]["biolink"] is None


def test_duplicate_edge_id_and_counts_mismatch(tmp_path):
    graph = copy_graph(tmp_path, {"nodes.tsv.gz", "edges.tsv.gz", "graph.json"})
    with gzip.open(graph / "edges.tsv.gz", "rt") as f:
        lines = f.readlines()
    with gzip.open(graph / "edges.tsv.gz", "wt") as f:
        f.writelines(lines + [lines[1]])
    r = summarize(graph, tmp_path / "out")
    f = flags(r)
    assert f["duplicate-edge-id"]["count"] == 1
    assert f["duplicate-edge-id"]["examples"] == [lines[1].split("\t")[0]]
    assert f["counts-mismatch"]["examples"] == ["edges: graph.json 12, files 13"]


def test_missing_column(tmp_path):
    graph = tmp_path / "input" / "graph"
    graph.mkdir(parents=True)
    (graph / "nodes.tsv").write_text("id\tcategory\nA:1\tbiolink:Gene\n")
    (graph / "edges.tsv").write_text("subject\tpredicate\nA:1\tbiolink:related_to\n")
    with pytest.raises(kg_summary.GraphError, match="no object column"):
        summarize(graph, tmp_path / "out")


def run_main(tmp_path, input_dir, **env):
    out = tmp_path / "output"
    result = subprocess.run([sys.executable, os.path.join(ROOT, "kg_summary.py")], capture_output=True, text=True,
                            env=dict(os.environ, INPUT_DIR=str(input_dir), OUTPUT_DIR=str(out), **env))
    summary = json.loads(result.stdout.strip().splitlines()[-1].removeprefix("KG_SUMMARY "))
    with open(out / "report.json") as f:
        return result, summary, json.load(f), out


def test_main(tmp_path):
    graph = copy_graph(tmp_path)
    result, summary, report, out = run_main(tmp_path, graph.parent, TOP_N="3", TOP_PER_CATEGORY="x")
    assert result.returncode == 0
    assert (summary["nodes"], summary["edges"], summary["components"], summary["warnings"]) == (12, 12, 3, 3)
    assert report["app"]["topN"] == 3 and report["app"]["topPerCategory"] == 5
    md = (out / "report.md").read_text()
    for heading in ("## Size", "## Schema", "## Connectivity", "## Hubs", "## Ontology use", "## Data quality"):
        assert heading in md
    assert "| warning | dangling-edge | 1 |" in md


def test_main_without_graph(tmp_path):
    (tmp_path / "input").mkdir()
    result, summary, report, _ = run_main(tmp_path, tmp_path / "input")
    assert result.returncode == 1 and summary["exitCode"] == 1
    assert "no graph input" in report["app"]["error"]


def write_graph(tmp_path, nodes, edges, graph_json=None):
    """A plain-TSV graph in tmp_path/input/graph."""
    graph = tmp_path / "input" / "graph"
    graph.mkdir(parents=True)
    (graph / "nodes.tsv").write_text(nodes)
    (graph / "edges.tsv").write_text(edges)
    if graph_json is not None:
        (graph / "graph.json").write_text(graph_json)
    return graph


def test_empty_graph(tmp_path):
    graph = write_graph(tmp_path, "id\tcategory\tname\n", "id\tsubject\tpredicate\tobject\n")
    r = summarize(graph, tmp_path / "out")
    assert (r["size"]["nodes"], r["size"]["edges"], r["connectivity"]["components"]) == (0, 0, 0)
    assert r["connectivity"]["degree"]["max"] is None and r["hubs"]["top"] == []
    assert r["outputs"]["histogram"] is None
    assert "## Data quality" in kg_summary.markdown(dict(r, app={"error": None}))


def test_nodes_only_graph(tmp_path):
    graph = write_graph(tmp_path, "id\tcategory\tname\nA:1\tbiolink:Gene\ta\nA:2\tbiolink:Gene\tb\n",
                        "id\tsubject\tpredicate\tobject\n")
    r = summarize(graph, tmp_path / "out")
    c = r["connectivity"]
    assert (c["components"], c["largestComponent"], c["isolatedNodes"]) == (2, 1, 2)
    assert r["schema"]["triples"] == [] and r["hubs"]["top"] == []


def test_biolink_first_record_category(tmp_path):
    """A record node's pennsieve: category is its primary one even when a Biolink class is first."""
    graph = write_graph(tmp_path,
                        "id\tcategory\tname\n"
                        "pennsieve:1\tbiolink:MaterialSample|pennsieve:sample\ts1\n"
                        "pennsieve:2\tbiolink:MaterialSample|pennsieve:specimen\ts2\n",
                        "id\tsubject\tpredicate\tobject\ne1\tpennsieve:1\tpennsieve:from\tpennsieve:2\n")
    r = summarize(graph, tmp_path / "out")
    assert {c["category"] for c in r["size"]["nodesByCategory"]} == {"pennsieve:sample", "pennsieve:specimen"}
    assert [(t["subject_category"], t["object_category"]) for t in r["schema"]["triples"]] == [
        ("pennsieve:sample", "pennsieve:specimen")]
    assert {h["category"] for h in r["hubs"]["byCategory"]} == {"pennsieve:sample", "pennsieve:specimen"}


def test_malformed_graph_json(tmp_path):
    bad = json.dumps({"biolink": ["full"], "prefixes": ["a"], "models": [{"name": 3}, {"name": "m", "version": 1}],
                      "counts": {"nodes": 1, "edges": [], "dropped_edges": "x"}})
    graph = write_graph(tmp_path, "id\tcategory\tname\nA:1\tbiolink:Gene\ta\n", "subject\tpredicate\tobject\n", bad)
    r = summarize(graph, tmp_path / "out")
    f = flags(r)
    assert f["unreadable-graph-json"]["examples"] == [
        "biolink is not a string", "prefixes is not an object", "counts.edges is not a count",
        "counts.dropped_edges is not a count", "models[0] is malformed"]
    assert "dropped-edges" not in f and "counts-mismatch" not in f
    assert r["size"]["biolink"] is None and [m["name"] for m in r["graphJson"]["models"]] == ["m"]

    (graph / "graph.json").write_text(json.dumps({"counts": [1, 2]}))
    assert flags(summarize(graph, tmp_path / "out2"))["unreadable-graph-json"]["examples"] == ["counts is not an object"]
    (graph / "graph.json").write_text("{not json")
    assert flags(summarize(graph, tmp_path / "out3"))["unreadable-graph-json"]["count"] == 1


EVIL = "![x](https://evil.example/p.png) <script>alert(1)</script> `code` [link](https://evil.example)"


def test_markdown_is_inert(tmp_path):
    graph = write_graph(tmp_path, f"id\tcategory\tname\nA:1\tbiolink:Gene\t{EVIL}\nA:2\tbiolink:Gene\tb\n",
                        "subject\tpredicate\tobject\nA:1\tbiolink:related_to\tA:2\n",
                        json.dumps({"generated_at": "<img src=x onerror=alert(1)>", "biolink": "[full](javascript:x)"}))
    r = summarize(graph, tmp_path / "out")
    md = kg_summary.markdown(dict(r, app={"error": None}))
    # The report's own image (the histogram) is the one markdown image allowed.
    md = md.replace(f"![Degree distribution]({kg_summary.HISTOGRAM_PNG})", "")
    for raw in ("![", "<script", "<img", "`code`"):
        assert raw not in md, raw
    assert not re.search(r"(?<!\\)\]\(", md)  # no unescaped link
    assert "&lt;script&gt;" in md and "\\[full\\]" in md


def test_md_cell():
    assert kg_summary.md_cell("a|b\r\nc") == "a\\|b  c"
    assert kg_summary.md_cell("<b>&") == "&lt;b&gt;&amp;"
    assert kg_summary.md_cell("\\[") == "\\\\\\["
    assert kg_summary.md_cell(2.0) == "2" and kg_summary.md_cell(3.25) == "3.2" and kg_summary.md_cell(1234) == "1,234"


def test_csv_formulas_are_inert(tmp_path):
    graph = write_graph(tmp_path, "id\tcategory\tname\n=HYPERLINK(1)\t+cat\t@SUM(A1)\nA:2\tbiolink:Gene\t-2\n",
                        "subject\tpredicate\tobject\n=HYPERLINK(1)\tbiolink:related_to\tA:2\n")
    out = tmp_path / "out"
    summarize(graph, out)
    text = "".join((out / name).read_text() for name in os.listdir(out) if name.endswith(".csv"))
    assert "'=HYPERLINK(1)" in text and "'+cat" in text and "'@SUM(A1)" in text and "'-2" in text
    for line in text.splitlines():
        for cell in line.split(","):
            assert not cell.startswith(("=", "+", "-", "@")), line


def test_many_parallel_edges_still_connect(tmp_path):
    """256 parallel edges sum to 0 in an int8 matrix; the adjacency must not lose them."""
    edges = "subject\tpredicate\tobject\n" + "A:1\tbiolink:related_to\tA:2\n" * 256
    graph = write_graph(tmp_path, "id\tcategory\tname\nA:1\tbiolink:Gene\ta\nA:2\tbiolink:Gene\tb\n", edges)
    c = summarize(graph, tmp_path / "out")["connectivity"]
    assert (c["components"], c["largestComponent"], c["degree"]["max"]) == (1, 2, 256)
