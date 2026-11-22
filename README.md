# KG Summary

A Pennsieve app that summarises a dataset's knowledge graph: the dataset's metadata
records and the ontology terms they use, as a [KGX](https://github.com/biolink/kgx)
graph (Knowledge Graph Exchange, with [Biolink Model](https://biolink.github.io/biolink-model/)
categories and predicates where they are known). It reports how big the graph is, how its
categories connect, how well connected it is, its hubs, which ontologies the records use,
and data-quality problems, as a short readable report with tables and a degree histogram.
It is built by Pennsieve on open-source libraries: [DuckDB](https://duckdb.org) reads and
aggregates the graph files, [SciPy](https://scipy.org) and [NumPy](https://numpy.org)
compute connected components and degrees, [Apache Arrow](https://arrow.apache.org) moves
data between them, and [Matplotlib](https://matplotlib.org) draws the histogram (see
[Acknowledgements](#acknowledgements)).

## Input

- **`graph`** (`conformsTo: kgx`, `placement: subfolder`, required): a dataset's graph, in
  `INPUT_DIR/graph`, laid out as the [kgx 1.0 structure](https://schemas.pennsieve.io/structures/kgx/v1.0)
  says:

```
graph/
  nodes.tsv.gz         KGX TSV, gzipped (canonical)
  edges.tsv.gz
  nodes.parquet        the same rows, typed
  edges.parquet
  graph.json           provenance, prefixes, models, counts
  graph_stats.yaml     kgx graph-summary form (not read)
```

The app reads `nodes.parquet` and `edges.parquet` when both are there, else the TSV pair
(`.tsv.gz`, or plain `.tsv`; read unquoted, as KGX TSV is written). It needs only the core
columns (`id` for nodes; `subject`, `predicate`, `object` for edges) and uses `category`,
`name`, the edge `id` and `pennsieve_property` when present, so a KGX graph from
elsewhere works too. `graph.json` is optional.

## What it reports

**Size.** Nodes and edges; nodes by category (a node with several categories counts under
each) and edges by predicate; record nodes (`pennsieve:<record uuid>`) and ontology term
nodes (any other id); the Biolink level and version from `graph.json`.

**Schema.** The metagraph: for each subject category, predicate and object category, how
many edges. Wherever a node needs one category (here, degrees and hubs), its first one is
used.

**Connectivity.** Weakly connected components (count, the largest and the next, nodes
without edges) and the degree distribution (min, median, p90, p99, max, mean), overall and
per category. Degree counts edges in either direction; a self-loop counts twice.

**Hubs.** The `TOP_N` nodes with the highest degree, with name, category and in/out
degree, and the `TOP_PER_CATEGORY` highest in each category.

**Ontology use.** Term nodes by prefix (how many, how many are used), how many records
link to terms through `biolink:has_attribute` edges (a record's coded value), the `TOP_N`
most-used terms with their names, and which record properties hold coded values
(`pennsieve_property`).

**Data quality.** Each check is counted in `quality.checks`; a non-zero one is a flag with
up to 10 examples. Warnings are likely errors; infos are worth knowing but expected in
some graphs.

| Check | Level | What |
|---|---|---|
| `dangling-edge` | warning | an edge's subject or object isn't in the node file (such edges are left out of the schema, connectivity and hubs) |
| `duplicate-node-id` / `duplicate-edge-id` | warning | an id on more than one row (the first node row is used) |
| `missing-node-id` | warning | a node row without an id (left out) |
| `empty-name` | warning | a node without a name |
| `missing-category` | warning | a node without a category |
| `counts-mismatch` | warning | `graph.json`'s node or edge count differs from the files' rows |
| `unreadable-graph-json` | warning | `graph.json` isn't valid JSON |
| `bare-prefix-category` | info | a category that is neither `biolink:` nor `pennsieve:`, such as `MESH`: the bare prefix the export gives a term of an ontology without a Biolink class |
| `self-loop` | info | an edge from a node to itself |
| `dropped-edges` | info | relationships the export dropped because they lead to records outside the graph (`graph.json`) |
| `no-graph-json` | info | no `graph.json` beside the files |

## Output

All in `OUTPUT_DIR` (the `report` output):

- **`report.md`**: the summary, with short tables (15 rows; the CSVs have the rest).
  Also printed to the log.
- **`report.json`**: machine-readable, with the keys `app`, `input`, `graphJson`,
  `size`, `schema`, `connectivity`, `hubs`, `ontology`, `quality` and `outputs`.
- **CSVs**: `nodes_by_category.csv`, `edges_by_predicate.csv`, `metagraph.csv`,
  `degree_by_category.csv`, `hubs.csv` (`scope` top or category) and `terms.csv` (every
  term node with how many edges and records use it).
- **`degree_histogram.png`**: nodes by degree, log-log.

The last log line is `KG_SUMMARY {...}`: nodes, edges, components, warning and info flags,
seconds and the exit code. The step exits 1 when the graph can't be read (no `graph/`
folder, no node and edge files, a missing core column); `report.json` then says why in
`app.error`. Data-quality flags never fail the step.

## Parameters

| Parameter | Default | What |
|---|---|---|
| `TOP_N` | `25` | how many hubs and most-used terms to list (1 to 1000) |
| `TOP_PER_CATEGORY` | `5` | how many hubs to list per category (0 to 100) |

## Scale

DuckDB reads the files once into compact tables (node ids become integer indexes, edges
keep only their endpoints' indexes, predicate, property and a hash of their id); components
and degrees are computed on integer arrays with SciPy and NumPy. No Python object is made
per edge, and DuckDB keeps within 60% of the container's memory, spilling to `TMPDIR`
beyond it.

Measured with `make bench` on a synthetic graph the size of Hetionet on Pennsieve (91,895
nodes, 2,295,061 edges, hubs of up to ~25,000 edges), in the image, at 1 vCPU / 2 GiB
(app.yml's runtime):

| Input | Time | Peak memory (cgroup) |
|---|---|---|
| Parquet (98 MB) | 1.7-1.9 s | 790-820 MiB |
| TSV, gzipped (80 MB) | 3 s | 500 MiB |

It also runs in 1 GiB; 2 GiB leaves room for graphs a few times larger.

## Using it on Pennsieve

Add KG Summary as a step (or open it in "Try this app") and link a dataset's graph to the
`graph` input: the run puts the graph of the dataset view's records in `graph/`. The
dataset needs metadata records and a view made after graphs were added. Read
`report.md` in the step's output.

## Running locally

```sh
mkdir -p /tmp/kgs/input /tmp/kgs/output
cp -R tests/fixtures/graph /tmp/kgs/input/graph    # or a view's metadata/kgx/ folder
INPUT_DIR=/tmp/kgs/input OUTPUT_DIR=/tmp/kgs/output python3 kg_summary.py
```

## Developing

```sh
make test        # pytest in a Python 3.12 virtualenv (.venv, made on first use)
make run-local   # build the image and run it like a compute node; report in ./out/output
make bench       # time and peak memory on a synthetic Hetionet-sized graph (FORMAT=tsv for TSV)
```

The fixture in `tests/fixtures/graph` (also the review input) is written by
`tests/fixtures/make_fixture.py`: three subjects, three samples, two protocols and four
terms, with one of each data-quality problem. `bench/make_graph.py` writes the synthetic
benchmark graph.

## Acknowledgements

KG Summary reads graphs in the KGX format and reports Biolink Model categories and
predicates, both from the Biolink community and the NCATS Biomedical Data Translator:

- [KGX](https://github.com/biolink/kgx), the Knowledge Graph Exchange format and tools.
- Unni DR, Moxon SAT, Bada M, Brush M, Bruskiewich R, Caufield JH, Clemons PA, Dancik V,
  Dumontier M, Fecho K, Glusman G, Hadlock JJ, Harris NL, Joshi A, Putman T, Qin G,
  Ramsey SA, Shefchek KA, Solbrig H, Soman K, Thessen AE, Haendel MA, Bizon C, Mungall CJ,
  The Biomedical Data Translator Consortium. Biolink Model: A universal schema for
  knowledge graphs in clinical, biomedical, and translational science. *Clinical and
  Translational Science* 15(8):1848-1855 (2022).
  [doi:10.1111/cts.13302](https://doi.org/10.1111/cts.13302)

It is built on these open-source libraries:

- [DuckDB](https://duckdb.org): Raasveldt M, Mühleisen H. DuckDB: an Embeddable
  Analytical Database. *SIGMOD 2019*, 1981-1984.
  [doi:10.1145/3299869.3320212](https://doi.org/10.1145/3299869.3320212)
- [SciPy](https://scipy.org): Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0:
  fundamental algorithms for scientific computing in Python. *Nature Methods*
  17:261-272 (2020). [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
- [NumPy](https://numpy.org): Harris CR, Millman KJ, van der Walt SJ, et al. Array
  programming with NumPy. *Nature* 585:357-362 (2020).
  [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- [Matplotlib](https://matplotlib.org): Hunter JD. Matplotlib: A 2D Graphics Environment.
  *Computing in Science & Engineering* 9(3):90-95 (2007).
  [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
- [Apache Arrow](https://arrow.apache.org) (PyArrow).

The full author lists are in [CITATION.cff](CITATION.cff).

## Credits

Built by [Pennsieve](https://pennsieve.io). Licensed under Apache-2.0.
