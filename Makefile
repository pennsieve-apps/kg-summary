.PHONY: test image run-local bench clean

IMAGE ?= kg-summary:local
PYTHON ?= python3.12
VENV := .venv
BENCH := out/bench

# A virtualenv with the app's and the tests' requirements, made on first use.
$(VENV)/bin/python: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt -r requirements-dev.txt
	touch $@

test: $(VENV)/bin/python
	$(VENV)/bin/python -m pytest -v tests

image:
	docker build -t $(IMAGE) .

# Run the image the way a compute node does: an arbitrary uid, a read-only file system,
# read-only inputs, the graph in INPUT_DIR/graph. The report lands in ./out/output.
run-local: image
	rm -rf out/input out/output && mkdir -p out/input out/output && chmod 777 out/output
	cp -R tests/fixtures/graph out/input/graph
	docker run --rm --user 65534:65534 --read-only --tmpfs /tmp \
		-v $(CURDIR)/out/input:/data/input:ro \
		-v $(CURDIR)/out/output:/data/output \
		-e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
		-e EXECUTION_RUN_ID=local-run -e ENVIRONMENT=local \
		$(IMAGE)

# Time and peak memory on a synthetic Hetionet-sized graph (92k nodes, 2.3M edges) in the
# image, limited to app.yml's runtime (1 vCPU, 2 GiB). FORMAT=tsv reads the TSV files.
FORMAT ?= parquet
bench: image $(VENV)/bin/python
	rm -rf $(BENCH) && mkdir -p $(BENCH)/output && chmod 777 $(BENCH)/output
	$(VENV)/bin/python bench/make_graph.py $(BENCH)/input $(if $(filter tsv,$(FORMAT)),--tsv)
	docker run --rm --user 65534:65534 --read-only --tmpfs /tmp --cpus 1 --memory 2g \
		-v $(CURDIR)/$(BENCH)/input:/data/input:ro -v $(CURDIR)/$(BENCH)/output:/data/output \
		-v $(CURDIR)/bench/measure.sh:/bench/measure.sh:ro \
		-e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
		--entrypoint sh $(IMAGE) /bench/measure.sh

clean:
	rm -rf out
