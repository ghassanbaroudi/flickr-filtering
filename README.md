# flico
exploring Flickr Commons

## Data
Uses the Flickr Commons collection by access via its [API](https://www.flickr.com/services/api/) 

through Alexis Mignon's [python implementation](https://github.com/alexis-mignon/python-flickr-api), ([docs](https://github.com/alexis-mignon/python-flickr-api/blob/master/docs/api-reference.md)
)

and/or

through the python library [flickrapi](https://pypi.org/project/flickrapi/)

## Setup
Run everything from the **workspace root** (the folder that contains `flico/`), so relative paths like `metadata/` resolve correctly.

1. Create and activate a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install base requirements

```bash
python3 -m pip install -r flico/pip-requirements.txt
```

3. Configure Flickr API credentials

- Copy and rename `flico/.env.example` to `flico/.env` (or use a project-level `.env`)
- Add:
  - `FLICKR_API_KEY=...`
  - `FLICKR_API_SECRET=...`

4. (Optional) Verify the Flickr API works

```bash
python3 flico/API/basic_test.py
```

## Prepare metadata (once, or when you refresh the dataset)
Download per-institution metadata CSVs into `metadata/`:

```bash
python3 flico/API/metadata.py --help
```

Then, if you want a single merged file (optional; the clustering scripts typically read `metadata/*.csv` directly):

```bash
python3 flico/API/aggregate_metadata.py
```

## Run filtering + clustering pipelines
All pipelines ultimately do **DBSCAN (haversine)** clustering on valid lat/lon. The typical knobs are:

- **`--eps-m`**: neighborhood radius in meters (start around `50`)
- **`--min-samples`**: density threshold (start around `5`)

### Pipeline A — basic clustering (geo only; no keyword filtering)
Clusters **all valid-geo** rows from `metadata/*.csv` and generates an interactive Leaflet report.

```bash
python3 flico/clusters/make_clusters.py \
  --metadata-dir metadata \
  --output-dir flico/clusters/outputs \
  --eps-m 50 \
  --min-samples 5
```

- Outputs (under `--output-dir`): `clusters.csv`, `cluster_summary.csv`, `cluster_report.html`

### Pipeline B — clustering with keyword filtering (optionally with Flickr context overlay)
Filters rows by **building/architecture keywords** (substring match over title/description/tags), then clusters the filtered set and generates an interactive report.

```bash
python3 flico/clusters_compare/make_context_comparison.py \
  --metadata-dir metadata \
  --output-dir flico/clusters_compare/outputs \
  --cluster-contexts all \
  --overlay-contexts none \
  --eps-m 50 \
  --min-samples 5
```

Common variants:

- Cluster only Flickr “outdoor” context (usually `2`):

```bash
python3 flico/clusters_compare/make_context_comparison.py \
  --metadata-dir metadata \
  --output-dir flico/clusters_compare/outputs \
  --cluster-contexts 2 \
  --overlay-contexts none \
  --eps-m 50 \
  --min-samples 5
```



- Outputs (under `--output-dir`): `clusters.csv`, `cluster_summary.csv`, `context_comparison_report.html`

### Pipeline C — clustering with keyword + vision filtering
Pipeline: **valid geo → optional context → keyword match → vision YES/NO** on each image, with optional DBSCAN + Leaflet map on the vision-YES subset.

#### 1) Install vision dependencies (one-time)

```bash
python3 -m pip install -r flico/keyword_vision/requirements.txt
```

If PyTorch isn’t installed automatically, install it first from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then re-run the command above.

#### 2) Run keyword + vision scoring
Default backend is **OpenCLIP** (local, no API key):

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --output-dir flico/keyword_vision/outputs
```

Optional backends:

```bash
# Ollama
python3 flico/keyword_vision/filter_keyword_vision.py \
  --backend ollama \
  --output-dir flico/keyword_vision/outputs

# OpenAI (requires OPENAI_API_KEY)
python3 flico/keyword_vision/filter_keyword_vision.py \
  --backend openai \
  --output-dir flico/keyword_vision/outputs
```

#### 3) Generate clusters + map for vision-YES (two options)
Option A (one command): score + cluster + Leaflet report:

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --output-dir flico/keyword_vision/outputs \
  --cluster-report
```

Option B (separate step): cluster only from the kept CSV:

```bash
python3 flico/keyword_vision/cluster_vision_map.py \
  --input-csv flico/keyword_vision/outputs/keyword_then_vision_keep.csv \
  --output-dir flico/keyword_vision/outputs \
  --eps-m 50 \
  --min-samples 5
```

- Outputs (under `--output-dir`): `keyword_then_vision.csv`, `keyword_then_vision_keep.csv`, `keyword_then_vision_reject.csv`, `vision_cache.jsonl`, `vision_summary.json`, plus (when clustering) `clusters.csv`, `cluster_summary.csv`, `vision_cluster_report.html`


