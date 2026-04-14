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

- Add an overlay layer (not clustered) for another context bucket:

```bash
python3 flico/clusters_compare/make_context_comparison.py \
  --metadata-dir metadata \
  --output-dir flico/clusters_compare/outputs \
  --cluster-contexts 2 \
  --overlay-contexts 1 \
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


# Dev logs



## Feb 24 2026

Had to fixed a **truncating issue**:
Flickr behaviors for high-volume Commons accounts: search scopes or server-side caps truncate results without error.
FIX:
Replaced `flickr.photos.search(user_id=inst_id, is_commons=True, ...)` with `flickr.people.getPublicPhotos(user_id=inst_id, ...)`.

[metadata script](API/metadata.py) script seems ok.

I checked the [Royal Museums Greenwhich](https://www.flickr.com/photos/nationalmaritimemuseum/with/38608148550/) that has 0 photos obtainable via search with the API. It seems that their pictures are copyrighted so it makes sense. This means that copyrighted picture don't appear.

## Feb 23 2026

| Field         | Available via extras? | Notes                                                            |
| ------------- | --------------------- | ---------------------------------------------------------------- |
| id            | ✅ Yes                 | Always in search results                                         |
| secret        | ✅ Yes                 | Always included                                                  |
| title         | ❌ No* (good enough)   | *Basic title sometimes in search, but full title needs getInfo() |
| description   | ✅ Yes                 | photo.description._content                                       |
| date_taken    | ✅ Yes                 | photo.datetaken                                                  |
| date_uploaded | ✅ Yes                 | photo.dateupload (Unix timestamp)                                |
| locations     | ✅ Yes                 | photo.latitude, photo.longitude                                  |
| comments      | ❌ No                  | Needs getInfo()                                                  |
| size          | ✅ Yes                 | url_o + o_width/o_height from o_dims                             |
| image_url     | ✅ Yes                 | url_o, url_c, url_m, etc.                                        |
| notes         | ❌ No                  | Needs getInfo()                                                  |
| tags          | ✅ Yes                 | photo.tags.tag[] array                                           |


Not all metadata can be obtained with `extras`
```    Skipping malformed photo: 'str' object has no attribute 'get'
extras="description,license,date_upload,date_taken,owner_name,geo,tags,machine_tags,o_dims,views,url_o,url_c"
```

Trying to use `extras` in search instead of `photo.getInfo`. (With one `search` query I can get 500 photos, while `getInfo`must be called on **every** photo).

Flickr limits the access to the API per key. They ask for us to stay under **3600 queries per hour** across the whole key (which means the aggregate of all the users of the integration).

I am getting a lot of error messages due to rate limiting.
```
201 : Sorry, the Flickr API service is not currently available.
```

I am switching to the flickrapi library instead, I think it is more stable.
