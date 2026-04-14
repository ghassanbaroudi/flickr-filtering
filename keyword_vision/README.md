# Keyword + small vision filter

Pipeline: **metadata → valid geo → optional Flickr context → keyword match (same as `clusters_compare`) → vision YES/NO** on each image.

**Default vision backend is OpenCLIP** (local, batched on GPU when available; no API key). [Ollama](https://ollama.com/) and OpenAI are optional (`--backend ollama` / `--backend openai`).

## Setup

```bash
cd /path/to/projet_bachelor
pip install -r flico/keyword_vision/requirements.txt
```

Install **PyTorch** for your platform if `pip` does not pull a suitable build: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### OpenCLIP (default)

- **`--backend openclip`** (default) uses a fast ViT-B-32 checkpoint (`laion2b_s34b_b79k` unless you set **`OPENCLIP_PRETRAINED`** or **`--clip-pretrained`**).
- **`OPENCLIP_MODEL`** overrides the architecture (default `ViT-B-32`).
- **`--clip-batch-size`** (default 32) and **`--clip-device`** (`cuda`, `cpu`, or empty for auto) tune throughput.
- Images are downloaded in memory only; the model weights are cached by PyTorch/OpenCLIP under your user cache (not in this repo).

### Ollama

1. Install and start **Ollama**.
2. Pull a **vision** model (required once):

   ```bash
   ollama pull llava
   ```

   Other options: `llama3.2-vision`, `bakllava`, etc. Set `--model` or env **`OLLAMA_MODEL`** (default `llava`).

3. Optional: **`OLLAMA_HOST`** if Ollama is not on `http://127.0.0.1:11434`.

The script checks that Ollama responds (`/api/tags`) before scoring so you don’t cache thousands of ERROR rows.

### OpenAI (optional)

- **`OPENAI_API_KEY`** in `flico/env.env` or project `.env`
- Run with **`--backend openai`**
- Optional: `OPENAI_BASE_URL`, **`VISION_MODEL`** (default `gpt-4o-mini`)

Env files are loaded from `flico/env.env`, `flico/.env`, and project `.env`.

**Cache:** `vision_cache.jsonl` stores `vision_backend`. Switching backends (e.g. OpenCLIP ↔ Ollama) re-scores rows. Cached **ERROR** rows are always re-tried.

## Run

From the **project root** (`projet_bachelor`):

Dry run on 20 rows (no API calls):

```bash
python3 flico/keyword_vision/filter_keyword_vision.py --limit 20 --dry-run
```

Full pass with **OpenCLIP** (default):

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --output-dir flico/keyword_vision/outputs
```

Full pass with **Ollama**:

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --backend ollama \
  --output-dir flico/keyword_vision/outputs
```

Explicit Ollama model / host:

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --backend ollama --model llava --ollama-host http://127.0.0.1:11434 \
  --output-dir flico/keyword_vision/outputs
```

**OpenAI:**

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --backend openai \
  --output-dir flico/keyword_vision/outputs
```

**Vision + DBSCAN + Leaflet** (append `--cluster-report`):

```bash
python3 flico/keyword_vision/filter_keyword_vision.py \
  --output-dir flico/keyword_vision/outputs \
  --cluster-report
```

Or cluster only from `keyword_then_vision_keep.csv`:

```bash
python3 flico/keyword_vision/cluster_vision_map.py \
  --input-csv flico/keyword_vision/outputs/keyword_then_vision_keep.csv \
  --output-dir flico/keyword_vision/outputs \
  --eps-m 50 --min-samples 5
```

If **`keyword_then_vision_keep.csv` is empty**, the script tries **`keyword_then_vision.csv`** in the same folder (`vision_label=YES` only).

Tune `--sleep-seconds` if needed (mostly relevant for cloud APIs).

## Outputs (under `--output-dir`)

| File | Purpose |
|------|---------|
| `keyword_then_vision.csv` | All keyword-matched rows + `vision_label`, `vision_reason`, `vision_model`, `vision_backend` |
| `keyword_then_vision_keep.csv` | Rows where vision said **YES** |
| `keyword_then_vision_reject.csv` | Rows where vision said **NO** |
| `vision_cache.jsonl` | Per-row cache (includes `vision_backend`) |
| `vision_summary.json` | Counts, `backend`, `model` |
| `clusters.csv` | DBSCAN labels (with `--cluster-report` or `cluster_vision_map.py`) |
| `cluster_summary.csv` | Cluster sizes / bbox |
| `vision_cluster_report.html` | **Leaflet** map |

## Interpretation

- Compare **keyword-only count** vs **`vision_label=YES` count**; the gap is “vision rejects”.
- Tune the **system prompt** in `filter_keyword_vision.py` if your definition of “building photo” differs.

## Notes

- **Ollama:** each image is **downloaded** and sent as base64 to `/api/chat` (Flickr URLs).
- **OpenAI:** `--image-mode` controls URL vs download (see `--help`).
