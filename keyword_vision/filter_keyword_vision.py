#!/usr/bin/env python3
"""
Keyword filter (same as clusters_compare) + small vision model YES/NO per image.

Outputs CSVs + JSON summary + JSONL cache for resume.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Load dotenv from common locations (same idea as flico/API/metadata.py)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

# Allow `python filter_keyword_vision.py` from this folder: parent `flico` on sys.path
_FLICO = Path(__file__).resolve().parent.parent
if str(_FLICO) not in sys.path:
    sys.path.insert(0, str(_FLICO))
from keyword_vision import metadata_keywords as mk  # noqa: E402


def detect_project_root(script_path: Path) -> Path:
    try:
        return script_path.resolve().parents[2]
    except Exception:
        return Path.cwd()


def load_env() -> None:
    """Load .env / env.env so API keys and OLLAMA_* vars are available (same locations as flico/API)."""
    if load_dotenv is None:
        return
    script = Path(__file__).resolve()
    flico_dir = script.parent.parent
    root = flico_dir.parent
    # Load all present files; later files fill vars not already set (override=False).
    for p in (flico_dir / "env.env", flico_dir / ".env", root / ".env"):
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def _ollama_base_url(host: str) -> str:
    return host.rstrip("/")


def check_ollama_reachable(host: str, timeout: float = 5.0) -> Optional[str]:
    """Return error message if Ollama is not reachable, else None."""
    try:
        r = requests.get(f"{_ollama_base_url(host)}/api/tags", timeout=timeout)
        r.raise_for_status()
        return None
    except Exception as exc:
        return str(exc)


def require_vision_backend_or_exit(*, backend: str, dry_run: bool, ollama_host: str) -> None:
    """Fail fast before scoring (avoids caching thousands of ERROR rows)."""
    if dry_run:
        return
    load_env()
    if backend == "openai":
        if os.getenv("OPENAI_API_KEY", "").strip():
            return
        print(
            "\n[FATAL] OPENAI_API_KEY is not set (required for --backend openai).\n\n"
            "  Add to flico/env.env: OPENAI_API_KEY=sk-...\n"
            "  Or use --backend openclip (default) or --backend ollama\n",
            file=sys.stderr,
        )
        sys.exit(1)
    if backend == "openclip":
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            print(
                "\n[FATAL] OpenCLIP backend requires PyTorch + open_clip.\n\n"
                "  pip install torch open-clip-torch\n"
                "  (CUDA: install torch from https://pytorch.org/get-started/locally/ first)\n",
                file=sys.stderr,
            )
            sys.exit(1)
        return
    err = check_ollama_reachable(ollama_host)
    if err:
        print(
            f"\n[FATAL] Cannot reach Ollama at {ollama_host!r}: {err}\n\n"
            "  Start Ollama, then pull a vision model, e.g.:\n"
            "    ollama pull llava\n\n"
            "  Set OLLAMA_HOST if Ollama runs elsewhere (default http://127.0.0.1:11434).\n"
            "  Or use --backend openclip (no Ollama).\n",
            file=sys.stderr,
        )
        sys.exit(1)


def default_vision_backend() -> str:
    b = os.getenv("VISION_BACKEND", "openclip").strip().lower()
    return b if b in ("ollama", "openai", "openclip") else "openclip"


def resolve_vision_model(backend: str, model_arg: str) -> str:
    """CLI --model wins; else env; else backend default."""
    if model_arg.strip():
        return model_arg.strip()
    if backend == "ollama":
        return os.getenv("OLLAMA_MODEL", "llava").strip() or "llava"
    if backend == "openclip":
        return os.getenv("OPENCLIP_MODEL", "ViT-B-32").strip() or "ViT-B-32"
    return os.getenv("VISION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def cached_backend_matches(cached: Dict[str, Any], current_backend: str) -> bool:
    """Legacy caches have no vision_backend — treat as openai-only."""
    b = str(cached.get("vision_backend", "openai")).strip().lower()
    return b == current_backend.strip().lower()


VISION_SYSTEM_PROMPT = """You help curate photos for a map of buildings and architecture.

Your job: decide if this image is suitable as a photograph of a building / monument / exterior architecture
(as the main or clearly visible subject).

Answer with EXACTLY two lines:
Line 1: only YES or NO (uppercase).
Line 2: a very short reason (max 15 words).

Answer NO if the image is mainly:
- floor plans, blueprints, architectural drawings, diagrams, maps, or documents;
- a ceremony, inauguration, or crowd where people dominate and the building is tiny or unclear;
- unrelated subject (portrait, nature-only, object close-up) even if metadata mentions a building.

Answer YES if a building or architectural structure is clearly visible and meaningful in the scene."""


def row_cache_key(row: pd.Series) -> str:
    """Stable key across institution CSVs (Flickr ids may collide)."""
    sid = str(row.get("source_dataset", "") or "").strip()
    pid = str(row.get("id", "") or "").strip()
    if sid and pid:
        return f"{sid}::{pid}"
    return pid or "unknown"


def parse_yes_no(text: str) -> Tuple[Optional[str], str]:
    """Return ('YES'|'NO', reason) or (None, raw) if parse fails."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return None, ""
    first = lines[0].upper()
    if first.startswith("YES"):
        label = "YES"
    elif first.startswith("NO"):
        label = "NO"
    else:
        m = re.search(r"\b(YES|NO)\b", first, re.I)
        if m:
            label = m.group(1).upper()
        else:
            return None, text.strip()
    reason = lines[1] if len(lines) > 1 else ""
    return label, reason


def fetch_image_base64(url: str, timeout: float = 30.0) -> Tuple[str, str]:
    """Return (mime_type, base64_str)."""
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "keyword-vision-filter/1.0"})
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    if not ctype.startswith("image/"):
        ctype = "image/jpeg"
    return ctype, base64.b64encode(r.content).decode("ascii")


def _chat_vision(client: Any, model: str, content: List[Dict[str, Any]], timeout: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


def call_openai_vision(
    *,
    image_url: str,
    image_mode: str,
    model: str,
    timeout: float,
) -> Tuple[str, str, Optional[str]]:
    """
    Returns (label, reason, error_message).
    label is YES, NO, or ERROR.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "ERROR", "", "Install openai package: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "ERROR", "", "OPENAI_API_KEY is not set"

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_text = "Evaluate this image for the building/architecture map."

    attempts: List[List[Dict[str, Any]]] = []
    if image_mode == "url":
        attempts.append(
            [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    elif image_mode == "download":
        try:
            mime, b64 = fetch_image_base64(image_url)
            data_url = f"data:{mime};base64,{b64}"
            attempts.append(
                [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
        except Exception as exc:
            return "ERROR", "", f"download failed: {exc}"
    else:
        # auto: try remote URL first, then download+base64
        attempts.append(
            [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        try:
            mime, b64 = fetch_image_base64(image_url)
            data_url = f"data:{mime};base64,{b64}"
            attempts.append(
                [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
        except Exception:
            pass

    last_err: Optional[str] = None
    for content in attempts:
        try:
            text = _chat_vision(client, model, content, timeout)
            label, reason = parse_yes_no(text)
            if label in ("YES", "NO"):
                return label, reason, None
            last_err = f"Unparseable model output: {text[:200]}"
        except Exception as exc:
            last_err = str(exc)
            continue

    return "ERROR", "", last_err or "vision call failed"


def call_ollama_vision(
    *,
    image_url: str,
    model: str,
    ollama_host: str,
    timeout: float,
) -> Tuple[str, str, Optional[str]]:
    """
    Local Ollama vision (/api/chat). Image is always downloaded and sent as base64.
    Returns (label, reason, error_message).
    """
    try:
        _mime, b64 = fetch_image_base64(image_url)
    except Exception as exc:
        return "ERROR", "", f"download failed: {exc}"

    base = _ollama_base_url(ollama_host)
    user_prompt = (
        "Evaluate this image for the building/architecture map. "
        "Follow the system instructions exactly: line 1 YES or NO, line 2 short reason."
    )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": [b64]},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        r = requests.post(f"{base}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""
        text = str(text).strip()
        label, reason = parse_yes_no(text)
        if label in ("YES", "NO"):
            return label, reason, None
        return "ERROR", "", f"Unparseable model output: {text[:300]}"
    except Exception as exc:
        return "ERROR", "", str(exc)


def process_row(
    row: pd.Series,
    *,
    backend: str,
    image_mode: str,
    model: str,
    timeout: float,
    dry_run: bool,
    ollama_host: str,
) -> Dict[str, Any]:
    pid = str(row.get("id", "")).strip()
    url = str(row.get("image_url", "") or "").strip()
    out: Dict[str, Any] = {
        "vision_label": "SKIP",
        "vision_reason": "",
        "vision_error": "",
        "vision_model": model if not dry_run else "dry-run",
        "vision_backend": backend,
    }
    if not url:
        out["vision_label"] = "SKIP"
        out["vision_error"] = "missing image_url"
        return out

    if dry_run:
        out["vision_label"] = "YES"
        out["vision_reason"] = "dry-run placeholder"
        return out

    if backend == "openclip":
        return (
            "ERROR",
            "",
            "openclip is scored in batched mode; use main() batch path, not process_row",
        )

    if backend == "ollama":
        label, reason, err = call_ollama_vision(
            image_url=url, model=model, ollama_host=ollama_host, timeout=timeout
        )
    else:
        label, reason, err = call_openai_vision(
            image_url=url, image_mode=image_mode, model=model, timeout=timeout
        )
    out["vision_label"] = label
    out["vision_reason"] = reason
    if err:
        out["vision_error"] = err
    return out


def read_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            key = str(obj.get("cache_key", "") or "").strip()
            if not key:
                pid = str(obj.get("id", "")).strip()
                sid = str(obj.get("source_dataset", "")).strip()
                key = f"{sid}::{pid}" if sid and pid else pid
            if key:
                out[key] = obj
        except json.JSONDecodeError:
            continue
    return out


def append_cache(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def call_with_retries(
    fn: "Callable[[], Tuple[str, str, Optional[str]]]",
    *,
    max_retries: int,
    base_delay: float,
    url: str = "",
) -> Tuple[str, str, Optional[str]]:
    """
    Call fn() up to max_retries+1 times.  Returns on the first non-ERROR result.
    Between attempts: sleeps base_delay * 2^attempt seconds.
    """
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            label, reason, err = fn()
        except Exception as exc:
            label, reason, err = "ERROR", "", str(exc)
        if label != "ERROR":
            return label, reason, err
        last_err = err
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            short = (url[:70] + "…") if len(url) > 73 else url
            print(
                f"[RETRY] attempt {attempt + 1}/{max_retries} failed"
                + (f" ({err})" if err else "")
                + f" — waiting {delay:.0f}s  [{short}]",
                flush=True,
            )
            time.sleep(delay)
    return "ERROR", "", last_err


def parse_args() -> argparse.Namespace:
    script = Path(__file__)
    root = detect_project_root(script)
    default_metadata = root / "metadata"
    default_out = script.resolve().parent / "outputs"

    p = argparse.ArgumentParser(description="Keyword filter + vision YES/NO for each image.")
    p.add_argument("--metadata-dir", type=Path, default=default_metadata)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument(
        "--cluster-contexts",
        type=str,
        default="all",
        help="Same as clusters_compare: Flickr context pre-filter before keywords, or 'all'.",
    )
    p.add_argument("--keywords", type=str, default="", help="Comma-separated keywords (default: building list).")
    p.add_argument("--keywords-file", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max rows after keyword filter (0 = all).")
    p.add_argument("--dry-run", action="store_true", help="No API calls; all vision_label=YES.")
    p.add_argument(
        "--backend",
        choices=["openclip", "ollama", "openai"],
        default=default_vision_backend(),
        help="Vision: OpenCLIP batched (default, fast on GPU), Ollama, or OpenAI.",
    )
    p.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL (default: http://127.0.0.1:11434 or env OLLAMA_HOST).",
    )
    p.add_argument(
        "--image-mode",
        choices=["auto", "url", "download"],
        default="auto",
        help="OpenAI only: auto/url/download. Ollama always downloads the image.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Model: OpenCLIP arch e.g. ViT-B-32 (OPENCLIP_MODEL); Ollama llava; OpenAI gpt-4o-mini.",
    )
    p.add_argument(
        "--clip-pretrained",
        type=str,
        default="",
        help="OpenCLIP pretrained tag (default: env OPENCLIP_PRETRAINED or laion2b_s34b_b79k).",
    )
    p.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
        help="OpenCLIP: images per GPU/CPU batch (default: 32).",
    )
    p.add_argument(
        "--clip-device",
        type=str,
        default="",
        help="OpenCLIP device, e.g. cuda, cuda:0, cpu (default: auto).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable OpenCLIP progress bar (tqdm) and periodic logs.",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Per-image API retries for Ollama/OpenAI before giving up (default: 3).",
    )
    p.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Base seconds between retries; doubles each attempt (default: 10).",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="OpenCLIP: max download retries per image on 429/503 (default: 1).",
    )
    p.add_argument(
        "--stop-after-errors",
        type=int,
        default=1,
        help=(
            "Stop the run after this many consecutive persistent errors (after all retries). "
            "Progress is saved; the next run resumes from the first unprocessed or errored row. "
            "0 = never stop early (default: 1)."
        ),
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = sequential).")
    p.add_argument("--no-cache", action="store_true", help="Ignore existing vision_cache.jsonl.")
    p.add_argument(
        "--skip-cached-errors",
        action="store_true",
        default=False,
        help=(
            "Do not retry rows that already have an ERROR result in the cache for the "
            "current backend.  Use this to skip permanently broken URLs (e.g. deleted "
            "Flickr images) so they are not re-scored on every run."
        ),
    )
    p.add_argument(
        "--cluster-report",
        action="store_true",
        help="After export, run DBSCAN + Leaflet on keyword_then_vision_keep.csv (vision_cluster_report.html).",
    )
    p.add_argument("--cluster-eps-m", type=float, default=50.0, help="DBSCAN eps (meters) for --cluster-report.")
    p.add_argument(
        "--cluster-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples for --cluster-report.",
    )
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    vision_model = resolve_vision_model(args.backend, args.model)
    require_vision_backend_or_exit(
        backend=args.backend, dry_run=args.dry_run, ollama_host=args.ollama_host
    )

    cluster_ctx = mk.parse_cluster_contexts(args.cluster_contexts)
    keywords = mk.load_keywords_list(
        keywords_arg=args.keywords if str(args.keywords).strip() else None,
        keywords_file=args.keywords_file,
    )

    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory does not exist: {args.metadata_dir}")

    print(f"[INFO] Keywords: {len(keywords)} term(s).")
    print(f"[INFO] Vision: backend={args.backend} model={vision_model!r}")
    df_all = mk.load_all_metadata(args.metadata_dir)
    df_geo = mk.keep_valid_geo(df_all)
    df_ctx = df_geo if cluster_ctx is None else mk.filter_contexts(df_geo, cluster_ctx)
    df_kw = mk.filter_by_keywords(df_ctx, keywords)
    if df_kw.empty:
        raise ValueError("No rows after keyword filter.")

    if args.limit > 0:
        df_kw = df_kw.iloc[: args.limit].reset_index(drop=True)
        print(f"[INFO] Limited to first {args.limit:,} rows.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "vision_cache.jsonl"

    cache: Dict[str, Dict[str, Any]] = {}
    if not args.no_cache:
        cache = read_cache(cache_path)
        print(f"[INFO] Loaded {len(cache):,} cached vision results.")

    rec_by_idx: Dict[Any, Dict[str, Any]] = {}
    errors = 0
    cache_lock = threading.Lock()

    indices = list(df_kw.index)
    pending: List[Tuple[Any, pd.Series]] = []
    for i in indices:
        row = df_kw.loc[i]
        ck = row_cache_key(row)
        if ck and ck in cache and not args.no_cache:
            c = cache[ck]
            if not cached_backend_matches(c, args.backend):
                pending.append((i, row))
            elif str(c.get("vision_label", "")).upper() == "ERROR":
                # Skip permanently-failed URLs when requested; otherwise always retry
                # ERROR rows (e.g. a missing API key from a first run is now fixed).
                if args.skip_cached_errors:
                    rec = row.to_dict()
                    rec["vision_label"] = "ERROR"
                    rec["vision_reason"] = c.get("vision_reason", "")
                    rec["vision_error"] = c.get("vision_error", "")
                    rec["vision_model"] = c.get("vision_model", vision_model)
                    rec["vision_backend"] = c.get("vision_backend", args.backend)
                    rec_by_idx[i] = rec
                else:
                    pending.append((i, row))
            else:
                rec = row.to_dict()
                rec["vision_label"] = c.get("vision_label", "ERROR")
                rec["vision_reason"] = c.get("vision_reason", "")
                rec["vision_error"] = c.get("vision_error", "")
                rec["vision_model"] = c.get("vision_model", vision_model)
                rec["vision_backend"] = c.get("vision_backend", args.backend)
                rec_by_idx[i] = rec
        else:
            pending.append((i, row))

    def _score_row(row: pd.Series) -> Dict[str, Any]:
        """Score one row with per-image retries for API-based backends."""
        url = str(row.get("image_url", "") or "").strip()
        base: Dict[str, Any] = {
            "vision_label": "SKIP",
            "vision_reason": "",
            "vision_error": "",
            "vision_model": vision_model if not args.dry_run else "dry-run",
            "vision_backend": args.backend,
        }
        if not url:
            base["vision_error"] = "missing image_url"
            return base
        if args.dry_run:
            base["vision_label"] = "YES"
            base["vision_reason"] = "dry-run placeholder"
            return base

        if args.backend == "ollama":
            def _call() -> Tuple[str, str, Optional[str]]:
                return call_ollama_vision(
                    image_url=url, model=vision_model, ollama_host=args.ollama_host, timeout=args.timeout
                )
        else:
            def _call() -> Tuple[str, str, Optional[str]]:
                return call_openai_vision(
                    image_url=url, image_mode=args.image_mode, model=vision_model, timeout=args.timeout
                )

        label, reason, err = call_with_retries(_call, max_retries=args.retries, base_delay=args.retry_delay, url=url)
        base["vision_label"] = label
        base["vision_reason"] = reason
        base["vision_error"] = err or ""
        return base

    def worker(item: Tuple[Any, pd.Series]) -> Tuple[Any, Dict[str, Any]]:
        i, row = item
        extra = _score_row(row)
        rec = row.to_dict()
        rec.update(extra)
        ck = row_cache_key(row)
        if ck and ck != "unknown" and not args.dry_run:
            obj = {
                "cache_key": ck,
                "id": str(row.get("id", "")).strip(),
                "source_dataset": str(row.get("source_dataset", "") or ""),
                "vision_label": extra["vision_label"],
                "vision_reason": extra["vision_reason"],
                "vision_error": extra.get("vision_error", ""),
                "vision_model": vision_model,
                "vision_backend": args.backend,
            }
            with cache_lock:
                append_cache(cache_path, obj)
        return i, rec

    if pending:
        n_cached = len(indices) - len(pending)
        print(
            f"[INFO] Vision scoring {len(pending):,} rows (workers={args.workers}); "
            f"cache hits {n_cached:,} (cached ERROR rows are always re-scored)."
        )

    stopped_early = False

    if pending and args.backend == "openclip" and not args.dry_run:
        if args.workers > 1:
            print("[INFO] OpenCLIP uses batched inference; --workers ignored.")
        from keyword_vision.clip_vision import ClipVisionRuntime, run_batched_clip

        clip_pretrained = (args.clip_pretrained or "").strip() or os.getenv(
            "OPENCLIP_PRETRAINED", "laion2b_s34b_b79k"
        ).strip()
        clip_dev = (args.clip_device or "").strip() or None
        print(
            f"[INFO] OpenCLIP loading {vision_model!r} pretrained={clip_pretrained!r} "
            f"(device={clip_dev or 'auto'})"
        )
        rt = ClipVisionRuntime(vision_model, clip_pretrained, device=clip_dev)
        print(f"[INFO] OpenCLIP ready on {rt.device}")
        urls = [str(row.get("image_url", "") or "").strip() for _, row in pending]

        def _on_clip_result(idx: int, label: str, reason: str, err: Optional[str]) -> None:
            """Write each result to cache immediately as it is scored."""
            nonlocal errors
            i, row = pending[idx]
            extra: Dict[str, Any] = {
                "vision_label": label,
                "vision_reason": reason,
                "vision_error": err or "",
                "vision_model": vision_model,
                "vision_backend": args.backend,
            }
            rec = row.to_dict()
            rec.update(extra)
            rec_by_idx[i] = rec
            if label == "ERROR":
                errors += 1
            ck = row_cache_key(row)
            if ck and ck != "unknown":
                obj: Dict[str, Any] = {
                    "cache_key": ck,
                    "id": str(row.get("id", "")).strip(),
                    "source_dataset": str(row.get("source_dataset", "") or ""),
                    "vision_label": label,
                    "vision_reason": reason,
                    "vision_error": err or "",
                    "vision_model": vision_model,
                    "vision_backend": args.backend,
                }
                append_cache(cache_path, obj)

        _scored, stopped_early = run_batched_clip(
            urls,
            rt,
            batch_size=args.clip_batch_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            progress=not args.no_progress,
            stop_after_errors=args.stop_after_errors,
            on_result=_on_clip_result,
        )

    elif args.workers <= 1:
        consecutive_errors = 0
        for item in pending:
            i, rec = worker(item)
            rec_by_idx[i] = rec
            if rec.get("vision_label") == "ERROR":
                errors += 1
                consecutive_errors += 1
                if args.stop_after_errors > 0 and consecutive_errors >= args.stop_after_errors:
                    print(
                        f"\n[STOP] {consecutive_errors} consecutive persistent error(s) "
                        f"(each retried {args.retries}×). "
                        "Saving progress and stopping. Re-run after a cooldown to resume.",
                        flush=True,
                    )
                    stopped_early = True
                    break
            else:
                consecutive_errors = 0

    else:
        stop_event = threading.Event()
        consecutive_errors = 0

        def worker_guarded(item: Tuple[Any, pd.Series]) -> Optional[Tuple[Any, Dict[str, Any]]]:
            if stop_event.is_set():
                return None
            return worker(item)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker_guarded, item): item for item in pending}
            for fut in as_completed(futs):
                try:
                    result = fut.result()
                    if result is None:
                        continue
                    i, rec = result
                    rec_by_idx[i] = rec
                    if rec.get("vision_label") == "ERROR":
                        errors += 1
                        consecutive_errors += 1
                        if args.stop_after_errors > 0 and consecutive_errors >= args.stop_after_errors:
                            if not stop_event.is_set():
                                print(
                                    f"\n[STOP] {consecutive_errors} consecutive persistent error(s) "
                                    f"(each retried {args.retries}×). "
                                    "Saving progress and stopping. Re-run after a cooldown to resume.",
                                    flush=True,
                                )
                                stop_event.set()
                                stopped_early = True
                    else:
                        consecutive_errors = 0
                except Exception as exc:
                    print(f"[WARN] Worker failed: {exc}")
                    errors += 1

    # Only include rows that were actually processed (partial on early stop).
    rows_out = [rec_by_idx[i] for i in indices if i in rec_by_idx]
    df_out = pd.DataFrame(rows_out)

    stem = out_dir / "keyword_then_vision"
    df_out.to_csv(stem.with_suffix(".csv"), index=False)
    keep = df_out[df_out["vision_label"] == "YES"]
    reject = df_out[df_out["vision_label"] == "NO"]
    keep.to_csv(stem.with_name("keyword_then_vision_keep.csv"), index=False)
    reject.to_csv(stem.with_name("keyword_then_vision_reject.csv"), index=False)

    n_total = len(df_out)
    n_yes = int((df_out["vision_label"] == "YES").sum()) if n_total else 0
    n_no = int((df_out["vision_label"] == "NO").sum()) if n_total else 0
    n_skip = int((df_out["vision_label"] == "SKIP").sum()) if n_total else 0
    n_err = int((df_out["vision_label"] == "ERROR").sum()) if n_total else 0

    rows_remaining = len(indices) - len(rows_out)
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "keyword_rows": int(len(df_kw)),
        "rows_scored": n_total,
        "rows_remaining": rows_remaining,
        "stopped_early": stopped_early,
        "vision_yes": n_yes,
        "vision_no": n_no,
        "vision_skip": n_skip,
        "vision_error": n_err,
        "rejection_rate_no": (n_no / n_total) if n_total else 0.0,
        "backend": args.backend,
        "model": vision_model,
        "dry_run": args.dry_run,
        "errors_during_run": errors,
    }
    (out_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if stopped_early:
        print(
            f"\n[INFO] Run stopped early: {n_total:,} rows saved, {rows_remaining:,} rows remain.\n"
            "       Re-run the same command after a cooldown — it will resume automatically.",
            flush=True,
        )
    print(f"[INFO] Wrote CSVs + vision_summary.json under {out_dir}")

    if args.cluster_report:
        keep_csv = out_dir / "keyword_then_vision_keep.csv"
        if not keep_csv.exists():
            print("[WARN] --cluster-report: keyword_then_vision_keep.csv missing; skip.")
        else:
            from keyword_vision.cluster_vision_map import run_cluster_and_report

            run_cluster_and_report(
                keep_csv,
                out_dir,
                eps_m=args.cluster_eps_m,
                min_samples=args.cluster_min_samples,
            )


if __name__ == "__main__":
    main()
