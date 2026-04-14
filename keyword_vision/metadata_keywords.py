"""
Reuse keyword + metadata helpers from `clusters_compare/make_context_comparison.py`
without duplicating code (dynamic import; same deps: pandas, numpy, sklearn).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_context_compare_module() -> ModuleType:
    script = Path(__file__).resolve()
    mod_path = script.parent.parent / "clusters_compare" / "make_context_comparison.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Expected module at {mod_path}")
    spec = importlib.util.spec_from_file_location("make_context_comparison_cc", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_m = _load_context_compare_module()

# Re-export for filter_keyword_vision
DEFAULT_BUILDING_KEYWORDS = _m.DEFAULT_BUILDING_KEYWORDS
load_keywords_list = _m.load_keywords_list
filter_by_keywords = _m.filter_by_keywords
load_all_metadata = _m.load_all_metadata
keep_valid_geo = _m.keep_valid_geo
parse_cluster_contexts = _m.parse_cluster_contexts
filter_contexts = _m.filter_contexts
