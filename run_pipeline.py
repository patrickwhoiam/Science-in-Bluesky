#!/usr/bin/env python3
"""
Science Bluesky Pipeline: top-level posts + DOI/OpenAlex + publication links + influence metrics
=============================================================================================

One-file data collection and preprocessing pipeline for Bluesky science posts.

Main design:
1. Keep only real top-level Bluesky posts:
   - URI must be app.bsky.feed.post
   - operation must be create, if operation column exists/configured
   - reply_parent_uri and reply_root_uri must be null/empty
2. Extract DOI evidence from text and facets_uris.
   - facets_uris is prioritized because Bluesky often truncates visible links in text.
   - text DOI is kept as fallback for manually typed DOI mentions.
3. Extract publication-link candidates from curated academic domains.
4. Validate unique DOI values with OpenAlex.
5. Build final datasets.
6. Hydrate public engagement counts from Bluesky AppView API:
   like_count, reply_count, repost_count, quote_count, bookmark_count if returned.

Default usage:
    python run_pipeline.py

Useful stage-by-stage usage:
    python run_pipeline.py --stage extract --workers 2
    python run_pipeline.py --stage validate
    python run_pipeline.py --stage build
    python run_pipeline.py --stage engage

Safe rerun:
    Run the same command again without --fresh. Existing per-file checkpoints and API caches are reused.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
import yaml
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_posts_dir: Path
    data_dir: Path
    interim_dir: Path
    processed_dir: Path
    final_dataset_dir: Path
    outputs_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path
    cache_dir: Path
    extracted_by_file_dir: Path


@dataclass(frozen=True)
class Config:
    paths: ProjectPaths
    file_glob: str
    max_files: int | None
    random_seed: int
    sample_rows: int
    manual_annotation_sample_size: int
    workers: int
    top_level_only: bool
    openalex_api_key: str
    openalex_batch_size: int
    openalex_sleep_seconds: float
    openalex_max_retries: int
    bluesky_api_base: str
    bluesky_batch_size: int
    bluesky_sleep_seconds: float
    bluesky_max_retries: int
    operation_filter: list[str] | None
    columns: dict[str, str]
    publication_domains: list[str]
    time_granularity: str


DEFAULT_CONFIG_TEXT = """# Science Bluesky pipeline config
# Edit only the values below.

project:
  # Folder containing your raw Bluesky parquet files.
  raw_posts_dir: "E:/parquet_data/posts"

  # Output location. Usually keep this as "." so outputs are created in this project folder.
  project_dir: "."

  # Raw parquet filename pattern.
  file_glob: "*.parquet"

pipeline:
  # Use null for the full dataset. Use a small number like 2 for testing.
  max_files: null

  # Keep only actual created post records if operation column exists.
  operation_filter: ["create"]

  # IMPORTANT: for influence analysis, keep original/top-level posts only.
  # This removes replies where reply_parent_uri/reply_root_uri are not null.
  top_level_only: true

  # Parallel parquet extraction. Start with 2. Use 1 if memory is high.
  workers: 2

  random_seed: 42
  sample_rows: 5
  manual_annotation_sample_size: 200

openalex:
  # Recommended for real runs. You may also set environment variable OPENALEX_API_KEY.
  # OpenAlex API keys are available from openalex.org/settings/api.
  api_key: ""
  batch_size: 50
  sleep_seconds: 0.2
  max_retries: 4

bluesky:
  # Public Bluesky AppView API. No login required for public post views.
  api_base: "https://public.api.bsky.app"
  batch_size: 25
  sleep_seconds: 0.25
  max_retries: 4

columns:
  uri: "uri"
  did: "did"
  operation: "operation"
  rkey: "rkey"
  created_at: "created_at"
  text: "text"
  langs: "langs"
  reply_parent_uri: "reply_parent_uri"
  reply_root_uri: "reply_root_uri"
  facets_uris: "facets_uris"

doi_resolver_domains:
  - "doi.org"
  - "dx.doi.org"

publication_domains:
  # Preprint / open repositories
  - "arxiv.org"
  - "biorxiv.org"
  - "medrxiv.org"
  - "ssrn.com"
  - "osf.io"
  - "preprints.org"
  - "researchsquare.com"

  # Major publishers / journals / libraries / proceedings
  - "nature.com"
  - "science.org"
  - "cell.com"
  - "nejm.org"
  - "thelancet.com"
  - "jamanetwork.com"
  - "bmj.com"
  - "plos.org"
  - "frontiersin.org"
  - "mdpi.com"
  - "sciencedirect.com"
  - "springer.com"
  - "link.springer.com"
  - "wiley.com"
  - "onlinelibrary.wiley.com"
  - "tandfonline.com"
  - "oup.com"
  - "academic.oup.com"
  - "cambridge.org"
  - "sagepub.com"
  - "acs.org"
  - "pubs.acs.org"
  - "rsc.org"
  - "ieee.org"
  - "ieeexplore.ieee.org"
  - "acm.org"
  - "dl.acm.org"
  - "jmlr.org"
  - "proceedings.mlr.press"
  - "aclweb.org"
  - "aclanthology.org"
  - "pubmed.ncbi.nlm.nih.gov"
  - "ncbi.nlm.nih.gov"
"""


# =============================================================================
# Regex and general utilities
# =============================================================================

# DOI regex intentionally broad. Cleanup and OpenAlex validation decide final quality.
DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/[^\s<>'\"\]\[{}|\\^`]+)",
    flags=re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s<>'\"\]\[{}]+", flags=re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?)]}>'\""
NULL_STRINGS = {"", "null", "none", "nan", "nat"}


def ensure_default_config(config_path: Path) -> None:
    if not config_path.exists():
        config_path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
        print(f"Created default config file: {config_path}")
        print("Edit config.yaml first, especially project.raw_posts_dir, then rerun.")
        sys.exit(0)


def load_config(config_path: Path, cli_workers: int | None = None, cli_max_files: int | None = None) -> Config:
    ensure_default_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    project_dir = Path(raw["project"].get("project_dir", ".")).expanduser().resolve()
    raw_posts_dir = Path(raw["project"]["raw_posts_dir"]).expanduser()
    if not raw_posts_dir.is_absolute():
        raw_posts_dir = (project_dir / raw_posts_dir).resolve()

    data_dir = project_dir / "data"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    final_dataset_dir = project_dir / "final_dataset"
    outputs_dir = project_dir / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    logs_dir = project_dir / "logs"
    cache_dir = project_dir / "cache"
    extracted_by_file_dir = interim_dir / "extracted_by_file"

    paths = ProjectPaths(
        root=project_dir,
        raw_posts_dir=raw_posts_dir,
        data_dir=data_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        final_dataset_dir=final_dataset_dir,
        outputs_dir=outputs_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
        cache_dir=cache_dir,
        extracted_by_file_dir=extracted_by_file_dir,
    )

    pipe = raw.get("pipeline", {})
    openalex = raw.get("openalex", {})
    bluesky = raw.get("bluesky", {})

    api_key = openalex.get("api_key", "") or os.getenv("OPENALEX_API_KEY", "")
    workers = cli_workers if cli_workers is not None else int(pipe.get("workers", 2))
    max_files = cli_max_files if cli_max_files is not None else pipe.get("max_files")

    return Config(
        paths=paths,
        file_glob=raw["project"].get("file_glob", "*.parquet"),
        max_files=max_files,
        random_seed=int(pipe.get("random_seed", 42)),
        sample_rows=int(pipe.get("sample_rows", 5)),
        manual_annotation_sample_size=int(pipe.get("manual_annotation_sample_size", 200)),
        workers=max(1, int(workers)),
        top_level_only=bool(pipe.get("top_level_only", True)),
        openalex_api_key=str(api_key).strip(),
        openalex_batch_size=min(int(openalex.get("batch_size", 50)), 100),
        openalex_sleep_seconds=float(openalex.get("sleep_seconds", 0.2)),
        openalex_max_retries=int(openalex.get("max_retries", 4)),
        bluesky_api_base=str(bluesky.get("api_base", "https://public.api.bsky.app")).rstrip("/"),
        bluesky_batch_size=min(int(bluesky.get("batch_size", 25)), 25),
        bluesky_sleep_seconds=float(bluesky.get("sleep_seconds", 0.25)),
        bluesky_max_retries=int(bluesky.get("max_retries", 4)),
        operation_filter=pipe.get("operation_filter"),
        columns=raw.get("columns", {}),
        publication_domains=sorted({d.lower().strip() for d in raw.get("publication_domains", []) if d}),
        time_granularity=raw.get("analysis", {}).get("time_granularity", "W"),
    )


def make_dirs(cfg: Config) -> None:
    for p in [
        cfg.paths.data_dir,
        cfg.paths.interim_dir,
        cfg.paths.processed_dir,
        cfg.paths.final_dataset_dir,
        cfg.paths.outputs_dir,
        cfg.paths.tables_dir,
        cfg.paths.figures_dir,
        cfg.paths.logs_dir,
        cfg.paths.cache_dir,
        cfg.paths.extracted_by_file_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> None:
    make_dirs(cfg)
    log_file = cfg.paths.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging to %s", log_file)


def list_raw_files(cfg: Config) -> list[Path]:
    files = sorted(cfg.paths.raw_posts_dir.glob(cfg.file_glob))
    if cfg.max_files is not None:
        files = files[: int(cfg.max_files)]
    return files


def col(cfg: Config, logical_name: str) -> str:
    return cfg.columns.get(logical_name, logical_name)


def is_nullish_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in NULL_STRINGS
    return False


def nullish_series(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().str.lower().isin(NULL_STRINGS)


def safe_text(value: Any) -> str:
    if is_nullish_scalar(value):
        return ""
    return str(value)


def flatten_strings(value: Any) -> list[str]:
    """Turn strings, lists, dicts, numpy arrays, Arrow scalars, or null into strings."""
    if is_nullish_scalar(value):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(flatten_strings(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for v in value:
            out.extend(flatten_strings(v))
        return out
    if hasattr(value, "tolist"):
        return flatten_strings(value.tolist())
    return [str(value)]


def facets_to_search_string(value: Any) -> str:
    return " ".join(flatten_strings(value))


def clean_url(url: str) -> str:
    return url.strip().rstrip(TRAILING_PUNCT)


def extract_urls_from_any(value: Any) -> list[str]:
    urls: list[str] = []
    for s in flatten_strings(value):
        urls.extend(clean_url(m.group(0)) for m in URL_RE.finditer(s))
    return list(dict.fromkeys(urls))


def normalize_doi(raw_doi: str) -> str:
    doi = str(raw_doi).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = unquote(doi)
    # URL tracking/query/fragment is not DOI content.
    doi = doi.split("?")[0].split("#")[0]
    # Visible Bluesky text may end links with ellipses.
    doi = doi.replace("…", "...")
    doi = doi.rstrip(TRAILING_PUNCT)
    doi = doi.rstrip(".")
    return doi.lower()


def doi_suffix(doi: str) -> str:
    if "/" not in doi:
        return ""
    return doi.split("/", 1)[1]


def is_obviously_incomplete_doi(doi: str) -> bool:
    """Reject DOI fragments created by shortened display URLs like 10.1126/ or 10.1017/S014..."""
    if not doi.startswith("10.") or "/" not in doi:
        return True
    suffix = doi_suffix(doi)
    if len(suffix) < 4:
        return True
    if suffix in {"full", "doi"}:
        return True
    return False


def extract_dois_from_any(value: Any) -> list[str]:
    dois: list[str] = []
    for s in flatten_strings(value):
        for m in DOI_RE.finditer(s):
            doi = normalize_doi(m.group(1))
            if doi.startswith("10.") and "/" in doi and not is_obviously_incomplete_doi(doi):
                dois.append(doi)
    return list(dict.fromkeys(dois))


def drop_text_doi_prefixes(text_dois: list[str], facet_dois: list[str]) -> list[str]:
    """If text DOI is a prefix of a fuller facets DOI, drop the text DOI.

    Example:
      text: 10.1126
      facets: 10.1126/sciadv.adu8437

    More commonly after regex:
      text: 10.1017/s014
      facets: 10.1017/s0140525x25000056
    """
    out = []
    for d in text_dois:
        is_prefix = any(fd != d and fd.startswith(d.rstrip("/")) for fd in facet_dois)
        if not is_prefix:
            out.append(d)
    return out


def get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split("@")[(-1)]
        host = host.split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def domain_matches(host: str, configured_domain: str) -> bool:
    d = configured_domain.lower().lstrip("www.")
    return host == d or host.endswith("." + d)


def matched_publication_domain(url: str, domains: list[str]) -> str | None:
    host = get_domain(url)
    if not host:
        return None
    matches = [d for d in domains if domain_matches(host, d)]
    if not matches:
        return None
    return sorted(matches, key=len, reverse=True)[0]


def classify_publication_source(domain: str, url: str) -> str:
    preprints = {
        "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "osf.io", "preprints.org", "researchsquare.com"
    }
    libraries = {
        "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "dl.acm.org", "ieeexplore.ieee.org", "aclanthology.org", "proceedings.mlr.press"
    }
    if domain in preprints:
        return "preprint_repository"
    if domain in libraries:
        return "index_or_digital_library"
    if "doi" in url.lower() or extract_dois_from_any(url):
        return "publisher_link_with_doi_path"
    return "publisher_or_journal_link"


def bsky_url_from_at_uri(uri: str) -> str | None:
    """Convert at://did/app.bsky.feed.post/rkey to bsky.app URL."""
    if not isinstance(uri, str) or not uri.startswith("at://"):
        return None
    parts = uri.replace("at://", "", 1).split("/")
    if len(parts) >= 3 and parts[1] == "app.bsky.feed.post":
        return f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}"
    return None


def read_needed_columns(path: Path, cfg: Config) -> pd.DataFrame:
    needed = [
        col(cfg, "uri"), col(cfg, "did"), col(cfg, "operation"), col(cfg, "rkey"),
        col(cfg, "created_at"), col(cfg, "text"), col(cfg, "langs"),
        col(cfg, "reply_parent_uri"), col(cfg, "reply_root_uri"), col(cfg, "facets_uris"),
    ]
    needed_unique = list(dict.fromkeys([c for c in needed if c]))
    try:
        return pd.read_parquet(path, columns=needed_unique)
    except Exception:
        return pd.read_parquet(path)


def select_existing_columns(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    return [n for n in names if n in df.columns]


def write_parquet_and_csv_preview(df: pd.DataFrame, parquet_path: Path, preview_path: Path | None = None, n: int = 50) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = df.copy()
    if to_write.empty and len(to_write.columns) == 0:
        to_write = pd.DataFrame({"_empty": pd.Series(dtype="string")})
    to_write.to_parquet(parquet_path, index=False)
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        to_write.head(n).to_csv(preview_path, index=False, encoding="utf-8-sig")


def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if list(df.columns) == ["_empty"] and df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def filter_top_level_posts_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep created top-level app.bsky.feed.post records only."""
    if df.empty:
        return df
    out = df.copy()

    uri_col = col(cfg, "uri")
    if uri_col in out.columns:
        out = out[out[uri_col].astype(str).str.contains("/app.bsky.feed.post/", regex=False, na=False)].copy()

    operation_col = col(cfg, "operation")
    if cfg.operation_filter and operation_col in out.columns:
        out = out[out[operation_col].astype(str).isin(cfg.operation_filter)].copy()

    if cfg.top_level_only:
        parent_col = col(cfg, "reply_parent_uri")
        root_col = col(cfg, "reply_root_uri")
        if parent_col in out.columns:
            out = out[nullish_series(out[parent_col])].copy()
        if root_col in out.columns:
            out = out[nullish_series(out[root_col])].copy()

    return out


def filter_top_level_processed_df(df: pd.DataFrame) -> pd.DataFrame:
    """For processed/interim files, column names are standardized."""
    if df.empty:
        return df
    out = df.copy()
    if "post_uri" in out.columns:
        out = out[out["post_uri"].astype(str).str.contains("/app.bsky.feed.post/", regex=False, na=False)].copy()
    if "reply_parent_uri" in out.columns:
        out = out[nullish_series(out["reply_parent_uri"])].copy()
    if "reply_root_uri" in out.columns:
        out = out[nullish_series(out["reply_root_uri"])].copy()
    return out


# =============================================================================
# [0] Inspect raw data
# =============================================================================

def inspect_raw_data(cfg: Config) -> None:
    logging.info("[0] Inspect raw data")
    if not cfg.paths.raw_posts_dir.exists():
        raise FileNotFoundError(f"Raw posts folder does not exist: {cfg.paths.raw_posts_dir}")
    files = list_raw_files(cfg)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {cfg.paths.raw_posts_dir} with pattern {cfg.file_glob}")

    first_file = files[0]
    sample = pd.read_parquet(first_file)
    top_sample = filter_top_level_posts_df(sample, cfg)

    report_lines = [
        "# Raw Bluesky Data Inspection",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Raw folder: `{cfg.paths.raw_posts_dir}`",
        f"File pattern: `{cfg.file_glob}`",
        f"Number of files used by this run: **{len(files)}**",
        f"First file: `{first_file.name}`",
        f"First file shape: **{sample.shape[0]} rows × {sample.shape[1]} columns**",
        f"Top-level rows in first file after filter: **{len(top_sample)}**",
        "",
        "## Filter used for real original posts",
        "",
        "- URI contains `app.bsky.feed.post`",
        f"- operation in `{cfg.operation_filter}`" if cfg.operation_filter else "- no operation filter",
        "- `reply_parent_uri` is null/empty",
        "- `reply_root_uri` is null/empty",
        "",
        "## Columns",
        "",
    ]
    for c in sample.columns:
        report_lines.append(f"- `{c}`")

    preview_cols = select_existing_columns(sample, [
        col(cfg, "uri"), col(cfg, "did"), col(cfg, "created_at"), col(cfg, "text"), col(cfg, "reply_parent_uri"), col(cfg, "reply_root_uri"), col(cfg, "facets_uris"),
    ])
    report_lines.extend(["", "## Sample rows", ""])
    try:
        report_lines.append(sample[preview_cols].head(cfg.sample_rows).to_markdown(index=False))
    except Exception:
        report_lines.append(sample[preview_cols].head(cfg.sample_rows).to_string(index=False))

    out_md = cfg.paths.tables_dir / "00_raw_data_inspection.md"
    out_csv = cfg.paths.tables_dir / "00_raw_data_sample.csv"
    out_md.write_text("\n".join(report_lines), encoding="utf-8")
    sample[preview_cols].head(100).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info("Raw inspection written to %s", out_md)


# =============================================================================
# [1] Extract candidates
# =============================================================================

def candidate_prefilter_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    text_col = col(cfg, "text")
    facets_col = col(cfg, "facets_uris")

    text_s = df[text_col].fillna("").astype(str) if text_col in df.columns else pd.Series([""] * len(df), index=df.index)
    facets_s = df[facets_col].map(facets_to_search_string) if facets_col in df.columns else pd.Series([""] * len(df), index=df.index)

    terms = ["10.", "doi.org", "/doi/", "doi:"] + cfg.publication_domains
    mask = pd.Series(False, index=df.index)
    for t in terms:
        mask = mask | text_s.str.contains(t, case=False, regex=False, na=False) | facets_s.str.contains(t, case=False, regex=False, na=False)
    return mask


def extract_from_file(path: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = read_needed_columns(path, cfg)
    raw_rows = len(df)
    df = filter_top_level_posts_df(df, cfg)
    top_level_rows = len(df)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"raw_rows": raw_rows, "top_level_rows": top_level_rows, "candidate_rows": 0}

    mask = candidate_prefilter_mask(df, cfg)
    df = df[mask].copy()
    candidate_rows = len(df)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"raw_rows": raw_rows, "top_level_rows": top_level_rows, "candidate_rows": 0}

    uri_col = col(cfg, "uri")
    did_col = col(cfg, "did")
    created_col = col(cfg, "created_at")
    text_col = col(cfg, "text")
    langs_col = col(cfg, "langs")
    parent_col = col(cfg, "reply_parent_uri")
    root_col = col(cfg, "reply_root_uri")
    facets_col = col(cfg, "facets_uris")

    doi_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        text_value = row.get(text_col, "")
        facets_value = row.get(facets_col, "")

        text = safe_text(text_value)
        facets_strings = flatten_strings(facets_value)

        dois_from_facets = extract_dois_from_any(facets_strings)
        dois_from_text_raw = extract_dois_from_any(text)
        dois_from_text = drop_text_doi_prefixes(dois_from_text_raw, dois_from_facets)

        # Prefer facets DOI first; keep text as fallback. Exact duplicates merge below.
        doi_source: dict[str, set[str]] = {}
        for d in dois_from_facets:
            doi_source.setdefault(d, set()).add("facets_uris")
        for d in dois_from_text:
            doi_source.setdefault(d, set()).add("text")
        all_dois = list(doi_source.keys())

        urls = list(dict.fromkeys(extract_urls_from_any(text) + extract_urls_from_any(facets_strings)))

        post_uri = row.get(uri_col, None)
        base = {
            "post_uri": post_uri,
            "bsky_url": bsky_url_from_at_uri(str(post_uri)) if post_uri is not None else None,
            "user_did": row.get(did_col, None),
            "created_at": row.get(created_col, None),
            "text": text,
            "langs": json.dumps(row.get(langs_col, None), ensure_ascii=False, default=str) if langs_col in row else None,
            "reply_parent_uri": row.get(parent_col, None),
            "reply_root_uri": row.get(root_col, None),
            "is_top_level_post": True,
            "raw_file": path.name,
        }

        for doi in all_dois:
            doi_rows.append({**base, "doi": doi, "doi_source_field": "+".join(sorted(doi_source.get(doi, set())))})

        for url in urls:
            domain = matched_publication_domain(url, cfg.publication_domains)
            if domain is None:
                continue
            extracted_url_dois = extract_dois_from_any(url)
            link_rows.append({
                **base,
                "matched_url": url,
                "matched_domain": domain,
                "source_type": classify_publication_source(domain, url),
                "url_contains_doi": bool(extracted_url_dois),
                "url_doi_values": ";".join(extracted_url_dois) if extracted_url_dois else None,
            })

    doi_df = pd.DataFrame(doi_rows).drop_duplicates() if doi_rows else pd.DataFrame()
    link_df = pd.DataFrame(link_rows).drop_duplicates() if link_rows else pd.DataFrame()
    stats = {"raw_rows": raw_rows, "top_level_rows": top_level_rows, "candidate_rows": candidate_rows}
    return doi_df, link_df, stats


def checkpoint_paths(path: Path, cfg: Config) -> tuple[Path, Path, Path]:
    safe_stem = path.stem.replace(".", "_")
    doi_checkpoint = cfg.paths.extracted_by_file_dir / f"{safe_stem}.doi.parquet"
    link_checkpoint = cfg.paths.extracted_by_file_dir / f"{safe_stem}.publication_links.parquet"
    stats_checkpoint = cfg.paths.extracted_by_file_dir / f"{safe_stem}.stats.json"
    return doi_checkpoint, link_checkpoint, stats_checkpoint


def extract_file_with_checkpoint(path_str: str, cfg: Config) -> dict[str, Any]:
    path = Path(path_str)
    doi_checkpoint, link_checkpoint, stats_checkpoint = checkpoint_paths(path, cfg)

    if doi_checkpoint.exists() and link_checkpoint.exists():
        doi_df = load_parquet_if_exists(doi_checkpoint)
        link_df = load_parquet_if_exists(link_checkpoint)
        return {
            "file": path.name,
            "status": "cached",
            "doi_rows": len(doi_df),
            "link_rows": len(link_df),
            "raw_rows": None,
            "top_level_rows": None,
            "candidate_rows": None,
        }

    doi_df, link_df, stats = extract_from_file(path, cfg)
    write_parquet_and_csv_preview(doi_df, doi_checkpoint)
    write_parquet_and_csv_preview(link_df, link_checkpoint)
    stats_checkpoint.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "file": path.name,
        "status": "processed",
        "doi_rows": len(doi_df),
        "link_rows": len(link_df),
        **stats,
    }


def combine_extraction_checkpoints(cfg: Config) -> None:
    doi_files = sorted(cfg.paths.extracted_by_file_dir.glob("*.doi.parquet"))
    link_files = sorted(cfg.paths.extracted_by_file_dir.glob("*.publication_links.parquet"))

    doi_parts = []
    for p in tqdm(doi_files, desc="Combining DOI checkpoints"):
        df = load_parquet_if_exists(p)
        if not df.empty:
            doi_parts.append(df)

    link_parts = []
    for p in tqdm(link_files, desc="Combining publication-link checkpoints"):
        df = load_parquet_if_exists(p)
        if not df.empty:
            link_parts.append(df)

    doi_all = pd.concat(doi_parts, ignore_index=True).drop_duplicates() if doi_parts else pd.DataFrame()
    link_all = pd.concat(link_parts, ignore_index=True).drop_duplicates() if link_parts else pd.DataFrame()

    # Safety: final combined interim files must be top-level only, even if old checkpoints were produced before this rule.
    doi_all = filter_top_level_processed_df(doi_all)
    link_all = filter_top_level_processed_df(link_all)

    write_parquet_and_csv_preview(doi_all, cfg.paths.interim_dir / "doi_post_pairs.parquet", cfg.paths.tables_dir / "01_preview_doi_post_pairs.csv")
    write_parquet_and_csv_preview(link_all, cfg.paths.interim_dir / "publication_link_candidates.parquet", cfg.paths.tables_dir / "01_preview_publication_link_candidates.csv")

    logging.info("Extracted top-level DOI candidate rows: %s", len(doi_all))
    logging.info("Extracted top-level publication-link candidate rows: %s", len(link_all))


def extract_candidates(cfg: Config) -> None:
    logging.info("[1] Extract candidates: top-level posts only")
    files = list_raw_files(cfg)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {cfg.paths.raw_posts_dir}")

    logging.info("Extraction workers: %s", cfg.workers)
    logging.info("Top-level only: %s", cfg.top_level_only)

    if cfg.workers <= 1:
        for path in tqdm(files, desc="Extracting candidates", unit="file"):
            try:
                result = extract_file_with_checkpoint(str(path), cfg)
                logging.info("Finished %s | status=%s | DOI rows=%s | publication-link rows=%s", result["file"], result["status"], result["doi_rows"], result["link_rows"])
            except Exception as e:
                logging.exception("Failed to process %s: %s", path, e)
    else:
        logging.info("Parallel extraction enabled. If RAM gets too high, rerun with --workers 1.")
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {ex.submit(extract_file_with_checkpoint, str(path), cfg): path for path in files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting candidates", unit="file"):
                path = futures[fut]
                try:
                    result = fut.result()
                    logging.info("Finished %s | status=%s | DOI rows=%s | publication-link rows=%s", result["file"], result["status"], result["doi_rows"], result["link_rows"])
                except Exception as e:
                    logging.exception("Failed to process %s: %s", path, e)

    combine_extraction_checkpoints(cfg)


# =============================================================================
# [2] OpenAlex DOI validation
# =============================================================================

def openalex_cache_path(cfg: Config) -> Path:
    return cfg.paths.cache_dir / "openalex_metadata_cache.parquet"


def normalize_openalex_doi(value: Any) -> str | None:
    if is_nullish_scalar(value):
        return None
    d = normalize_doi(str(value))
    return d if d else None


def compact_openalex_work(work: dict[str, Any], requested_doi: str) -> dict[str, Any]:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    topics = work.get("topics") or []
    top_topics = []
    for t in topics[:3]:
        display = t.get("display_name") if isinstance(t, dict) else None
        if display:
            top_topics.append(display)

    returned_doi = normalize_openalex_doi(work.get("doi")) or requested_doi
    return {
        "doi": returned_doi,
        "requested_doi": requested_doi,
        "openalex_found": True,
        "openalex_id": work.get("id"),
        "paper_title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "work_type": work.get("type"),
        "venue_name": source.get("display_name"),
        "venue_type": source.get("type"),
        "is_oa": open_access.get("is_oa"),
        "cited_by_count": work.get("cited_by_count"),
        "topics_top3": "; ".join(top_topics),
        "openalex_checked_at": datetime.now().isoformat(timespec="seconds"),
        "openalex_error": None,
    }


def missing_openalex_record(doi: str, error: str | None = None) -> dict[str, Any]:
    return {
        "doi": doi,
        "requested_doi": doi,
        "openalex_found": False,
        "openalex_id": None,
        "paper_title": None,
        "publication_year": None,
        "work_type": None,
        "venue_name": None,
        "venue_type": None,
        "is_oa": None,
        "cited_by_count": None,
        "topics_top3": None,
        "openalex_checked_at": datetime.now().isoformat(timespec="seconds"),
        "openalex_error": error,
    }


def fetch_openalex_batch(dois: list[str], cfg: Config) -> list[dict[str, Any]]:
    if not cfg.openalex_api_key:
        return [missing_openalex_record(d, "missing_openalex_api_key") for d in dois]

    url = "https://api.openalex.org/works"
    params = {
        "filter": "doi:" + "|".join(dois),
        "per_page": 100,
        "select": "id,doi,display_name,publication_year,type,primary_location,open_access,cited_by_count,topics",
        "api_key": cfg.openalex_api_key,
    }

    last_error = None
    for attempt in range(cfg.openalex_max_retries):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 429:
                sleep_for = min(60, 2 ** attempt + random.random())
                logging.warning("OpenAlex rate limit hit. Sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            payload = r.json()
            results = payload.get("results", [])
            found_by_doi: dict[str, dict[str, Any]] = {}
            for work in results:
                returned_doi = normalize_openalex_doi(work.get("doi"))
                if returned_doi:
                    found_by_doi[returned_doi] = compact_openalex_work(work, returned_doi)
            return [found_by_doi.get(d, missing_openalex_record(d, None)) for d in dois]
        except Exception as e:
            last_error = str(e)
            sleep_for = min(60, 2 ** attempt + random.random())
            logging.warning("OpenAlex request failed attempt %s/%s: %s", attempt + 1, cfg.openalex_max_retries, e)
            time.sleep(sleep_for)
    return [missing_openalex_record(d, last_error) for d in dois]


def validate_dois_with_openalex(cfg: Config) -> None:
    logging.info("[2] Validate unique DOIs with OpenAlex")
    doi_path = cfg.paths.interim_dir / "doi_post_pairs.parquet"
    if not doi_path.exists():
        logging.info("DOI interim file missing; running extraction first.")
        extract_candidates(cfg)

    doi_df = load_parquet_if_exists(doi_path)
    doi_df = filter_top_level_processed_df(doi_df)
    if doi_df.empty:
        logging.warning("No DOI candidates found.")
        write_parquet_and_csv_preview(pd.DataFrame(), cfg.paths.interim_dir / "openalex_metadata.parquet")
        return

    unique_dois = sorted({normalize_doi(d) for d in doi_df["doi"].dropna().astype(str) if not is_obviously_incomplete_doi(normalize_doi(d))})
    logging.info("Unique top-level DOI candidates: %s", len(unique_dois))

    cache_file = openalex_cache_path(cfg)
    cache_df = load_parquet_if_exists(cache_file)
    cached_dois = set(cache_df["doi"].dropna().astype(str)) if not cache_df.empty and "doi" in cache_df.columns else set()
    to_fetch = [d for d in unique_dois if d not in cached_dois]

    if not cfg.openalex_api_key:
        logging.warning("No OpenAlex API key found. DOI rows will be marked as not validated. Add api_key in config.yaml or OPENALEX_API_KEY.")

    new_records: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(to_fetch), cfg.openalex_batch_size), desc="Validating DOI batches with OpenAlex"):
        batch = to_fetch[i : i + cfg.openalex_batch_size]
        if not batch:
            continue
        records = fetch_openalex_batch(batch, cfg)
        new_records.extend(records)

        batch_df = pd.DataFrame(new_records)
        combined = pd.concat([cache_df, batch_df], ignore_index=True) if not cache_df.empty else batch_df
        combined = combined.drop_duplicates(subset=["doi"], keep="last") if not combined.empty and "doi" in combined.columns else combined
        write_parquet_and_csv_preview(combined, cache_file)
        time.sleep(cfg.openalex_sleep_seconds)

    final_cache = load_parquet_if_exists(cache_file)
    if final_cache.empty and new_records:
        final_cache = pd.DataFrame(new_records)
    if not final_cache.empty and "doi" in final_cache.columns:
        final_cache = final_cache[final_cache["doi"].isin(unique_dois)].drop_duplicates(subset=["doi"], keep="last")

    write_parquet_and_csv_preview(final_cache, cfg.paths.interim_dir / "openalex_metadata.parquet", cfg.paths.tables_dir / "02_preview_openalex_metadata.csv")
    logging.info("Saved OpenAlex metadata rows: %s", len(final_cache))


# =============================================================================
# [3] Build final research datasets
# =============================================================================

def build_final_datasets(cfg: Config) -> None:
    logging.info("[3] Build final research datasets: top-level posts only")
    doi_path = cfg.paths.interim_dir / "doi_post_pairs.parquet"
    link_path = cfg.paths.interim_dir / "publication_link_candidates.parquet"
    meta_path = cfg.paths.interim_dir / "openalex_metadata.parquet"

    if not doi_path.exists() or not link_path.exists():
        extract_candidates(cfg)
    if not meta_path.exists():
        validate_dois_with_openalex(cfg)

    doi_df = filter_top_level_processed_df(load_parquet_if_exists(doi_path))
    link_df = filter_top_level_processed_df(load_parquet_if_exists(link_path))
    meta_df = load_parquet_if_exists(meta_path)

    if not doi_df.empty:
        doi_df["doi"] = doi_df["doi"].astype(str).map(normalize_doi)
        doi_df = doi_df[~doi_df["doi"].map(is_obviously_incomplete_doi)].drop_duplicates()
    if not meta_df.empty and "doi" in meta_df.columns:
        meta_df["doi"] = meta_df["doi"].astype(str).map(normalize_doi)

    if not doi_df.empty and not meta_df.empty:
        doi_enriched = doi_df.merge(meta_df, on="doi", how="left", suffixes=("", "_openalex"))
    else:
        doi_enriched = doi_df.copy()
        if not doi_enriched.empty and "openalex_found" not in doi_enriched.columns:
            doi_enriched["openalex_found"] = False

    # Save all DOI candidates with metadata status.
    if not doi_enriched.empty:
        doi_enriched["evidence_type"] = "doi_candidate"
        doi_enriched["confidence_layer"] = "candidate_until_openalex_validated"
    write_parquet_and_csv_preview(
        doi_enriched,
        cfg.paths.processed_dir / "doi_candidate_post_pairs_all.parquet",
        cfg.paths.tables_dir / "03_preview_doi_candidate_post_pairs_all.csv",
    )

    # Strict high-confidence DOI layer.
    if not doi_enriched.empty and "openalex_found" in doi_enriched.columns:
        doi_confirmed = doi_enriched[doi_enriched["openalex_found"] == True].copy()  # noqa: E712
    else:
        doi_confirmed = pd.DataFrame()

    if not doi_confirmed.empty:
        doi_confirmed["evidence_type"] = "doi_openalex_validated"
        doi_confirmed["confidence_layer"] = "high"
        doi_confirmed["is_top_level_post"] = True

    if not doi_confirmed.empty:
        sort_cols = [c for c in ["post_uri", "created_at", "doi"] if c in doi_confirmed.columns]
        unique_posts = doi_confirmed.sort_values(sort_cols).drop_duplicates(subset=["post_uri"], keep="first")
    else:
        unique_posts = pd.DataFrame()

    if not link_df.empty:
        link_df = link_df.drop_duplicates().copy()
        link_df["evidence_type"] = "publication_link_candidate"
        link_df["confidence_layer"] = "medium_needs_manual_check"
        link_df["is_top_level_post"] = True

    combined_parts = []
    if not doi_confirmed.empty:
        doi_combined = doi_confirmed.copy()
        if "matched_url" not in doi_combined.columns:
            doi_combined["matched_url"] = None
        if "matched_domain" not in doi_combined.columns:
            doi_combined["matched_domain"] = None
        if "source_type" not in doi_combined.columns:
            doi_combined["source_type"] = "doi_validated_by_openalex"
        combined_parts.append(doi_combined)
    if not link_df.empty:
        link_combined = link_df.copy()
        for c in ["doi", "openalex_found", "openalex_id", "paper_title", "publication_year", "venue_name"]:
            if c not in link_combined.columns:
                link_combined[c] = None
        combined_parts.append(link_combined)

    combined = pd.concat(combined_parts, ignore_index=True, sort=False).drop_duplicates() if combined_parts else pd.DataFrame()

    write_parquet_and_csv_preview(doi_confirmed, cfg.paths.processed_dir / "science_post_doi_pairs.parquet", cfg.paths.tables_dir / "03_preview_science_post_doi_pairs.csv")
    write_parquet_and_csv_preview(unique_posts, cfg.paths.processed_dir / "science_posts_unique.parquet", cfg.paths.tables_dir / "03_preview_science_posts_unique.csv")
    write_parquet_and_csv_preview(link_df, cfg.paths.processed_dir / "publication_link_candidate_posts.parquet", cfg.paths.tables_dir / "03_preview_publication_link_candidate_posts.csv")
    write_parquet_and_csv_preview(combined, cfg.paths.processed_dir / "paper_related_candidates_combined.parquet", cfg.paths.tables_dir / "03_preview_paper_related_candidates_combined.csv")

    logging.info("All top-level DOI candidate post-DOI pairs: %s", len(doi_enriched))
    logging.info("DOI-confirmed top-level science post-DOI pairs: %s", len(doi_confirmed))
    logging.info("Unique DOI-confirmed top-level science posts: %s", len(unique_posts))
    logging.info("Publication-link top-level candidate rows: %s", len(link_df))
    logging.info("Combined top-level paper-related candidate rows: %s", len(combined))


# =============================================================================
# [3.5] Hydrate Bluesky engagement counts
# =============================================================================

def engagement_cache_path(cfg: Config) -> Path:
    return cfg.paths.cache_dir / "bluesky_engagement_cache.parquet"


def missing_engagement_record(uri: str, status: str = "missing") -> dict[str, Any]:
    return {
        "post_uri": uri,
        "hydration_status": status,
        "like_count": None,
        "reply_count": None,
        "repost_count": None,
        "quote_count": None,
        "bookmark_count": None,
        "engagement_total": None,
        "post_cid": None,
        "indexed_at": None,
        "author_handle": None,
        "author_display_name": None,
        "engagement_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def compact_bsky_post_view(post: dict[str, Any]) -> dict[str, Any]:
    author = post.get("author") or {}
    like_count = post.get("likeCount")
    reply_count = post.get("replyCount")
    repost_count = post.get("repostCount")
    quote_count = post.get("quoteCount")
    bookmark_count = post.get("bookmarkCount")

    numeric = [like_count, reply_count, repost_count, quote_count, bookmark_count]
    total = sum(int(x) for x in numeric if isinstance(x, int))

    return {
        "post_uri": post.get("uri"),
        "hydration_status": "found",
        "like_count": like_count,
        "reply_count": reply_count,
        "repost_count": repost_count,
        "quote_count": quote_count,
        "bookmark_count": bookmark_count,
        "engagement_total": total,
        "post_cid": post.get("cid"),
        "indexed_at": post.get("indexedAt"),
        "author_handle": author.get("handle"),
        "author_display_name": author.get("displayName"),
        "engagement_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def fetch_bsky_engagement_batch(uris: list[str], cfg: Config) -> list[dict[str, Any]]:
    endpoint = f"{cfg.bluesky_api_base}/xrpc/app.bsky.feed.getPosts"
    params = [("uris", u) for u in uris]
    last_error = None

    for attempt in range(cfg.bluesky_max_retries):
        try:
            r = requests.get(endpoint, params=params, timeout=40)
            if r.status_code == 429:
                sleep_for = min(60, 2 ** attempt + random.random())
                logging.warning("Bluesky rate limit hit. Sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            payload = r.json()
            posts = payload.get("posts", [])
            found = {p.get("uri"): compact_bsky_post_view(p) for p in posts if p.get("uri")}
            return [found.get(u, missing_engagement_record(u, "not_found_or_deleted")) for u in uris]
        except Exception as e:
            last_error = str(e)
            sleep_for = min(60, 2 ** attempt + random.random())
            logging.warning("Bluesky engagement request failed attempt %s/%s: %s", attempt + 1, cfg.bluesky_max_retries, e)
            time.sleep(sleep_for)

    return [missing_engagement_record(u, f"error:{last_error}") for u in uris]


def collect_post_uris_for_engagement(cfg: Config) -> list[str]:
    paths = [
        cfg.paths.processed_dir / "science_posts_unique.parquet",
        cfg.paths.processed_dir / "science_post_doi_pairs.parquet",
        cfg.paths.processed_dir / "publication_link_candidate_posts.parquet",
        cfg.paths.processed_dir / "paper_related_candidates_combined.parquet",
    ]
    uris: set[str] = set()
    for p in paths:
        df = load_parquet_if_exists(p)
        df = filter_top_level_processed_df(df)
        if not df.empty and "post_uri" in df.columns:
            uris.update(df["post_uri"].dropna().astype(str).tolist())
    return sorted(uris)


def hydrate_bluesky_engagement(cfg: Config) -> None:
    logging.info("[3.5] Hydrate Bluesky engagement counts")
    if not (cfg.paths.processed_dir / "science_posts_unique.parquet").exists():
        logging.info("Processed datasets missing; running build first.")
        build_final_datasets(cfg)

    uris = collect_post_uris_for_engagement(cfg)
    logging.info("Unique top-level post URIs to hydrate: %s", len(uris))
    if not uris:
        write_parquet_and_csv_preview(pd.DataFrame(), cfg.paths.processed_dir / "bluesky_engagement_counts.parquet")
        return

    cache_file = engagement_cache_path(cfg)
    cache_df = load_parquet_if_exists(cache_file)
    cached_uris = set(cache_df["post_uri"].dropna().astype(str)) if not cache_df.empty and "post_uri" in cache_df.columns else set()
    to_fetch = [u for u in uris if u not in cached_uris]
    logging.info("Already cached engagement rows: %s", len(cached_uris))
    logging.info("Post URIs remaining to hydrate: %s", len(to_fetch))

    new_records: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(to_fetch), cfg.bluesky_batch_size), desc="Hydrating Bluesky engagement batches"):
        batch = to_fetch[i : i + cfg.bluesky_batch_size]
        if not batch:
            continue
        records = fetch_bsky_engagement_batch(batch, cfg)
        new_records.extend(records)

        batch_df = pd.DataFrame(new_records)
        combined = pd.concat([cache_df, batch_df], ignore_index=True) if not cache_df.empty else batch_df
        combined = combined.drop_duplicates(subset=["post_uri"], keep="last") if not combined.empty and "post_uri" in combined.columns else combined
        write_parquet_and_csv_preview(combined, cache_file)
        time.sleep(cfg.bluesky_sleep_seconds)

    final_cache = load_parquet_if_exists(cache_file)
    if final_cache.empty and new_records:
        final_cache = pd.DataFrame(new_records)
    if not final_cache.empty and "post_uri" in final_cache.columns:
        final_cache = final_cache[final_cache["post_uri"].isin(uris)].drop_duplicates(subset=["post_uri"], keep="last")

    engagement_out = cfg.paths.processed_dir / "bluesky_engagement_counts.parquet"
    write_parquet_and_csv_preview(final_cache, engagement_out, cfg.paths.tables_dir / "03_5_preview_bluesky_engagement_counts.csv")

    # Merge engagement counts back into final processed datasets.
    merge_engagement_into_processed_files(cfg, final_cache)
    logging.info("Saved engagement rows: %s", len(final_cache))


def merge_engagement_into_processed_files(cfg: Config, engagement_df: pd.DataFrame | None = None) -> None:
    if engagement_df is None:
        engagement_df = load_parquet_if_exists(cfg.paths.processed_dir / "bluesky_engagement_counts.parquet")
    if engagement_df.empty or "post_uri" not in engagement_df.columns:
        logging.warning("No engagement data to merge.")
        return

    files = [
        "doi_candidate_post_pairs_all.parquet",
        "science_post_doi_pairs.parquet",
        "science_posts_unique.parquet",
        "publication_link_candidate_posts.parquet",
        "paper_related_candidates_combined.parquet",
    ]
    engagement_cols = [
        "post_uri", "hydration_status", "like_count", "reply_count", "repost_count", "quote_count", "bookmark_count", "engagement_total", "post_cid", "indexed_at", "author_handle", "author_display_name", "engagement_fetched_at",
    ]
    engagement_small = engagement_df[[c for c in engagement_cols if c in engagement_df.columns]].drop_duplicates(subset=["post_uri"], keep="last")

    for fname in files:
        p = cfg.paths.processed_dir / fname
        df = load_parquet_if_exists(p)
        if df.empty or "post_uri" not in df.columns:
            continue
        # Remove old engagement columns before re-merge.
        drop_cols = [c for c in engagement_small.columns if c != "post_uri" and c in df.columns]
        df = df.drop(columns=drop_cols) if drop_cols else df
        df = df.merge(engagement_small, on="post_uri", how="left")
        write_parquet_and_csv_preview(df, p, cfg.paths.tables_dir / f"engaged_preview_{fname.replace('.parquet', '.csv')}")
        logging.info("Merged engagement into %s", fname)

    copy_to_final_dataset(cfg)


def copy_to_final_dataset(cfg: Config) -> None:
    """Copy the two analysis-ready files from data/processed/ to final_dataset/."""
    final_files = [
        "science_posts_unique.parquet",
        "publication_link_candidate_posts.parquet",
    ]
    cfg.paths.final_dataset_dir.mkdir(parents=True, exist_ok=True)
    for fname in final_files:
        src = cfg.paths.processed_dir / fname
        dst = cfg.paths.final_dataset_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            logging.info("Copied %s → final_dataset/", fname)
        else:
            logging.warning("Could not copy %s — file not found in data/processed/", fname)


# =============================================================================
# Orchestration
# =============================================================================

def run_stage(stage: str, cfg: Config) -> None:
    if stage == "inspect":
        inspect_raw_data(cfg)
    elif stage == "extract":
        extract_candidates(cfg)
    elif stage == "validate":
        validate_dois_with_openalex(cfg)
    elif stage == "build":
        build_final_datasets(cfg)
    elif stage == "engage":
        hydrate_bluesky_engagement(cfg)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_all(cfg: Config) -> None:
    for stage in ["inspect", "extract", "validate", "build", "engage"]:
        run_stage(stage, cfg)


def fresh_reset(cfg: Config) -> None:
    """Delete generated outputs while keeping config.yaml and raw data untouched."""
    for p in [cfg.paths.data_dir, cfg.paths.outputs_dir, cfg.paths.logs_dir, cfg.paths.cache_dir]:
        if p.exists():
            shutil.rmtree(p)
    make_dirs(cfg)


def print_pipeline() -> None:
    print(
        """
Raw Bluesky parquet posts
        ↓
[0] Inspect raw data
        ↓
[1] Extract candidates from real original/top-level posts only
        ├── keep app.bsky.feed.post + operation=create
        ├── remove replies where reply_parent_uri/reply_root_uri are not null
        ├── DOI candidates from text + facets_uris
        └── publication-link candidates from article/preprint/publisher URLs
        ↓
[2] Validate unique DOIs with OpenAlex
        ↓
[3] Build final research datasets
        ├── DOI-confirmed science posts
        ├── unique post-level science posts
        └── combined paper-related candidates
        ↓
[3.5] Hydrate Bluesky engagement counts
        ├── like_count
        ├── reply_count
        ├── repost_count
        ├── quote_count
        ├── bookmark_count if available
        └── engagement_total
""".strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean one-file data collection/preprocessing pipeline for top-level science-related Bluesky posts with influence metrics.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "inspect", "extract", "validate", "build", "engage"],
        help="Run one stage only, or all stages. Default: all",
    )
    parser.add_argument("--workers", type=int, default=None, help="Override number of parallel extraction workers. Use 1 if memory is high.")
    parser.add_argument("--max-files", type=int, default=None, help="Override max files for testing, e.g., --max-files 2")
    parser.add_argument("--fresh", action="store_true", help="Delete generated outputs/cache and rerun cleanly. Raw parquet files are never touched.")
    parser.add_argument("--show-pipeline", action="store_true", help="Print the project pipeline and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_pipeline:
        print_pipeline()
        return

    cfg = load_config(Path(args.config), cli_workers=args.workers, cli_max_files=args.max_files)
    make_dirs(cfg)
    setup_logging(cfg)
    print_pipeline()

    if args.fresh:
        logging.warning("Fresh reset requested. Generated data/cache/output folders will be deleted. Raw parquet files are untouched.")
        fresh_reset(cfg)

    logging.info("Project root: %s", cfg.paths.root)
    logging.info("Raw posts folder: %s", cfg.paths.raw_posts_dir)
    logging.info("Stage: %s", args.stage)
    logging.info("Top-level only: %s", cfg.top_level_only)

    if args.stage == "all":
        run_all(cfg)
    else:
        run_stage(args.stage, cfg)

    logging.info("Done.")
    logging.info("Important outputs:")
    logging.info("- %s", cfg.paths.processed_dir / "science_post_doi_pairs.parquet")
    logging.info("- %s", cfg.paths.processed_dir / "science_posts_unique.parquet")
    logging.info("- %s", cfg.paths.processed_dir / "publication_link_candidate_posts.parquet")
    logging.info("- %s", cfg.paths.processed_dir / "paper_related_candidates_combined.parquet")
    logging.info("- %s", cfg.paths.processed_dir / "bluesky_engagement_counts.parquet")


if __name__ == "__main__":
    main()
