#!/usr/bin/env python3
"""
resolve_publication_links.py
============================
Stage 1.5 — Resolve DOIs from publication-domain URLs in candidate posts.

Reads publication_link_candidate_posts.parquet (posts containing publisher URLs
but no explicit doi.org link), extracts DOIs using two methods, validates with
OpenAlex, and merges confirmed posts into science_posts_unique.parquet.

Design principles:
  - Idempotent: safe to interrupt and rerun at any point
  - Incremental: all caches and checkpoints preserved between runs
  - Deduplicated: each unique URL fetched at most once
  - Fast: requests.Session reuse, lxml parser, URL deduplication before fetching

Methods:
  Method 1 — URL pattern matching (instant, no HTTP request)
    Covers: arXiv, bioRxiv, medRxiv, Nature, Science, PNAS, PLoS, Frontiers,
            Springer, Wiley, Tandfonline, OUP, SAGE, ACS, ACM, Cambridge,
            BMJ, NEJM, Lancet, and a generic DOI-in-URL-path fallback.

  Method 2 — HTML page metadata fetch (requires beautifulsoup4 + lxml)
    Reads <meta name="citation_doi"> and related scholarly meta tags.
    Covers: PubMed, JAMA, ScienceDirect, Cell, RSC, IEEE, and any publisher
            using standard scholarly HTML metadata.

Inputs (already exist from run_pipeline.py):
  D:/sciencebluesky/final_dataset/publication_link_candidate_posts.parquet
  D:/sciencebluesky/final_dataset/science_posts_unique.parquet
  D:/sciencebluesky/cache/openalex_metadata_cache.parquet
  D:/sciencebluesky/cache/bluesky_engagement_cache.parquet

Outputs:
  D:/sciencebluesky/final_dataset/science_posts_unique.parquet  (updated)
  D:/sciencebluesky/cache/openalex_metadata_cache.parquet       (appended)
  D:/sciencebluesky/cache/bluesky_engagement_cache.parquet      (appended)
  D:/sciencebluesky/cache/resolved_publication_dois.parquet     (resolution log)
  D:/sciencebluesky/cache/page_fetch_checkpoint.parquet         (page fetch progress)

Usage:
  python resolve_publication_links.py --limit 500 --no-fetch-pages  # test
  python resolve_publication_links.py --no-fetch-pages              # Method 1 only
  python resolve_publication_links.py --page-delay 0.2              # Method 1+2 (recommended)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
import yaml
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import lxml  # noqa: F401
    HTML_PARSER = "lxml"
except ImportError:
    HTML_PARSER = "html.parser"


# =============================================================================
# Paths
# =============================================================================

BASE_DIR          = Path("D:/sciencebluesky")
FINAL_DATASET     = BASE_DIR / "final_dataset"
CACHE_DIR         = BASE_DIR / "cache"
LOGS_DIR          = BASE_DIR / "logs"
TABLES_DIR        = BASE_DIR / "outputs/tables"

CANDIDATE_FILE    = FINAL_DATASET / "publication_link_candidate_posts.parquet"
SCIENCE_POSTS     = FINAL_DATASET / "science_posts_unique.parquet"
OA_CACHE          = CACHE_DIR / "openalex_metadata_cache.parquet"
ENG_CACHE         = CACHE_DIR / "bluesky_engagement_cache.parquet"
RESOLVED_CACHE    = CACHE_DIR / "resolved_publication_dois.parquet"
PAGE_FETCH_CKPT   = CACHE_DIR / "page_fetch_checkpoint.parquet"

BLUESKY_API    = "https://public.api.bsky.app"
OPENALEX_API   = "https://api.openalex.org/works"
TRAILING_PUNCT = ".,;:!?)]}>'\""
NULL_STRINGS   = {"", "null", "none", "nan", "nat"}


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"resolve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("Log: %s", log_file)
    logging.info("HTML parser: %s", HTML_PARSER)


# =============================================================================
# Config
# =============================================================================

def load_openalex_key(config_path: Path) -> str:
    key = os.getenv("OPENALEX_API_KEY", "")
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            key = raw.get("openalex", {}).get("api_key", key) or key
        except Exception:
            pass
    return str(key).strip()


def load_pub_domains(config_path: Path) -> list[str]:
    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            domains = raw.get("publication_domains", [])
            if domains:
                return sorted({d.lower().strip() for d in domains if d})
        except Exception:
            pass
    return [
        "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "osf.io",
        "preprints.org", "researchsquare.com", "nature.com", "science.org",
        "cell.com", "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com",
        "plos.org", "frontiersin.org", "mdpi.com", "sciencedirect.com",
        "springer.com", "link.springer.com", "wiley.com",
        "onlinelibrary.wiley.com", "tandfonline.com", "oup.com",
        "academic.oup.com", "cambridge.org", "sagepub.com", "acs.org",
        "pubs.acs.org", "rsc.org", "ieee.org", "ieeexplore.ieee.org",
        "acm.org", "dl.acm.org", "jmlr.org", "proceedings.mlr.press",
        "aclweb.org", "aclanthology.org", "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    ]


# =============================================================================
# Utilities
# =============================================================================

def is_nullish(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return isinstance(v, str) and v.strip().lower() in NULL_STRINGS


def normalize_doi(raw: str) -> str:
    doi = str(raw).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = unquote(doi).split("?")[0].split("#")[0]
    doi = doi.replace("\u2026", "...").rstrip(TRAILING_PUNCT).rstrip(".")
    return doi.lower()


def is_incomplete(doi: str) -> bool:
    if not doi.startswith("10.") or "/" not in doi:
        return True
    suffix = doi.split("/", 1)[1]
    return len(suffix) < 4 or suffix in {"full", "doi"}


def get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def is_pub_domain(url: str, domains: list[str]) -> bool:
    host = get_domain(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in domains)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        return pd.DataFrame() if list(df.columns) == ["_empty"] and df.empty else df
    except Exception as e:
        logging.warning("Cannot load %s: %s", path, e)
        return pd.DataFrame()


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = df.copy()
    if to_write.empty and not to_write.columns.tolist():
        to_write = pd.DataFrame({"_empty": pd.Series(dtype="string")})
    to_write.to_parquet(path, index=False)


def save_preview(df: pd.DataFrame, name: str) -> None:
    out = TABLES_DIR / f"resolve_{name}_preview.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.head(200).to_csv(out, index=False, encoding="utf-8-sig")


# =============================================================================
# Method 1 — Extract DOI from URL pattern (no HTTP request)
# =============================================================================

def doi_from_url_pattern(url: str) -> str | None:
    """
    Extract DOI directly from publisher URL structure.
    No HTTP request needed — instant.
    Returns normalized DOI or None if pattern not recognised.
    """
    # arXiv
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d+)", url, re.I)
    if m:
        return normalize_doi(f"10.48550/arXiv.{m.group(1)}")

    # bioRxiv / medRxiv
    m = re.search(
        r"(?:bio|med)rxiv\.org/content/(10\.\d{4,}/[^\s?#v]+?)(?:v\d+)?(?:[?#]|$)",
        url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Nature
    m = re.search(r"nature\.com/articles/(s[\w-]+\d)", url, re.I)
    if m:
        d = normalize_doi(f"10.1038/{m.group(1)}")
        return d if not is_incomplete(d) else None

    # Science
    m = re.search(r"science\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # PNAS
    m = re.search(r"pnas\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # PLoS
    m = re.search(r"plos\w*\.org/[^?]+\?.*?id=(10\.\d{4,}/[^&\s#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Frontiers
    m = re.search(r"frontiersin\.org/articles/(10\.\d{4,}/[^\s?#/]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Springer
    m = re.search(r"springer\.com/article/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Wiley
    m = re.search(r"wiley\.com/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Tandfonline
    m = re.search(
        r"tandfonline\.com/doi/(?:full|abs|pdf|epdf)/(10\.\d{4,}/[^\s?#]+)",
        url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Oxford OUP
    m = re.search(r"oup\.com/[^/]+/article/(10\.\d{4,}/[^\s?#/]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # SAGE
    m = re.search(r"sagepub\.com/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # ACS
    m = re.search(r"acs\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # ACM
    m = re.search(r"dl\.acm\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Cambridge
    m = re.search(r"cambridge\.org/.+/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # BMJ
    m = re.search(r"bmj\.com/content/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # NEJM
    m = re.search(r"nejm\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Lancet
    m = re.search(r"thelancet\.com/.+/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    # Generic fallback
    m = re.search(r"/(10\.\d{4,}/[^\s?#&\"'<>]{4,})", url)
    if m:
        d = normalize_doi(m.group(1))
        return d if not is_incomplete(d) else None

    return None


# =============================================================================
# Method 2 — Extract DOI from HTML page metadata
# =============================================================================

def doi_from_page(
    url: str,
    session: requests.Session,
    timeout: int = 12,
) -> str | None:
    """
    Fetch publisher page HTML and read citation_doi meta tag.
    Uses shared requests.Session for TCP connection reuse.
    Uses lxml parser if available (faster, no XML warnings).
    Requires: pip install beautifulsoup4 lxml
    """
    if not BS4_AVAILABLE:
        return None

    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, HTML_PARSER)

        # Standard scholarly meta tags — in priority order
        for attr, val in [
            ("name",     "citation_doi"),
            ("name",     "dc.identifier"),
            ("name",     "DC.Identifier"),
            ("name",     "prism.doi"),
            ("name",     "bepress_citation_doi"),
        ]:
            tag = soup.find(
                "meta",
                attrs={attr: re.compile(f"^{re.escape(val)}$", re.I)},
            )
            if tag and tag.get("content"):
                raw = tag["content"].strip()
                raw = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw, flags=re.I)
                raw = re.sub(r"^doi:\s*", "", raw, flags=re.I)
                d = normalize_doi(raw)
                if d.startswith("10.") and not is_incomplete(d):
                    return d

        # doi.org anchor links (arXiv page style)
        link = soup.find("a", href=re.compile(r"doi\.org/10\.", re.I))
        if link:
            href = link.get("href", "")
            raw = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", href, flags=re.I)
            d = normalize_doi(raw)
            if d.startswith("10.") and not is_incomplete(d):
                return d

        # Canonical URL — sometimes contains DOI
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            d = doi_from_url_pattern(canonical["href"])
            if d:
                return d

        return None

    except Exception as e:
        logging.debug("Page fetch failed %s: %s", url, e)
        return None


# =============================================================================
# Resolve DOIs from all candidate posts
# =============================================================================

def resolve_candidates(
    candidates: pd.DataFrame,
    pub_domains: list[str],
    fetch_pages: bool,
    page_delay: float,
) -> pd.DataFrame:
    """
    Resolve publication-domain URLs to DOIs.

    Optimizations applied:
    1. URL deduplication — each unique URL processed exactly once
    2. Cache loading — previously resolved URLs skipped instantly
    3. Page fetch checkpoint — interrupted runs resume without re-fetching
    4. requests.Session — TCP connection reuse across all page fetches
    5. lxml parser — faster HTML parsing, no XML warnings
    6. Checkpoint saved every 500 URLs — safe to interrupt at any time

    Returns DataFrame with columns:
        post_uri, matched_url, resolved_doi, method
    """
    logging.info("Resolving DOIs from %s candidate records...", len(candidates))

    if "matched_url" not in candidates.columns:
        logging.error("matched_url column missing from candidate posts.")
        return pd.DataFrame()

    # ── Load previously resolved URLs (vectorized, not iterrows) ──────────────
    prev_resolved: dict[str, tuple[str, str]] = {}
    prev_df = load_parquet(RESOLVED_CACHE)
    if not prev_df.empty and "matched_url" in prev_df.columns and "resolved_doi" in prev_df.columns:
        mask = prev_df["matched_url"].notna() & prev_df["resolved_doi"].notna()
        urls_col    = prev_df.loc[mask, "matched_url"].astype(str).tolist()
        dois_col    = prev_df.loc[mask, "resolved_doi"].astype(str).tolist()
        methods_col = prev_df.loc[mask, "method"].astype(str).tolist() \
            if "method" in prev_df.columns else ["url_pattern"] * sum(mask)
        prev_resolved = {u: (d, m) for u, d, m in zip(urls_col, dois_col, methods_col)}
        logging.info("Loaded %s previously resolved URLs from cache", len(prev_resolved))

    # ── Load page fetch checkpoint ─────────────────────────────────────────────
    page_attempted: set[str] = set()
    ckpt_df = load_parquet(PAGE_FETCH_CKPT)
    if not ckpt_df.empty and "url" in ckpt_df.columns:
        page_attempted = set(ckpt_df["url"].dropna().astype(str))
        logging.info("Loaded %s page-fetch attempts from checkpoint", len(page_attempted))

    # ── Filter to publication-domain URLs only ─────────────────────────────────
    df = candidates[["post_uri", "matched_url"]].copy()
    df["_url"] = df["matched_url"].fillna("").astype(str)
    df = df[df["_url"].str.len() > 0].copy()
    df = df[~df["_url"].str.contains("doi.org/10.", case=False, na=False)].copy()
    df = df[df["_url"].apply(lambda u: is_pub_domain(u, pub_domains))].copy()
    logging.info("Publication-domain records: %s", len(df))

    # ── Deduplicate URLs ───────────────────────────────────────────────────────
    unique_urls     = df["_url"].unique().tolist()
    urls_to_process = [u for u in unique_urls if u not in prev_resolved]
    logging.info(
        "Unique URLs: %s | Already resolved: %s | To process: %s",
        len(unique_urls),
        len(unique_urls) - len(urls_to_process),
        len(urls_to_process),
    )

    # ── Resolve each unique URL ────────────────────────────────────────────────
    url_to_result: dict[str, tuple[str, str]] = dict(prev_resolved)
    counts: dict[str, int] = {"already_cached": len(unique_urls) - len(urls_to_process)}
    page_ckpt_new: list[dict[str, str]] = []

    # Shared HTTP session — reuses TCP connections across all page fetches
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; science-bluesky-research-bot/1.0; "
            "+https://github.com/patrickwhoiam/Science-in-Bluesky)"
        )
    })

    try:
        for url in tqdm(urls_to_process, desc="Resolving unique URLs"):

            # Method 1 — URL pattern (instant, no HTTP)
            doi = doi_from_url_pattern(url)
            if doi:
                url_to_result[url] = (doi, "url_pattern")
                counts["url_pattern"] = counts.get("url_pattern", 0) + 1
                continue

            # Method 2 — Page metadata fetch
            if fetch_pages and BS4_AVAILABLE:
                if url in page_attempted:
                    counts["page_attempted_no_doi"] = counts.get("page_attempted_no_doi", 0) + 1
                    continue

                time.sleep(page_delay)
                doi = doi_from_page(url, session)
                page_ckpt_new.append({
                    "url": url,
                    "doi": doi or "",
                    "ts":  datetime.now().isoformat(timespec="seconds"),
                })

                if doi:
                    url_to_result[url] = (doi, "page_metadata")
                    counts["page_metadata"] = counts.get("page_metadata", 0) + 1
                else:
                    counts["not_found"] = counts.get("not_found", 0) + 1

                # Save checkpoint every 500 page fetches
                if len(page_ckpt_new) % 500 == 0:
                    _save_page_checkpoint(page_ckpt_new)

            else:
                counts["not_found"] = counts.get("not_found", 0) + 1

    finally:
        session.close()
        # Always save checkpoint on exit (including Ctrl+C)
        if page_ckpt_new:
            _save_page_checkpoint(page_ckpt_new)
            logging.info("Page fetch checkpoint saved: %s new attempts", len(page_ckpt_new))

    # ── Map results back to all post records ───────────────────────────────────
    results: list[dict[str, Any]] = []
    seen_post_uris: set[str] = set()

    for _, row in df.iterrows():
        url      = row["_url"]
        post_uri = str(row.get("post_uri", ""))
        if url in url_to_result and post_uri not in seen_post_uris:
            doi, method = url_to_result[url]
            seen_post_uris.add(post_uri)
            results.append({
                "post_uri":     post_uri,
                "matched_url":  url,
                "resolved_doi": doi,
                "method":       method,
            })

    logging.info("Resolution summary:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        logging.info("  %-30s %s", k + ":", f"{v:,}")
    logging.info("Unique URLs resolved: %s / %s", len(url_to_result), len(unique_urls))
    logging.info("Post records with resolved DOI: %s", len(results))

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["post_uri", "matched_url", "resolved_doi", "method"]
    )


def _save_page_checkpoint(new_records: list[dict[str, str]]) -> None:
    """Append new page fetch attempts to the checkpoint file."""
    new_df = pd.DataFrame(new_records)
    existing = load_parquet(PAGE_FETCH_CKPT)
    combined = pd.concat([existing, new_df], ignore_index=True) \
        if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["url"], keep="last")
    save_parquet(combined, PAGE_FETCH_CKPT)


# =============================================================================
# OpenAlex validation
# =============================================================================

def oa_work_record(work: dict[str, Any], req_doi: str) -> dict[str, Any]:
    loc    = work.get("primary_location") or {}
    src    = loc.get("source") or {}
    oa     = work.get("open_access") or {}
    topics = [t["display_name"] for t in (work.get("topics") or [])[:3]
              if isinstance(t, dict) and t.get("display_name")]
    ndoi   = normalize_doi(work.get("doi") or req_doi)
    return {
        "doi":                 ndoi,
        "requested_doi":       req_doi,
        "openalex_found":      True,
        "openalex_id":         work.get("id"),
        "paper_title":         work.get("display_name"),
        "publication_year":    work.get("publication_year"),
        "work_type":           work.get("type"),
        "venue_name":          src.get("display_name"),
        "venue_type":          src.get("type"),
        "is_oa":               oa.get("is_oa"),
        "cited_by_count":      work.get("cited_by_count"),
        "topics_top3":         "; ".join(topics),
        "openalex_checked_at": datetime.now().isoformat(timespec="seconds"),
        "openalex_error":      None,
    }


def oa_missing_record(doi: str, err: str | None = None) -> dict[str, Any]:
    return {
        "doi": doi, "requested_doi": doi, "openalex_found": False,
        "openalex_id": None, "paper_title": None, "publication_year": None,
        "work_type": None, "venue_name": None, "venue_type": None,
        "is_oa": None, "cited_by_count": None, "topics_top3": None,
        "openalex_checked_at": datetime.now().isoformat(timespec="seconds"),
        "openalex_error": err,
    }


def fetch_oa_batch(dois: list[str], api_key: str) -> list[dict[str, Any]]:
    params = {
        "filter":   "doi:" + "|".join(dois),
        "per_page": 100,
        "select":   "id,doi,display_name,publication_year,type,"
                    "primary_location,open_access,cited_by_count,topics",
        "api_key":  api_key,
    }
    for attempt in range(4):
        try:
            r = requests.get(OPENALEX_API, params=params, timeout=40)
            if r.status_code == 429:
                wait = min(60, 2 ** attempt + random.random())
                logging.warning("OpenAlex rate limit — sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            found: dict[str, dict[str, Any]] = {}
            for w in r.json().get("results", []):
                if w.get("doi"):
                    nd = normalize_doi(str(w["doi"]))
                    found[nd] = oa_work_record(w, nd)
            time.sleep(0.2)
            return [found.get(d, oa_missing_record(d)) for d in dois]
        except Exception as e:
            wait = min(60, 2 ** attempt + random.random())
            logging.warning("OpenAlex attempt %s/4 failed: %s", attempt + 1, e)
            time.sleep(wait)
    return [oa_missing_record(d, "all_retries_failed") for d in dois]


def validate_with_openalex(dois: list[str], api_key: str) -> pd.DataFrame:
    """Validate DOIs with OpenAlex. Reuses cache — only fetches new DOIs."""
    logging.info("Validating %s unique DOIs with OpenAlex...", len(dois))

    cache = load_parquet(OA_CACHE)
    cached = set(cache["doi"].dropna().astype(str)) \
        if not cache.empty and "doi" in cache.columns else set()
    to_fetch = [d for d in dois if d not in cached]
    logging.info("Cached: %s | To fetch: %s", len(cached), len(to_fetch))

    if to_fetch:
        new: list[dict[str, Any]] = []
        for i in tqdm(range(0, len(to_fetch), 50), desc="OpenAlex batches"):
            batch = to_fetch[i: i + 50]
            new.extend(fetch_oa_batch(batch, api_key))
            # Save cache every 10 batches (500 DOIs) to reduce disk writes
            if (i // 50 + 1) % 10 == 0 or i + 50 >= len(to_fetch):
                combined = pd.concat([cache, pd.DataFrame(new)], ignore_index=True) \
                    if not cache.empty else pd.DataFrame(new)
                if "doi" in combined.columns:
                    combined = combined.drop_duplicates(subset=["doi"], keep="last")
                save_parquet(combined, OA_CACHE)

    final = load_parquet(OA_CACHE)
    if final.empty or "doi" not in final.columns:
        return pd.DataFrame()
    return final[final["doi"].isin(dois)].copy()


# =============================================================================
# Bluesky engagement hydration
# =============================================================================

def bsky_post_record(post: dict[str, Any]) -> dict[str, Any]:
    a   = post.get("author") or {}
    lc  = post.get("likeCount")
    rc  = post.get("replyCount")
    rpc = post.get("repostCount")
    qc  = post.get("quoteCount")
    bc  = post.get("bookmarkCount")
    total = sum(int(x) for x in [lc, rc, rpc, qc, bc] if isinstance(x, int))
    return {
        "post_uri":              post.get("uri"),
        "hydration_status":      "found",
        "like_count":            lc,
        "reply_count":           rc,
        "repost_count":          rpc,
        "quote_count":           qc,
        "bookmark_count":        bc,
        "engagement_total":      total,
        "post_cid":              post.get("cid"),
        "indexed_at":            post.get("indexedAt"),
        "author_handle":         a.get("handle"),
        "author_display_name":   a.get("displayName"),
        "engagement_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def bsky_missing_record(uri: str, status: str = "missing") -> dict[str, Any]:
    return {
        "post_uri": uri, "hydration_status": status,
        "like_count": None, "reply_count": None, "repost_count": None,
        "quote_count": None, "bookmark_count": None, "engagement_total": None,
        "post_cid": None, "indexed_at": None,
        "author_handle": None, "author_display_name": None,
        "engagement_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def hydrate_engagement(post_uris: list[str]) -> pd.DataFrame:
    """Fetch engagement counts for new posts. Reuses cache."""
    logging.info("Hydrating engagement for %s posts...", len(post_uris))

    cache = load_parquet(ENG_CACHE)
    cached = set(cache["post_uri"].dropna().astype(str)) \
        if not cache.empty and "post_uri" in cache.columns else set()
    to_fetch = [u for u in post_uris if u not in cached]
    logging.info("Cached: %s | To fetch: %s", len(cached), len(to_fetch))

    if not to_fetch:
        return cache[cache["post_uri"].isin(post_uris)].copy() \
            if not cache.empty else pd.DataFrame()

    endpoint = f"{BLUESKY_API}/xrpc/app.bsky.feed.getPosts"
    new: list[dict[str, Any]] = []

    for i in tqdm(range(0, len(to_fetch), 25), desc="Engagement hydration"):
        batch    = to_fetch[i: i + 25]
        params   = [("uris", u) for u in batch]
        last_err = None

        for attempt in range(4):
            try:
                r = requests.get(endpoint, params=params, timeout=40)
                if r.status_code == 429:
                    wait = min(60, 2 ** attempt + random.random())
                    logging.warning("Bluesky rate limit — sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                found = {
                    p.get("uri"): bsky_post_record(p)
                    for p in r.json().get("posts", []) if p.get("uri")
                }
                new.extend(found.get(u, bsky_missing_record(u, "not_found")) for u in batch)
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(min(60, 2 ** attempt + random.random()))
        else:
            new.extend(bsky_missing_record(u, f"error:{last_err}") for u in batch)

        time.sleep(0.25)
        combined = pd.concat([cache, pd.DataFrame(new)], ignore_index=True) \
            if not cache.empty else pd.DataFrame(new)
        if "post_uri" in combined.columns:
            combined = combined.drop_duplicates(subset=["post_uri"], keep="last")
        save_parquet(combined, ENG_CACHE)

    final = load_parquet(ENG_CACHE)
    return final[final["post_uri"].isin(post_uris)].copy() \
        if not final.empty else pd.DataFrame()


# =============================================================================
# Build and merge new science posts
# =============================================================================

def build_new_posts(
    candidates: pd.DataFrame,
    resolved: pd.DataFrame,
    oa_meta: pd.DataFrame,
    engagement: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine candidate post metadata + resolved DOI + OpenAlex metadata
    + engagement counts into rows compatible with science_posts_unique.parquet.
    Uses explicit suffixes on all merges to prevent column name collisions.
    """
    if resolved.empty or oa_meta.empty:
        return pd.DataFrame()

    confirmed = oa_meta[oa_meta["openalex_found"] == True].copy()  # noqa: E712
    if confirmed.empty:
        return pd.DataFrame()

    # resolved DOIs → OpenAlex metadata
    merged = resolved.merge(
        confirmed,
        left_on="resolved_doi",
        right_on="doi",
        how="inner",
        suffixes=("_resolved", ""),
    )
    drop = [c for c in merged.columns if c.endswith("_resolved")]
    if drop:
        merged = merged.drop(columns=drop)

    # candidate base columns → merged
    base_cols = [c for c in [
        "post_uri", "bsky_url", "user_did", "created_at", "text",
        "langs", "reply_parent_uri", "reply_root_uri",
        "is_top_level_post", "raw_file",
    ] if c in candidates.columns]

    cands_base = candidates[base_cols].drop_duplicates("post_uri")
    merged = merged.merge(
        cands_base, on="post_uri", how="left", suffixes=("", "_cand")
    )
    drop = [c for c in merged.columns if c.endswith("_cand")]
    if drop:
        merged = merged.drop(columns=drop)

    # Engagement counts
    if not engagement.empty:
        eng_cols = [c for c in [
            "post_uri", "hydration_status", "like_count", "reply_count",
            "repost_count", "quote_count", "bookmark_count", "engagement_total",
            "post_cid", "indexed_at", "author_handle", "author_display_name",
            "engagement_fetched_at",
        ] if c in engagement.columns]
        merged = merged.merge(
            engagement[eng_cols].drop_duplicates("post_uri"),
            on="post_uri", how="left", suffixes=("", "_eng"),
        )
        drop = [c for c in merged.columns if c.endswith("_eng")]
        if drop:
            merged = merged.drop(columns=drop)

    merged["evidence_type"]    = "doi_openalex_validated"
    merged["confidence_layer"] = "high"
    merged["doi_source_field"] = "publication_url:" + merged["method"]
    merged = merged.drop_duplicates(subset=["post_uri"], keep="first")

    logging.info("New confirmed science posts built: %s", len(merged))
    return merged


def merge_into_dataset(new_posts: pd.DataFrame) -> int:
    """
    Merge new posts into final_dataset/science_posts_unique.parquet.
    Deduplicates on post_uri — existing posts never overwritten or duplicated.
    Returns number of net new posts added.
    """
    existing = load_parquet(SCIENCE_POSTS)

    if existing.empty:
        logging.warning("science_posts_unique.parquet not found — creating new.")
        save_parquet(new_posts, SCIENCE_POSTS)
        save_preview(new_posts, "science_posts_unique")
        return len(new_posts)

    existing_uris = set(existing["post_uri"].dropna().astype(str))
    truly_new = new_posts[~new_posts["post_uri"].isin(existing_uris)].copy()

    if truly_new.empty:
        logging.info("No new posts to add — all already exist in dataset.")
        return 0

    combined = pd.concat([existing, truly_new], ignore_index=True, sort=False)
    save_parquet(combined, SCIENCE_POSTS)
    save_preview(combined, "science_posts_unique")
    logging.info("Saved updated dataset → %s", SCIENCE_POSTS)
    return len(truly_new)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resolve DOIs from publication-domain URLs in candidate posts."
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N records (for testing, e.g. --limit 500)")
    p.add_argument("--no-fetch-pages", action="store_true",
                   help="Method 1 only — skip page fetch. Misses PubMed/JAMA/ScienceDirect.")
    p.add_argument("--page-delay", type=float, default=0.2,
                   help="Delay between page fetch requests in seconds (default: 0.2)")
    p.add_argument("--no-engage", action="store_true",
                   help="Skip Bluesky engagement hydration for new posts")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args        = parse_args()
    setup_logging()
    config_path = Path(args.config)
    api_key     = load_openalex_key(config_path)
    pub_domains = load_pub_domains(config_path)
    fetch_pages = not args.no_fetch_pages

    logging.info("=" * 60)
    logging.info("resolve_publication_links.py")
    logging.info("=" * 60)
    logging.info("Candidate file     : %s", CANDIDATE_FILE)
    logging.info("Science posts      : %s", SCIENCE_POSTS)
    logging.info("OpenAlex key       : %s", "set" if api_key else "MISSING")
    logging.info("BeautifulSoup      : %s", "available" if BS4_AVAILABLE else "not installed")
    logging.info("HTML parser        : %s", HTML_PARSER)
    logging.info("Fetch pages        : %s", fetch_pages and BS4_AVAILABLE)
    logging.info("Page delay         : %.2fs", args.page_delay)
    logging.info("Publication domains: %s configured", len(pub_domains))
    if args.limit:
        logging.info("Limit              : %s records", args.limit)

    if not BS4_AVAILABLE and fetch_pages:
        logging.warning("beautifulsoup4 not installed — Method 2 disabled.")
        logging.warning("Install: pip install beautifulsoup4 lxml")
    if HTML_PARSER == "html.parser":
        logging.info("Tip: install lxml for faster parsing — pip install lxml")

    # ── Step 1: Load candidates ───────────────────────────────────────────────
    if not CANDIDATE_FILE.exists():
        logging.error("Not found: %s", CANDIDATE_FILE)
        sys.exit(1)

    candidates = load_parquet(CANDIDATE_FILE)
    logging.info("Loaded %s candidate records", len(candidates))

    if args.limit:
        candidates = candidates.head(args.limit)
        logging.info("Limited to first %s records", args.limit)

    if candidates.empty:
        logging.info("No candidates to process.")
        sys.exit(0)

    # ── Step 2: Resolve DOIs from URLs ────────────────────────────────────────
    resolved = resolve_candidates(
        candidates, pub_domains, fetch_pages, args.page_delay
    )

    if resolved.empty:
        logging.info("No DOIs resolved. Nothing to do.")
        sys.exit(0)

    save_parquet(resolved, RESOLVED_CACHE)
    save_preview(resolved, "resolved_dois")
    logging.info("Saved resolution log → %s", RESOLVED_CACHE)

    # ── Step 3: Validate with OpenAlex ────────────────────────────────────────
    if not api_key:
        logging.error("OpenAlex API key missing. Set api_key in config.yaml.")
        sys.exit(1)

    unique_dois = sorted({
        d for d in resolved["resolved_doi"].dropna().astype(str)
        if d.startswith("10.") and not is_incomplete(d)
    })
    logging.info("Unique DOIs to validate: %s", len(unique_dois))

    oa_meta     = validate_with_openalex(unique_dois, api_key)
    n_confirmed = int((oa_meta["openalex_found"] == True).sum()) \
        if not oa_meta.empty else 0  # noqa: E712
    logging.info("OpenAlex confirmed: %s / %s", n_confirmed, len(unique_dois))

    if n_confirmed == 0:
        logging.info("No DOIs confirmed by OpenAlex. Nothing to merge.")
        sys.exit(0)

    # ── Step 4: Hydrate engagement ────────────────────────────────────────────
    confirmed_dois = set(
        oa_meta[oa_meta["openalex_found"] == True]["doi"].astype(str)  # noqa: E712
    ) if not oa_meta.empty else set()

    new_uris = list(
        resolved[resolved["resolved_doi"].isin(confirmed_dois)]["post_uri"]
        .dropna().astype(str).unique()
    )
    logging.info("New post URIs for engagement: %s", len(new_uris))

    engagement = pd.DataFrame()
    if not args.no_engage and new_uris:
        engagement = hydrate_engagement(new_uris)
        logging.info("Engagement hydrated: %s posts", len(engagement))
    else:
        logging.info("Engagement hydration skipped.")

    # ── Step 5: Build and merge ───────────────────────────────────────────────
    new_posts = build_new_posts(candidates, resolved, oa_meta, engagement)

    if new_posts.empty:
        logging.info("No new posts to merge.")
        sys.exit(0)

    n_added = merge_into_dataset(new_posts)

    # ── Summary ───────────────────────────────────────────────────────────────
    final = load_parquet(SCIENCE_POSTS)
    logging.info("=" * 60)
    logging.info("DONE")
    logging.info("=" * 60)
    logging.info("Candidate records processed   : %s", len(candidates))
    logging.info("DOIs resolved from URLs       : %s", len(resolved))
    logging.info("DOIs confirmed by OpenAlex    : %s", n_confirmed)
    logging.info("New posts added to dataset    : %s", n_added)
    logging.info("Total science posts (updated) : %s", len(final))
    logging.info("")
    logging.info("Updated files:")
    logging.info("  %s", SCIENCE_POSTS)
    logging.info("  %s", OA_CACHE)
    if not args.no_engage:
        logging.info("  %s", ENG_CACHE)
    logging.info("  %s", RESOLVED_CACHE)
    logging.info("  %s", PAGE_FETCH_CKPT)


if __name__ == "__main__":
    main()