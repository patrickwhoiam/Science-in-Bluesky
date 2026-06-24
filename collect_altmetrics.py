#!/usr/bin/env python3
"""
collect_altmetrics.py
=====================
Step 3 of final dataset pipeline.

Reads unique DOIs from science_posts_unique.parquet and collects
Altmetric attention scores using the Altmetric Explorer API.

Authentication (Altmetric Explorer API):
    Digest = HMAC-SHA1(secret, filter_string)
    filter_string = filter_key1|value1|filter_key2|value2|...
                    sorted alphabetically, key appears once per filter
                    even for multi-value filters
    Excludes: order, page[number], page[size] from digest
    For POST identifier_lists: digest of empty string

API key setup (.env file):
    ALTMETRIC_API_KEY=your_key_here
    ALTMETRIC_API_SECRET=your_secret_here

Usage:
    python collect_altmetrics.py --limit-dois 100   # test
    python collect_altmetrics.py                     # full run (~3-4 hours)
    python collect_altmetrics.py                     # safe to rerun — resumes
    python collect_altmetrics.py --force             # refetch everything
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EXPLORER_BASE   = "https://www.altmetric.com/explorer/api"
DEFAULT_SLEEP   = 1.0
DEFAULT_RETRY   = 5.0
DEFAULT_RETRIES = 4
BATCH_SIZE      = 100
PAGE_SIZE       = 100


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   default="D:/sciencebluesky/final_dataset/science_posts_unique.parquet")
    p.add_argument("--output-dir",
                   default="D:/sciencebluesky/final_dataset/altmetrics")
    p.add_argument("--api-key",    default="")
    p.add_argument("--api-secret", default="")
    p.add_argument("--limit-dois", type=int, default=None)
    p.add_argument("--sleep",      type=float, default=DEFAULT_SLEEP)
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"altmetrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log: %s", log_file)


# =============================================================================
# DOI loading
# =============================================================================

def normalize_doi(doi: str) -> str:
    import re
    from urllib.parse import unquote
    doi = str(doi).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = unquote(doi).split("?")[0].split("#")[0]
    doi = doi.rstrip(".,;:!?)]}>'\"")
    return doi.lower()


def load_unique_dois(args: argparse.Namespace) -> list[str]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_parquet(path)
    logging.info("Loaded %s rows from %s", len(df), path)
    dois = (
        df["doi"].dropna().astype(str)
        .map(normalize_doi)
        .pipe(lambda s: s[s.str.startswith("10.")])
        .drop_duplicates().tolist()
    )
    logging.info("Unique valid DOIs: %s", len(dois))
    if args.limit_dois:
        dois = dois[: args.limit_dois]
        logging.info("Limited to first %s DOIs for testing", args.limit_dois)
    return dois


# =============================================================================
# HMAC-SHA1 digest — matches Altmetric official Python client exactly
#
# From official client source (altmetric/altmetric-explorer-api-client):
#   def digest(secret, message):
#       hmac_sha1 = hmac.new(secret.encode('utf-8'),
#                            message.encode('utf-8'), hashlib.sha1)
#       return hmac_sha1.hexdigest()
#
# Filter message format (from Filters.message()):
#   For each (key, value) sorted by key:
#     append key
#     if value is list/tuple/set: append each item
#     else: append value
#   join with '|'
#
# Verified against Altmetric docs:
#   filters: q=pandemic, scope=all, timeframe=3m, type=[article,dataset]
#   message: 'q|pandemic|scope|all|timeframe|3m|type|article|dataset'
#   digest:  '9fe61af3372cc2902d57b803b007519a65e6fe40'
# =============================================================================

def make_filter_message(filters: dict[str, Any]) -> str:
    """
    Build the filter message string for HMAC digest computation.
    Matches Altmetric official client Filters.message() exactly.
    Excludes order, page[number], page[size] from digest.
    """
    EXCLUDE = {"order", "page_size", "page_number"}
    parts: list[str] = []
    for key in sorted(filters.keys()):
        if key in EXCLUDE:
            continue
        val = filters[key]
        parts.append(key)
        if isinstance(val, (list, tuple, set)):
            for v in val:
                parts.append(str(v))
        else:
            parts.append(str(val))
    return "|".join(parts)


def compute_digest(secret: str, filter_message: str) -> str:
    """HMAC-SHA1 digest — matches Altmetric official client exactly."""
    return hmac.new(
        secret.encode("utf-8"),
        filter_message.encode("utf-8"),
        hashlib.sha1,
    ).hexdigest()


def verify_digest() -> bool:
    """Verify digest computation against known Altmetric docs example."""
    secret  = "0178eec6221448cea012e9d41a4922ba"
    filters = {"q": "pandemic", "scope": "all", "timeframe": "3m",
               "type": ["article", "dataset"]}
    expected = "9fe61af3372cc2902d57b803b007519a65e6fe40"
    message  = make_filter_message(filters)
    result   = compute_digest(secret, message)
    return result == expected


# =============================================================================
# Build signed URL (matches official client urlfor() method)
# =============================================================================

def build_url(api_key: str, api_secret: str, path: str,
              filters: dict[str, Any],
              page: int = 1, page_size: int = PAGE_SIZE) -> str:
    """
    Build a signed Explorer API URL.
    Digest computed on filter message (excluding page params).
    """
    message = make_filter_message(filters)
    digest  = compute_digest(api_secret, message)

    # Build query string
    parts = [f"key={api_key}", f"digest={digest}"]

    for key in sorted(filters.keys()):
        val = filters[key]
        if key == "order":
            parts.append(f"filter[order]={val}")
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                parts.append(f"filter[{key}][]={v}")
        else:
            parts.append(f"filter[{key}]={val}")

    parts.append(f"page[number]={page}")
    parts.append(f"page[size]={page_size}")

    return f"{EXPLORER_BASE}/{path}?{'&'.join(parts)}"


# =============================================================================
# Altmetric Explorer API client
# =============================================================================

class AltmetricClient:

    def __init__(self, api_key: str, api_secret: str,
                 sleep: float = DEFAULT_SLEEP,
                 max_retries: int = DEFAULT_RETRIES,
                 retry_sleep: float = DEFAULT_RETRY) -> None:
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.sleep       = sleep
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session     = requests.Session()
        self.session.headers["User-Agent"] = "science-bluesky-research/1.0"

    def _get(self, url: str) -> dict[str, Any] | None:
        """Make GET request with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, timeout=60)
                if r.status_code in (429, 403):
                    wait = self.retry_sleep * attempt * 2
                    logging.warning("Rate limited (HTTP %s) — waiting %.1fs",
                                    r.status_code, wait)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    wait = self.retry_sleep * attempt
                    logging.warning("Server error (HTTP %s) attempt %s/%s",
                                    r.status_code, attempt, self.max_retries)
                    time.sleep(wait)
                    continue
                if r.status_code == 400:
                    logging.error("Bad Request (400): %s", r.text[:300])
                    return None
                r.raise_for_status()
                time.sleep(self.sleep)
                return r.json()
            except Exception as e:
                wait = self.retry_sleep * attempt
                logging.warning("GET failed attempt %s/%s: %s",
                                attempt, self.max_retries, e)
                time.sleep(wait)
        return None

    def _post_identifier_list(self, identifiers: list[str]) -> dict[str, Any] | None:
        """
        POST to identifier_lists endpoint.

        From Altmetric docs:
          POST /explorer/api/identifier_lists HTTP/1.1
          Content-Type: application/x-www-form-urlencoded
          Body: key={key}&digest={digest}&identifiers=10.1038/xxx%0A10.1126/yyy

        Critical points from documentation:
          - Content-Type must be application/x-www-form-urlencoded (NOT JSON)
          - identifiers is a whitespace-separated string (NOT a JSON array)
          - key, digest, identifiers ALL go in the request BODY (NOT query params)
          - Digest is computed on the identifiers string itself (the raw identifier list)
        """
        url = f"{EXPLORER_BASE}/identifier_lists"

        # identifiers as newline-separated string (as shown in docs example)
        identifiers_str = "\n".join(identifiers)

        # Digest computed on the identifiers string for this endpoint
        digest = compute_digest(self.api_secret, identifiers_str)

        # All params go in the body as form-encoded data
        body_data = {
            "key":         self.api_key,
            "digest":      digest,
            "identifiers": identifiers_str,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(
                    url,
                    data=body_data,   # form-encoded body
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=60,
                )
                if r.status_code in (429, 403):
                    wait = self.retry_sleep * attempt * 2
                    logging.warning("Rate limited (HTTP %s) — waiting %.1fs",
                                    r.status_code, wait)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    wait = self.retry_sleep * attempt
                    time.sleep(wait)
                    continue
                if r.status_code == 400:
                    logging.error("POST identifier_lists 400: %s", r.text[:400])
                    return None
                r.raise_for_status()
                time.sleep(self.sleep)
                return r.json()
            except Exception as e:
                wait = self.retry_sleep * attempt
                logging.warning("POST failed attempt %s/%s: %s",
                                attempt, self.max_retries, e)
                time.sleep(wait)
        return None

    def create_identifier_list(self, dois: list[str]) -> str | None:
        """Upload DOIs as Identifier List. Returns list UUID or None."""
        resp = self._post_identifier_list(dois)
        if resp:
            list_id = (resp.get("data") or {}).get("id")
            if list_id:
                logging.debug("Created identifier list: %s", list_id)
                return list_id
            logging.error("No id in identifier_list response: %s", str(resp)[:200])
        return None

    def delete_identifier_list(self, list_id: str) -> None:
        """Clean up identifier list after use."""
        try:
            digest = compute_digest(self.api_secret, "")
            self.session.delete(
                f"{EXPLORER_BASE}/identifier_lists/{list_id}",
                data={"key": self.api_key, "digest": digest},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
        except Exception:
            pass

    def fetch_research_outputs(self, list_id: str) -> list[dict[str, Any]]:
        """Fetch all research outputs for identifier list with pagination."""
        all_outputs: list[dict[str, Any]] = []
        page = 1
        filters = {"identifier_list_id": list_id, "scope": "all"}

        while True:
            url  = build_url(self.api_key, self.api_secret,
                             "research_outputs", filters, page, PAGE_SIZE)
            resp = self._get(url)
            if not resp:
                break

            data = resp.get("data") or []
            all_outputs.extend(data)

            meta        = resp.get("meta") or {}
            resp_meta   = meta.get("response") or {}
            total_pages = resp_meta.get("total-pages", 1)

            if page >= total_pages or not data:
                break
            page += 1

        return all_outputs


# =============================================================================
# Parse Explorer API response
# =============================================================================

def parse_output(item: dict[str, Any], doi: str) -> dict[str, Any]:
    attrs     = item.get("attributes") or {}
    mentions  = attrs.get("mentions") or {}
    history   = attrs.get("historical-mentions") or {}
    dims      = attrs.get("dimensions") or {}
    readers   = attrs.get("readers") or {}
    sentiment = attrs.get("sentiment-analysis-totals") or {}
    return {
        "doi":                       doi,
        "altmetric_found":           True,
        "altmetric_status":          "found",
        "altmetric_id":              item.get("id"),
        "altmetric_score":           attrs.get("altmetric-score"),
        "altmetric_score_1d":        history.get("1d"),
        "altmetric_score_1w":        history.get("1w"),
        "altmetric_score_1m":        history.get("1m"),
        "altmetric_score_1y":        history.get("1y"),
        "altmetric_score_all_time":  history.get("at"),
        "cited_by_bluesky_count":    mentions.get("bluesky", 0),
        "cited_by_twitter_count":    mentions.get("tweet", 0),
        "cited_by_news_count":       mentions.get("msm", 0),
        "cited_by_blogs_count":      mentions.get("blog", 0),
        "cited_by_policies_count":   mentions.get("policy", 0),
        "cited_by_reddit_count":     mentions.get("rdt", 0),
        "cited_by_facebook_count":   mentions.get("fbwall", 0),
        "cited_by_wikipedia_count":  mentions.get("wikipedia", 0),
        "cited_by_videos_count":     mentions.get("video", 0),
        "cited_by_posts_count":      sum(mentions.values()) if mentions else 0,
        "dimensions_citations":      dims.get("citations"),
        "mendeley_readers":          readers.get("mendeley"),
        "oa_status":                 attrs.get("oa-status"),
        "oa_type":                   attrs.get("oa-type"),
        "output_type":               attrs.get("output-type"),
        "sentiment_strong_positive": sentiment.get("strong-positive"),
        "sentiment_positive":        sentiment.get("positive"),
        "sentiment_neutral_positive":sentiment.get("neutral-positive"),
        "sentiment_neutral":         sentiment.get("neutral"),
        "sentiment_neutral_negative":sentiment.get("neutral-negative"),
        "sentiment_negative":        sentiment.get("negative"),
        "sentiment_strong_negative": sentiment.get("strong-negative"),
        "altmetric_title":           attrs.get("title"),
        "altmetric_published_on":    attrs.get("publication-date"),
        "altmetric_badge_url":       attrs.get("badge-url"),
        "altmetric_fetched_at":      datetime.now(timezone.utc).isoformat(),
        "error_message":             None,
    }


def empty_record(doi: str, status: str = "not_found",
                 error: str | None = None) -> dict[str, Any]:
    return {
        "doi": doi, "altmetric_found": False, "altmetric_status": status,
        "altmetric_id": None, "altmetric_score": None,
        "altmetric_score_1d": None, "altmetric_score_1w": None,
        "altmetric_score_1m": None, "altmetric_score_1y": None,
        "altmetric_score_all_time": None,
        "cited_by_bluesky_count": None, "cited_by_twitter_count": None,
        "cited_by_news_count": None, "cited_by_blogs_count": None,
        "cited_by_policies_count": None, "cited_by_reddit_count": None,
        "cited_by_facebook_count": None, "cited_by_wikipedia_count": None,
        "cited_by_videos_count": None, "cited_by_posts_count": None,
        "dimensions_citations": None, "mendeley_readers": None,
        "oa_status": None, "oa_type": None, "output_type": None,
        "sentiment_strong_positive": None, "sentiment_positive": None,
        "sentiment_neutral_positive": None, "sentiment_neutral": None,
        "sentiment_neutral_negative": None, "sentiment_negative": None,
        "sentiment_strong_negative": None,
        "altmetric_title": None, "altmetric_published_on": None,
        "altmetric_badge_url": None,
        "altmetric_fetched_at": datetime.now(timezone.utc).isoformat(),
        "error_message": error,
    }


# =============================================================================
# Checkpointing
# =============================================================================

def batch_ckpt_path(output_dir: Path, batch_dois: list[str]) -> Path:
    key  = "|".join(sorted(batch_dois))
    slug = hashlib.sha1(key.encode()).hexdigest()
    return output_dir / "checkpoints" / f"{slug}.json"


def save_ckpt(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_ckpt(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Save outputs
# =============================================================================

def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_parquet(output_dir / "altmetrics.parquet", index=False)
    df.head(500).to_csv(output_dir / "altmetrics_preview.csv",
                        index=False, encoding="utf-8-sig")
    logging.info("Saved %s rows → %s", len(df),
                 output_dir / "altmetrics.parquet")

    not_found = df[df["altmetric_status"] == "not_found"]
    if not not_found.empty:
        not_found[["doi", "altmetric_status", "altmetric_fetched_at"]].to_csv(
            output_dir / "altmetrics_not_found.csv",
            index=False, encoding="utf-8-sig")

    errors = df[df["altmetric_status"] == "error"]
    if not errors.empty:
        errors[["doi", "altmetric_status", "error_message",
                "altmetric_fetched_at"]].to_csv(
            output_dir / "altmetrics_errors.csv",
            index=False, encoding="utf-8-sig")

    found       = (df["altmetric_status"] == "found").sum()
    not_found_n = (df["altmetric_status"] == "not_found").sum()
    error_n     = (df["altmetric_status"] == "error").sum()

    logging.info("=" * 60)
    logging.info("Total DOIs processed:      %s", len(df))
    logging.info("Found in Altmetric:        %s (%.1f%%)",
                 found, found / len(df) * 100)
    logging.info("Not in Altmetric (404):    %s (%.1f%%)",
                 not_found_n, not_found_n / len(df) * 100)
    logging.info("Errors:                    %s", error_n)

    found_df = df[df["altmetric_found"] == True].copy()  # noqa: E712
    if not found_df.empty:
        logging.info("--- Platform coverage ---")
        logging.info("Bluesky mentions:    %s papers",
                     (found_df["cited_by_bluesky_count"].fillna(0) > 0).sum())
        logging.info("Twitter/X mentions:  %s papers",
                     (found_df["cited_by_twitter_count"].fillna(0) > 0).sum())
        logging.info("News mentions:       %s papers",
                     (found_df["cited_by_news_count"].fillna(0) > 0).sum())
        logging.info("Avg Altmetric score: %.1f",
                     found_df["altmetric_score"].dropna().mean())
    logging.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args       = parse_args()
    api_key    = args.api_key    or os.getenv("ALTMETRIC_API_KEY", "")
    api_secret = args.api_secret or os.getenv("ALTMETRIC_API_SECRET", "")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("=" * 60)
    logging.info("collect_altmetrics.py — Altmetric Explorer API")
    logging.info("=" * 60)
    logging.info("Input:      %s", args.input)
    logging.info("Output:     %s", output_dir)
    logging.info("API key:    %s", "set" if api_key else "MISSING")
    logging.info("API secret: %s", "set" if api_secret else "MISSING")
    logging.info("Batch size: %s DOIs per request", BATCH_SIZE)

    if not api_key or not api_secret:
        logging.error("API key and secret required. Add to .env file.")
        sys.exit(1)

    # Verify digest against known Altmetric docs example
    if verify_digest():
        logging.info("Digest computation verified ✓")
    else:
        logging.error("Digest computation FAILED — check HMAC implementation")
        sys.exit(1)

    dois = load_unique_dois(args)
    if not dois:
        logging.error("No valid DOIs found.")
        sys.exit(1)

    client  = AltmetricClient(api_key, api_secret, sleep=args.sleep)
    batches = [dois[i: i + BATCH_SIZE] for i in range(0, len(dois), BATCH_SIZE)]
    logging.info("Total batches: %s", len(batches))

    all_records: list[dict[str, Any]] = []
    cached  = 0
    fetched = 0

    for idx, batch_dois in enumerate(tqdm(batches, desc="Processing batches")):
        ckpt = batch_ckpt_path(output_dir, batch_dois)

        if ckpt.exists() and not args.force:
            all_records.extend(load_ckpt(ckpt))
            cached += 1
            continue

        # Step 1: Create Identifier List
        list_id = client.create_identifier_list(batch_dois)
        if not list_id:
            for doi in batch_dois:
                all_records.append(empty_record(doi, "error",
                                                "identifier_list_failed"))
            continue

        # Step 2: Fetch Research Outputs
        try:
            outputs = client.fetch_research_outputs(list_id)
        except Exception as e:
            logging.error("Fetch failed for batch %s: %s", idx, e)
            outputs = []

        # Step 3: Match results to DOIs
        found_by_doi: dict[str, dict[str, Any]] = {}
        for item in outputs:
            attrs = item.get("attributes") or {}
            ids   = attrs.get("identifiers") or {}
            for item_doi in (ids.get("dois") or []):
                nd = normalize_doi(item_doi)
                if nd:
                    found_by_doi[nd] = item

        batch_records = [
            parse_output(found_by_doi[doi], doi) if doi in found_by_doi
            else empty_record(doi, "not_found")
            for doi in batch_dois
        ]

        save_ckpt(ckpt, batch_records)
        all_records.extend(batch_records)
        fetched += 1
        client.delete_identifier_list(list_id)

        if fetched % 10 == 0:
            save_outputs(all_records, output_dir)
            logging.info("Progress: batch %s/%s | cached: %s | fetched: %s",
                         idx + 1, len(batches), cached, fetched)

    save_outputs(all_records, output_dir)
    logging.info("Cached: %s batches | Fetched: %s batches", cached, fetched)
    logging.info("Output: %s", output_dir / "altmetrics.parquet")


if __name__ == "__main__":
    main()
