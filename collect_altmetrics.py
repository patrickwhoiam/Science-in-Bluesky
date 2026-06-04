#!/usr/bin/env python3
"""
collect_altmetrics.py
=====================
Step 3 of final dataset pipeline.

Reads unique DOIs from science_posts_unique.parquet and collects
Altmetric attention scores for each paper.

API used:
    https://api.altmetric.com/v1/doi/{doi}
    Free tier — no API key required.
    Rate limit: 1 request per second strictly.
    404 = paper not tracked by Altmetric (normal, not an error).
    429 = rate limited — back off and retry.

What Altmetric tracks per paper:
    - Bluesky mention count
    - Twitter/X mention count
    - News outlet mention count
    - Blog mention count
    - Policy document mention count
    - Reddit mention count
    - Facebook mention count
    - Wikipedia mention count
    - Overall Altmetric attention score
    - Score history over time

Inputs:
    D:/sciencebluesky/final_dataset/science_posts_unique.parquet
    (uses the doi column — 82,154 unique DOIs)

Outputs:
    D:/sciencebluesky/final_dataset/altmetrics/altmetrics.parquet
        one row per DOI with all platform counts
    D:/sciencebluesky/final_dataset/altmetrics/altmetrics_preview.csv
        first 500 rows for quick inspection
    D:/sciencebluesky/final_dataset/altmetrics/altmetrics_not_found.csv
        DOIs with no Altmetric record (404)
    D:/sciencebluesky/final_dataset/altmetrics/checkpoints/<doi_hash>.json
        per-DOI checkpoint — safe to interrupt and rerun
    D:/sciencebluesky/final_dataset/altmetrics/logs/

Usage:
    # Full run (will take ~23 hours for 82k DOIs at free tier rate limit)
    python collect_altmetrics.py

    # Test with 50 DOIs first
    python collect_altmetrics.py --limit-dois 50

    # If it crashes or is interrupted — just rerun, resumes from checkpoints
    python collect_altmetrics.py

    # Force refetch everything ignoring checkpoints
    python collect_altmetrics.py --force

    # Custom paths
    python collect_altmetrics.py
        --input D:/sciencebluesky/final_dataset/science_posts_unique.parquet
        --output-dir D:/sciencebluesky/final_dataset/altmetrics

Notes:
    - Free tier has no official documented rate limit number but
      Altmetric's documentation says "be reasonable" and suggests
      1 request/second. This script uses 1.1s sleep to be safe.
    - Papers not in Altmetric database return 404 — stored as
      altmetric_found=False. This is expected for ~30-50% of DOIs.
    - The script handles the case where a paper IS on Bluesky but
      Altmetric hasn't indexed it yet (recent papers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

ALTMETRIC_API_BASE = "https://api.altmetric.com/v1"

# Free tier: 1 request per second. Use 1.1s to be safe.
DEFAULT_SLEEP = 1.1
DEFAULT_RETRY_SLEEP = 5.0
DEFAULT_MAX_RETRIES = 4


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Altmetric attention scores for DOIs in science dataset."
    )
    p.add_argument(
        "--input",
        default="D:/sciencebluesky/final_dataset/science_posts_unique.parquet",
        help="Path to science_posts_unique.parquet (must have a 'doi' column).",
    )
    p.add_argument(
        "--output-dir",
        default="D:/sciencebluesky/final_dataset/altmetrics",
        help="Output folder for Altmetric data.",
    )
    p.add_argument(
        "--limit-dois",
        type=int,
        default=None,
        help="Only process first N unique DOIs. For testing.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Sleep between requests in seconds. Default: {DEFAULT_SLEEP}s (free tier safe).",
    )
    p.add_argument(
        "--retry-sleep",
        type=float,
        default=DEFAULT_RETRY_SLEEP,
        help=f"Base sleep after failed request. Default: {DEFAULT_RETRY_SLEEP}s.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per request. Default: {DEFAULT_MAX_RETRIES}.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and refetch everything.",
    )
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
    logging.info("Logging to %s", log_file)


# =============================================================================
# Load unique DOIs
# =============================================================================

def normalize_doi(doi: str) -> str:
    """Normalize DOI to bare form: 10.xxxx/xxxxx (no https://doi.org/ prefix)."""
    import re
    from urllib.parse import unquote
    doi = str(doi).strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = unquote(doi).split("?")[0].split("#")[0]
    doi = doi.rstrip(".,;:!?)]}>'\"")
    return doi.lower()


def load_unique_dois(args: argparse.Namespace) -> list[str]:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_parquet(input_path)
    logging.info("Loaded %s rows from %s", len(df), input_path)

    if "doi" not in df.columns:
        raise KeyError("'doi' column not found in input file.")

    # Normalize and deduplicate DOIs
    dois = (
        df["doi"]
        .dropna()
        .astype(str)
        .map(normalize_doi)
        .pipe(lambda s: s[s.str.startswith("10.")])  # valid DOIs start with 10.
        .drop_duplicates()
        .tolist()
    )

    logging.info("Unique valid DOIs to query: %s", len(dois))

    if args.limit_dois is not None:
        dois = dois[: args.limit_dois]
        logging.info("Limited to first %s DOIs for testing", args.limit_dois)

    return dois


# =============================================================================
# Altmetric API
# =============================================================================

class AltmetricClient:
    def __init__(self, sleep: float, retry_sleep: float, max_retries: int) -> None:
        self.sleep = sleep
        self.retry_sleep = retry_sleep
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "science-bluesky-research/1.0"

    def fetch(self, doi: str) -> dict[str, Any]:
        """
        Fetch Altmetric data for a DOI.

        Returns a dict with:
            status: "found" | "not_found" | "error"
            ... all Altmetric fields if found

        Important status codes:
            200 = found, full data returned
            404 = not tracked by Altmetric (normal for many papers)
            429 = rate limited — back off
            403 = also rate limit on free tier sometimes
            5xx = server error — retry
        """
        url = f"{ALTMETRIC_API_BASE}/doi/{doi}"
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, timeout=30)

                # 404 = not in Altmetric database — this is normal, not an error
                if r.status_code == 404:
                    time.sleep(self.sleep)
                    return {"status": "not_found", "doi": doi}

                # 429 or 403 = rate limited — wait longer and retry
                if r.status_code in (429, 403):
                    wait = self.retry_sleep * attempt * 2
                    logging.warning(
                        "Rate limited (HTTP %s) for DOI %s — waiting %.1fs",
                        r.status_code, doi, wait,
                    )
                    time.sleep(wait)
                    continue

                # 5xx = server error — retry with backoff
                if r.status_code >= 500:
                    wait = self.retry_sleep * attempt
                    logging.warning(
                        "Server error (HTTP %s) for DOI %s attempt %s/%s — waiting %.1fs",
                        r.status_code, doi, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                r.raise_for_status()

                data = r.json()
                data["status"] = "found"
                data["doi"] = doi
                time.sleep(self.sleep)
                return data

            except requests.exceptions.Timeout:
                wait = self.retry_sleep * attempt
                logging.warning(
                    "Timeout for DOI %s attempt %s/%s — waiting %.1fs",
                    doi, attempt, self.max_retries, wait,
                )
                last_error = "timeout"
                time.sleep(wait)

            except requests.exceptions.ConnectionError as e:
                wait = self.retry_sleep * attempt
                logging.warning(
                    "Connection error for DOI %s attempt %s/%s: %s — waiting %.1fs",
                    doi, attempt, self.max_retries, str(e)[:80], wait,
                )
                last_error = str(e)[:120]
                time.sleep(wait)

            except Exception as e:
                wait = self.retry_sleep * attempt
                logging.warning(
                    "Unexpected error for DOI %s attempt %s/%s: %s",
                    doi, attempt, self.max_retries, e,
                )
                last_error = str(e)[:120]
                time.sleep(wait)

        # All retries exhausted
        logging.error("Failed DOI %s after %s retries: %s", doi, self.max_retries, last_error)
        return {"status": "error", "doi": doi, "error_message": last_error}


# =============================================================================
# Parse Altmetric response into flat record
# =============================================================================

def parse_altmetric_response(data: dict[str, Any], doi: str) -> dict[str, Any]:
    """
    Flatten the Altmetric API response into a clean record with consistent columns.

    Altmetric API response key reference:
        score               overall Altmetric attention score
        cited_by_posts_count        total posts across all platforms
        cited_by_tweeters_count     Twitter/X mentions
        cited_by_msm_count          mainstream news mentions
        cited_by_feeds_count        blog/RSS feed mentions
        cited_by_policies_count     policy document mentions
        cited_by_rdts_count         Reddit mentions
        cited_by_fbwalls_count      Facebook mentions
        cited_by_wikipedia_count    Wikipedia mentions
        cited_by_gplus_count        Google+ (deprecated, may still appear)
        cited_by_videos_count       YouTube/video mentions
        cited_by_accounts_count     total unique accounts mentioning
        bluesky_posts               Bluesky-specific post count (newer field)
        history                     score history dict {1d, 2d, 3d, 1w, 1m, 3m, 6m, 1y, at}
        images                      badge image URLs (not needed for research)
        details_url                 link to Altmetric details page
        url                         canonical URL of the paper
        title                       paper title from Altmetric
        journal                     journal name from Altmetric
        published_on                unix timestamp of publication
        added_on                    unix timestamp when Altmetric first tracked
        last_updated                unix timestamp of last Altmetric update
    """
    status = data.get("status", "found")

    if status == "not_found":
        return {
            "doi":                      doi,
            "altmetric_found":          False,
            "altmetric_status":         "not_found",
            "altmetric_id":             None,
            "altmetric_score":          None,
            "altmetric_score_1d":       None,
            "altmetric_score_1w":       None,
            "altmetric_score_1m":       None,
            "altmetric_score_1y":       None,
            "altmetric_score_all_time": None,
            "cited_by_posts_count":     None,
            "cited_by_bluesky_count":   None,
            "cited_by_twitter_count":   None,
            "cited_by_news_count":      None,
            "cited_by_blogs_count":     None,
            "cited_by_policies_count":  None,
            "cited_by_reddit_count":    None,
            "cited_by_facebook_count":  None,
            "cited_by_wikipedia_count": None,
            "cited_by_videos_count":    None,
            "cited_by_accounts_count":  None,
            "altmetric_title":          None,
            "altmetric_journal":        None,
            "altmetric_details_url":    None,
            "altmetric_published_on":   None,
            "altmetric_last_updated":   None,
            "altmetric_fetched_at":     datetime.now(timezone.utc).isoformat(),
            "error_message":            None,
        }

    if status == "error":
        return {
            "doi":                      doi,
            "altmetric_found":          False,
            "altmetric_status":         "error",
            "altmetric_id":             None,
            "altmetric_score":          None,
            "altmetric_score_1d":       None,
            "altmetric_score_1w":       None,
            "altmetric_score_1m":       None,
            "altmetric_score_1y":       None,
            "altmetric_score_all_time": None,
            "cited_by_posts_count":     None,
            "cited_by_bluesky_count":   None,
            "cited_by_twitter_count":   None,
            "cited_by_news_count":      None,
            "cited_by_blogs_count":     None,
            "cited_by_policies_count":  None,
            "cited_by_reddit_count":    None,
            "cited_by_facebook_count":  None,
            "cited_by_wikipedia_count": None,
            "cited_by_videos_count":    None,
            "cited_by_accounts_count":  None,
            "altmetric_title":          None,
            "altmetric_journal":        None,
            "altmetric_details_url":    None,
            "altmetric_published_on":   None,
            "altmetric_last_updated":   None,
            "altmetric_fetched_at":     datetime.now(timezone.utc).isoformat(),
            "error_message":            data.get("error_message"),
        }

    # status == "found" — parse full response
    history = data.get("history") or {}

    # Bluesky count: newer Altmetric responses include bluesky_posts
    # Fall back to 0 if not present (older papers may not have this field)
    bluesky_count = data.get("bluesky_posts", data.get("cited_by_bluesky_count", 0))

    # Convert unix timestamps to ISO strings for readability
    def ts(val: Any) -> str | None:
        if val is None:
            return None
        try:
            return datetime.fromtimestamp(int(val), tz=timezone.utc).isoformat()
        except Exception:
            return str(val)

    return {
        "doi":                      doi,
        "altmetric_found":          True,
        "altmetric_status":         "found",
        "altmetric_id":             data.get("altmetric_id"),
        "altmetric_score":          data.get("score"),
        "altmetric_score_1d":       history.get("1d"),
        "altmetric_score_1w":       history.get("1w"),
        "altmetric_score_1m":       history.get("1m"),
        "altmetric_score_1y":       history.get("1y"),
        "altmetric_score_all_time": history.get("at"),
        "cited_by_posts_count":     data.get("cited_by_posts_count"),
        "cited_by_bluesky_count":   bluesky_count,
        "cited_by_twitter_count":   data.get("cited_by_tweeters_count"),
        "cited_by_news_count":      data.get("cited_by_msm_count"),
        "cited_by_blogs_count":     data.get("cited_by_feeds_count"),
        "cited_by_policies_count":  data.get("cited_by_policies_count"),
        "cited_by_reddit_count":    data.get("cited_by_rdts_count"),
        "cited_by_facebook_count":  data.get("cited_by_fbwalls_count"),
        "cited_by_wikipedia_count": data.get("cited_by_wikipedia_count"),
        "cited_by_videos_count":    data.get("cited_by_videos_count"),
        "cited_by_accounts_count":  data.get("cited_by_accounts_count"),
        "altmetric_title":          data.get("title"),
        "altmetric_journal":        data.get("journal"),
        "altmetric_details_url":    data.get("details_url"),
        "altmetric_published_on":   ts(data.get("published_on")),
        "altmetric_last_updated":   ts(data.get("last_updated")),
        "altmetric_fetched_at":     datetime.now(timezone.utc).isoformat(),
        "error_message":            None,
    }


# =============================================================================
# Checkpointing
# Stores full raw API response so reruns never call the API again.
# =============================================================================

def ckpt_path(output_dir: Path, doi: str) -> Path:
    slug = hashlib.sha1(doi.encode()).hexdigest()
    return output_dir / "checkpoints" / f"{slug}.json"


def save_ckpt(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_ckpt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Save outputs
# =============================================================================

def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(records)

    if df.empty:
        logging.warning("No records to save.")
        return

    # Main output — all DOIs
    out_parquet = output_dir / "altmetrics.parquet"
    out_csv     = output_dir / "altmetrics_preview.csv"
    df.to_parquet(out_parquet, index=False)
    df.head(500).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info("Saved %s rows → %s", len(df), out_parquet)

    # Separate file for not-found DOIs (useful for coverage analysis)
    not_found = df[df["altmetric_status"] == "not_found"].copy()
    if not not_found.empty:
        not_found[["doi", "altmetric_status", "altmetric_fetched_at"]].to_csv(
            output_dir / "altmetrics_not_found.csv",
            index=False, encoding="utf-8-sig",
        )
        logging.info("Not found (no Altmetric record): %s DOIs", len(not_found))

    # Separate file for errors (need rerun)
    errors = df[df["altmetric_status"] == "error"].copy()
    if not errors.empty:
        errors[["doi", "altmetric_status", "error_message", "altmetric_fetched_at"]].to_csv(
            output_dir / "altmetrics_errors.csv",
            index=False, encoding="utf-8-sig",
        )
        logging.info("Errors (need rerun): %s DOIs", len(errors))

    # Summary stats
    found_count     = (df["altmetric_status"] == "found").sum()
    not_found_count = (df["altmetric_status"] == "not_found").sum()
    error_count     = (df["altmetric_status"] == "error").sum()

    logging.info("=" * 60)
    logging.info("Total DOIs processed:    %s", len(df))
    logging.info("Found in Altmetric:      %s (%.1f%%)", found_count, found_count / len(df) * 100)
    logging.info("Not in Altmetric (404):  %s (%.1f%%)", not_found_count, not_found_count / len(df) * 100)
    logging.info("Errors:                  %s", error_count)

    found_df = df[df["altmetric_found"] == True].copy()  # noqa: E712
    if not found_df.empty:
        logging.info("--- Platform coverage (among found papers) ---")
        logging.info(
            "Bluesky mentions:        %s papers",
            (found_df["cited_by_bluesky_count"].fillna(0) > 0).sum(),
        )
        logging.info(
            "Twitter/X mentions:      %s papers",
            (found_df["cited_by_twitter_count"].fillna(0) > 0).sum(),
        )
        logging.info(
            "News mentions:           %s papers",
            (found_df["cited_by_news_count"].fillna(0) > 0).sum(),
        )
        logging.info(
            "Avg Altmetric score:     %.1f",
            found_df["altmetric_score"].dropna().mean(),
        )
    logging.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Input: %s", args.input)
    logging.info("Output dir: %s", output_dir)
    logging.info("Sleep between requests: %.2fs", args.sleep)
    logging.info("Max retries: %s", args.max_retries)

    dois = load_unique_dois(args)

    if not dois:
        logging.error("No valid DOIs found in input file.")
        sys.exit(1)

    client = AltmetricClient(
        sleep=args.sleep,
        retry_sleep=args.retry_sleep,
        max_retries=args.max_retries,
    )

    records: list[dict[str, Any]] = []
    skipped_cached = 0
    newly_fetched  = 0

    for doi in tqdm(dois, desc="Collecting Altmetrics"):
        ckpt = ckpt_path(output_dir, doi)

        if ckpt.exists() and not args.force:
            # Load from checkpoint — no API call needed
            raw = load_ckpt(ckpt)
            skipped_cached += 1
        else:
            # Fetch from Altmetric API
            raw = client.fetch(doi)
            # Save raw response — includes all fields for future use
            save_ckpt(ckpt, raw)
            newly_fetched += 1

        # Parse into flat record
        record = parse_altmetric_response(raw, doi)
        records.append(record)

        # Save intermediate output every 1000 DOIs so progress is never lost
        if len(records) % 1000 == 0:
            save_outputs(records, output_dir)
            logging.info(
                "Progress checkpoint: %s/%s DOIs processed (%s cached, %s fetched)",
                len(records), len(dois), skipped_cached, newly_fetched,
            )

    # Final save
    save_outputs(records, output_dir)

    logging.info("Cached (from checkpoint): %s", skipped_cached)
    logging.info("Newly fetched from API:   %s", newly_fetched)
    logging.info("Output: %s", output_dir / "altmetrics.parquet")
    logging.info("Next step: python build_final_structure.py")


if __name__ == "__main__":
    main()
