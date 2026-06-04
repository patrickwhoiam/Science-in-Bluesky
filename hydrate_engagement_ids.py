#!/usr/bin/env python3
"""
hydrate_engagement_ids.py
=========================
Step 2 of final dataset pipeline.

Produces TWO types of records matching the required structure:

TYPE 1 — Top-level science post record:
{
  "doi":              "https://doi.org/10.xxxx/xxxxx",
  "uri":              "at://did:plc:alice/app.bsky.feed.post/3lxyz",
  "did":              "did:plc:alice",
  "operation":        "create",
  "rkey":             "3lxyz",
  "created_at":       "2025-04-10T16:41:07.251Z",
  "text":             "...",
  "langs":            ["en"],
  "reply_parent_uri": null,
  "reply_root_uri":   null,
  "like_count":       3,
  "like_uris":        ["at://did:plc:u1/app.bsky.feed.like/...", ...],
  "repost_count":     2,
  "repost_uris":      ["at://did:plc:u4/app.bsky.feed.repost/...", ...],
  "reply_count":      2,
  "reply_uris":       ["at://did:plc:u6/app.bsky.feed.post/...", ...]
}

NOTE on like_uris and repost_uris:
  Bluesky's public AppView API (getLikes, getRepostedBy) does NOT return the
  like/repost record rkeys. It only returns actor profiles (DID + handle).
  True AT-URIs like "at://did/app.bsky.feed.like/rkey" require authentication.
  This script stores actor DIDs as identifiers instead.
  like_uris   → ["did:plc:u1", "did:plc:u2", ...]
  repost_uris → ["did:plc:u4", "did:plc:u5", ...]
  reply_uris  → ["at://did:plc:u6/app.bsky.feed.post/3lp101", ...] ← full AT-URIs ✓

TYPE 2 — Reply post record (in reply_edges.parquet):
{
  "uri":              "at://did:plc:d/app.bsky.feed.post/ddd",
  "did":              "did:plc:d",
  "created_at":       "2025-09-13T02:03:00.000Z",
  "text":             "This is the third level reply",
  "langs":            [...],
  "reply_parent_uri": "at://did:plc:c/app.bsky.feed.post/ccc",
  "reply_root_uri":   "at://did:plc:a/app.bsky.feed.post/aaa",
  "reply_depth":      3,
  "reply_path_uris":  ["aaa", "bbb", "ccc", "ddd"],
  "source_post_uri":  "at://did:plc:a/app.bsky.feed.post/aaa"
}

Outputs:
  engagement/posts_with_engagement.parquet   TYPE 1 records — top-level posts
  engagement/reply_edges.parquet             TYPE 2 records — all reply posts
  engagement/like_edges.parquet              one row per like actor (actor detail)
  engagement/repost_edges.parquet            one row per repost actor (actor detail)
  engagement/checkpoints/<sha1>.json         per-post resumable checkpoint
  engagement/logs/

Usage:
  python hydrate_engagement_ids.py --limit-posts 20     # test
  python hydrate_engagement_ids.py                      # full run
  python hydrate_engagement_ids.py                      # rerun = resume from checkpoint
  python hydrate_engagement_ids.py --force              # ignore checkpoints
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

PUBLIC_APPVIEW = "https://public.api.bsky.app"


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hydrate Bluesky engagement URI lists matching required dataset structure."
    )
    p.add_argument(
        "--input",
        default="D:/sciencebluesky/final_dataset/science_posts_unique.parquet",
        help="Path to science_posts_unique.parquet",
    )
    p.add_argument(
        "--output-dir",
        default="D:/sciencebluesky/final_dataset/engagement",
        help="Output folder.",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="Only process top N posts sorted by engagement_total.",
    )
    p.add_argument(
        "--limit-posts", type=int, default=None,
        help="Only process first N posts. For testing.",
    )
    p.add_argument(
        "--skip-zero-engagement", action="store_true", default=True,
        help="Skip posts with engagement_total=0 (default: True).",
    )
    p.add_argument(
        "--no-skip-zero-engagement",
        dest="skip_zero_engagement", action="store_false",
    )
    p.add_argument(
        "--max-items", type=int, default=1000,
        help="Max likes/reposts/replies per post. Default: 1000.",
    )
    p.add_argument(
        "--thread-depth", type=int, default=6,
        help="Reply thread depth. Default: 6.",
    )
    p.add_argument(
        "--request-sleep", type=float, default=0.2,
        help="Sleep between requests. Default: 0.2s.",
    )
    p.add_argument(
        "--max-retries", type=int, default=4,
        help="Max retries per request. Default: 4.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoints and refetch everything.",
    )
    p.add_argument("--skip-likes",   action="store_true")
    p.add_argument("--skip-reposts", action="store_true")
    p.add_argument("--skip-replies", action="store_true")
    return p.parse_args()


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hydrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# Load posts
# =============================================================================

def extract_rkey(uri: str) -> str | None:
    """Extract rkey from AT-URI: at://did/app.bsky.feed.post/rkey → rkey"""
    try:
        return uri.rsplit("/", 1)[-1]
    except Exception:
        return None


def load_posts(args: argparse.Namespace) -> pd.DataFrame:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_parquet(input_path)
    logging.info("Loaded %s rows from %s", len(df), input_path)

    # Validate AT-URIs
    df = df[df["post_uri"].astype(str).str.startswith("at://", na=False)].copy()
    df = df.drop_duplicates(subset=["post_uri"]).reset_index(drop=True)

    # Skip zero-engagement posts
    if args.skip_zero_engagement and "engagement_total" in df.columns:
        before = len(df)
        df = df[df["engagement_total"].fillna(0) > 0].copy()
        logging.info(
            "Skipped %s zero-engagement posts → %s remain",
            before - len(df), len(df),
        )

    # Sort highest engagement first
    if "engagement_total" in df.columns:
        df = df.sort_values("engagement_total", ascending=False).reset_index(drop=True)

    if args.top_n is not None:
        df = df.head(args.top_n)
    if args.limit_posts is not None:
        df = df.head(args.limit_posts)

    logging.info("Posts to hydrate: %s", len(df))
    return df


# =============================================================================
# API client
# =============================================================================

class BskyClient:
    def __init__(self, request_sleep: float, max_retries: int) -> None:
        self.request_sleep = request_sleep
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "science-bluesky-pipeline/2.0"

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{PUBLIC_APPVIEW}/xrpc/{endpoint}"
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    wait = min(60, 2 ** attempt)
                    logging.warning("Rate limited — waiting %ss", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                time.sleep(self.request_sleep)
                return r.json()
            except Exception as e:
                wait = min(60, 2 ** attempt)
                logging.warning(
                    "Request failed attempt %s/%s: %s", attempt, self.max_retries, e
                )
                time.sleep(wait)
        raise RuntimeError(f"Failed {endpoint} after {self.max_retries} retries")

    def paginate(
        self,
        endpoint: str,
        params: dict[str, Any],
        key: str,
        max_items: int,
    ) -> list[Any]:
        items: list[Any] = []
        cursor = None
        while len(items) < max_items:
            p = dict(params)
            p["limit"] = min(100, max_items - len(items))
            if cursor:
                p["cursor"] = cursor
            payload = self.get(endpoint, p)
            batch = payload.get(key) or []
            if not isinstance(batch, list):
                break
            items.extend(batch)
            cursor = payload.get("cursor")
            if not cursor or not batch:
                break
        return items[:max_items]


# =============================================================================
# Fetch likes
#
# getLikes returns actor profiles (DID + handle + displayName) + timestamps.
# The like record rkey is NOT returned by the public API.
# We store actor_did as the identifier and provide full actor detail rows.
# like_uris in the summary = list of actor DIDs (best available public data).
# =============================================================================

def fetch_like_rows(
    client: BskyClient, post_uri: str, max_items: int
) -> list[dict[str, Any]]:
    items = client.paginate(
        "app.bsky.feed.getLikes", {"uri": post_uri}, "likes", max_items
    )
    rows = []
    for item in items:
        actor = item.get("actor") or {}
        rows.append({
            "source_post_uri":    post_uri,
            "actor_did":          actor.get("did"),
            "actor_handle":       actor.get("handle"),
            "actor_display_name": actor.get("displayName"),
            "like_created_at":    item.get("createdAt"),
            "like_indexed_at":    item.get("indexedAt"),
        })
    return rows


# =============================================================================
# Fetch reposts
#
# getRepostedBy returns actor profiles only.
# repost record rkeys are NOT returned by the public API.
# repost_uris in the summary = list of actor DIDs.
# =============================================================================

def fetch_repost_rows(
    client: BskyClient, post_uri: str, max_items: int
) -> list[dict[str, Any]]:
    actors = client.paginate(
        "app.bsky.feed.getRepostedBy", {"uri": post_uri}, "repostedBy", max_items
    )
    rows = []
    for actor in actors:
        rows.append({
            "source_post_uri":    post_uri,
            "actor_did":          actor.get("did"),
            "actor_handle":       actor.get("handle"),
            "actor_display_name": actor.get("displayName"),
        })
    return rows


# =============================================================================
# Fetch replies via getPostThread
#
# Produces TYPE 2 records matching required structure:
# {
#   uri, did, created_at, text, langs,
#   reply_parent_uri, reply_root_uri,
#   reply_depth, reply_path_uris,
#   source_post_uri
# }
# reply_path_uris is the full chain from root to this reply node.
# reply_depth counts from 1 (direct reply to source = depth 1).
# =============================================================================

def walk_thread(
    node: Any,
    source_post_uri: str,
    path_so_far: list[str],
    rows: list[dict[str, Any]],
    max_items: int,
) -> None:
    """
    Recursively walk getPostThread response tree.

    path_so_far: URIs from source post down to current node's parent.
    depth = len(path_so_far) + 1 for reply nodes (source post = depth 0).
    reply_path_uris = full chain including source post and this reply.
    """
    if not isinstance(node, dict) or len(rows) >= max_items:
        return

    post     = node.get("post") or {}
    post_uri = post.get("uri")

    if post_uri and post_uri != source_post_uri:
        # Reply node
        author    = post.get("author") or {}
        record    = post.get("record") or {}
        reply_ref = record.get("reply") or {}

        current_path = path_so_far + [post_uri]
        full_path    = [source_post_uri] + current_path   # root → this node
        depth        = len(current_path)                   # 1 = direct reply

        rows.append({
            # TYPE 2 required fields
            "uri":              post_uri,
            "did":              author.get("did"),
            "created_at":       record.get("createdAt"),
            "text":             record.get("text"),
            "langs":            json.dumps(record.get("langs") or []),
            "reply_parent_uri": (reply_ref.get("parent") or {}).get("uri"),
            "reply_root_uri":   (reply_ref.get("root") or {}).get("uri"),
            "reply_depth":      depth,
            "reply_path_uris":  json.dumps(full_path),
            # Extra context fields
            "source_post_uri":             source_post_uri,
            "reply_cid":                   post.get("cid"),
            "reply_author_handle":         author.get("handle"),
            "reply_author_display_name":   author.get("displayName"),
            "reply_indexed_at":            post.get("indexedAt"),
            "reply_like_count":            post.get("likeCount"),
            "reply_reply_count":           post.get("replyCount"),
            "reply_repost_count":          post.get("repostCount"),
            "reply_quote_count":           post.get("quoteCount"),
        })

        for child in (node.get("replies") or []):
            walk_thread(child, source_post_uri, current_path, rows, max_items)

    else:
        # Root/source post — start path from here
        root_path = [post_uri] if post_uri else []
        for child in (node.get("replies") or []):
            walk_thread(child, source_post_uri, root_path, rows, max_items)


def fetch_reply_rows(
    client: BskyClient, post_uri: str, thread_depth: int, max_items: int
) -> list[dict[str, Any]]:
    payload = client.get(
        "app.bsky.feed.getPostThread",
        {"uri": post_uri, "depth": thread_depth, "parentHeight": 0},
    )
    rows: list[dict[str, Any]] = []
    walk_thread(payload.get("thread"), post_uri, [], rows, max_items)
    return rows


# =============================================================================
# Checkpointing
# Stores full edge rows so reruns rebuild parquet files without re-calling API.
# =============================================================================

def ckpt_path(output_dir: Path, post_uri: str) -> Path:
    slug = hashlib.sha1(post_uri.encode()).hexdigest()
    return output_dir / "checkpoints" / f"{slug}.json"


def save_ckpt(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_ckpt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# Process one post
# =============================================================================

def process_post(
    client: BskyClient,
    post_uri: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "post_uri":    post_uri,
        "fetched_at":  datetime.now(timezone.utc).isoformat(),
        "status":      "ok",
        "error":       None,
        # Summary URI lists
        "like_uris":   [],   # actor DIDs (public API limitation)
        "repost_uris": [],   # actor DIDs (public API limitation)
        "reply_uris":  [],   # full AT-URIs of reply posts ✓
        # Full edge rows (stored in checkpoint for safe rerun)
        "like_rows":   [],
        "repost_rows": [],
        "reply_rows":  [],
    }

    try:
        if not args.skip_likes:
            rows = fetch_like_rows(client, post_uri, args.max_items)
            result["like_rows"] = rows
            result["like_uris"] = [r["actor_did"] for r in rows if r.get("actor_did")]

        if not args.skip_reposts:
            rows = fetch_repost_rows(client, post_uri, args.max_items)
            result["repost_rows"] = rows
            result["repost_uris"] = [r["actor_did"] for r in rows if r.get("actor_did")]

        if not args.skip_replies:
            rows = fetch_reply_rows(client, post_uri, args.thread_depth, args.max_items)
            result["reply_rows"] = rows
            result["reply_uris"] = [r["uri"] for r in rows if r.get("uri")]

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        logging.warning("Failed %s: %s", post_uri, e)

    return result


# =============================================================================
# Save helpers
# =============================================================================

def save_parquet(rows: list[dict[str, Any]], path: Path, preview: int = 500) -> None:
    df = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if not df.empty:
        df.head(preview).to_csv(
            path.with_name(path.stem + "_preview.csv"),
            index=False, encoding="utf-8-sig",
        )
    logging.info("Saved %s rows → %s", len(df), path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Load source posts — keep all columns from science_posts_unique.parquet
    df = load_posts(args)
    post_uris = df["post_uri"].tolist()

    client = BskyClient(
        request_sleep=args.request_sleep,
        max_retries=args.max_retries,
    )

    all_like_rows:   list[dict[str, Any]] = []
    all_repost_rows: list[dict[str, Any]] = []
    all_reply_rows:  list[dict[str, Any]] = []

    # Summary: one row per post with engagement URI lists
    engagement_rows: list[dict[str, Any]] = []

    for post_uri in tqdm(post_uris, desc="Hydrating engagement IDs"):
        ckpt = ckpt_path(output_dir, post_uri)

        if ckpt.exists() and not args.force:
            data = load_ckpt(ckpt)
        else:
            data = process_post(client, post_uri, args)
            save_ckpt(ckpt, data)   # includes full edge rows for safe rerun

        all_like_rows.extend(data.get("like_rows", []))
        all_repost_rows.extend(data.get("repost_rows", []))
        all_reply_rows.extend(data.get("reply_rows", []))

        engagement_rows.append({
            "post_uri":               post_uri,
            "fetched_at":             data.get("fetched_at"),
            "status":                 data.get("status"),
            "error":                  data.get("error"),
            # TYPE 1 required fields
            "like_uris":              json.dumps(data.get("like_uris", [])),
            "repost_uris":            json.dumps(data.get("repost_uris", [])),
            "reply_uris":             json.dumps(data.get("reply_uris", [])),
            # Counts of what was actually collected
            "like_count_collected":   len(data.get("like_uris", [])),
            "repost_count_collected": len(data.get("repost_uris", [])),
            "reply_count_collected":  len(data.get("reply_uris", [])),
        })

    # --- Build TYPE 1: top-level posts with engagement ---
    # Merge engagement back into original dataset (preserves all original columns)
    engagement_df = pd.DataFrame(engagement_rows)
    posts_with_engagement = df.merge(
        engagement_df, on="post_uri", how="left"
    )
    # Ensure rkey is present (extract from post_uri if not already a column)
    if "rkey" not in posts_with_engagement.columns:
        posts_with_engagement["rkey"] = posts_with_engagement["post_uri"].map(extract_rkey)

    save_parquet(
        posts_with_engagement.to_dict(orient="records"),
        output_dir / "posts_with_engagement.parquet",
    )

    # --- Build TYPE 2: reply edge records ---
    save_parquet(all_reply_rows,  output_dir / "reply_edges.parquet")

    # --- Actor detail files (for network/user analysis) ---
    save_parquet(all_like_rows,   output_dir / "like_edges.parquet")
    save_parquet(all_repost_rows, output_dir / "repost_edges.parquet")

    # Final stats
    ok_count  = engagement_df[engagement_df["status"] == "ok"].shape[0]  if not engagement_df.empty else 0
    err_count = engagement_df[engagement_df["status"] == "error"].shape[0] if not engagement_df.empty else 0

    logging.info("=" * 60)
    logging.info("Posts processed:  %s", len(post_uris))
    logging.info("Successful:       %s", ok_count)
    logging.info("Errors:           %s", err_count)
    logging.info("Like edges:       %s", len(all_like_rows))
    logging.info("Repost edges:     %s", len(all_repost_rows))
    logging.info("Reply edges:      %s", len(all_reply_rows))
    logging.info("=" * 60)
    logging.info("TYPE 1 output: posts_with_engagement.parquet")
    logging.info("TYPE 2 output: reply_edges.parquet")
    logging.info("Outputs saved to: %s", output_dir)
    logging.info("Next step: python collect_altmetrics.py")


if __name__ == "__main__":
    main()
