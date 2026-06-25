# Science on Bluesky : Pipeline, Dataset, and Analysis 
**Project:** Dynamics of Scientific Discussion in Decentralized Online Social Networks  
**Platform:** Bluesky (AT Protocol)  
**Data period:** March 2025 – June 2025  

---
<img width="1837" height="282" alt="image" src="https://github.com/user-attachments/assets/a35370c9-d6e7-4cae-bdfb-ad6209727e6b" />

## Overview

This repository contains the full data collection pipeline and analysis notebooks for a study of how scientific papers are shared and discussed on Bluesky. The study collects original science posts from the Bluesky Firehose, validates them against OpenAlex scholarly data, and analyzes engagement patterns across three research questions.

The analysis is organized around three research questions:

| # | Research Question | Status |
|---|------------------|--------|
| RQ1 | What are the quantitative characteristics and disciplinary distributions of scientific paper mentions on Bluesky? | Ready |
| RQ2 | How do users engage in scientific discourse on Bluesky, and what characterizes the quality of these discussions? | Ready |
| RQ3 | How do Bluesky's decentralized components influence the visibility and flow of scientific information? | pending analysis approach (Modification Needed) |

---

## Key Statistics


 | Metric | Count |

|--------|------:|

| Publication-link candidate records | 666,309 |

| Unique publication-link candidate posts | 479,073 |

| OpenAlex-validated science posts | 307,711 |

| Unique scholarly papers / OpenAlex works | 179,558 |

| Unique science-posting Bluesky users | 36,840 |

| Like records | 792,449 |

| Repost records | 292,867 |

| Reply records with tree metadata | 322,779 |

| Unique reply users | 27,166 |

| Total engagement records | 1,408,095 |

| Institutional science-posting accounts | 3,740 |

| Individual science-posting accounts | 33,100 |

---

## Structure recommendation

```
sciencebluesky/
│
├── analysis/
│   ├── RQ1/
│   ├── RQ2/
│   └── RQ3/
│   │   RQ1_dissemination_landscape_revised.ipynb
│   │   RQ2_final_complete.ipynb
│   │   RQ3_Architectural_Influence.ipynb
│
├── final_dataset/
│   ├── science_posts_unique.parquet    <- copied here automatically after Stage 4
│   ├── publication_link_candidate_posts.parquet
│   └── engagement/                     <- written directly by hydrate_engagement_ids.py
│       ├── posts_with_engagement.parquet
│       ├── like_edges.parquet
│       ├── repost_edges.parquet
│       └── reply_edges.parquet
│
├── data/
│   ├── interim/                        <- Stages 1-2 outputs
│   └── processed/                      <- Stages 3-4 outputs (source before final copy)
│
├── cache/
│   ├── openalex_metadata_cache.parquet
│   ├── openalex_domain_field_cache.parquet  <- written by RQ1 notebook enrichment
│   └── bluesky_engagement_cache.parquet
│
├── outputs/                            <- auto-created by run_pipeline.py
│   ├── tables/                         <- CSV previews from each stage
│   └── figures/
│
├── logs/
│
├── run_pipeline.py
├── hydrate_engagement_ids.py
├── collect_altmetrics.py
├── config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For analysis notebooks, also install:

```bash
pip install networkx scikit-learn nltk vaderSentiment transformers torch python-louvain
```

### 2. Configure paths

Edit `config.yaml` and set `raw_posts_dir` to the folder containing your raw Bluesky Firehose parquet files:

```yaml
project:
  raw_posts_dir: "E:/parquet_data/posts"   # path to your Firehose parquets
  project_dir: "."
```

Optionally set your OpenAlex API key (free from [openalex.org/settings/api](https://openalex.org/settings/api)):

```yaml
openalex:
  api_key: "your_key_here"
```

### 3. Run the pipeline

```bash
# Stage 1 — Extract posts with DOIs from Firehose parquets
python run_pipeline.py --stage extract --workers 2

# Stage 2 — Validate DOIs against OpenAlex, fetch paper metadata
python run_pipeline.py --stage validate

# Stage 3 — Join posts + metadata, deduplicate, build final dataset
python run_pipeline.py --stage build

# Stage 4 — Hydrate engagement counts via Bluesky AppView API
python run_pipeline.py --stage engage
```

After Stage 4, the main dataset is at `final_dataset/science_posts_unique.parquet`.

```bash
# Stage 5 — Collect who liked, reposted, and replied to each post
python hydrate_engagement_ids.py

# Stage 6 — Collect cross-platform Altmetric scores (requires API key)
python collect_altmetrics.py --api-key YOUR_KEY
```

**Note:** Stages 1–4 are safe to rerun. Checkpoints and API caches prevent redundant work. To force a full refresh, delete the relevant file in `cache/`.

---

## Datasets

### Main dataset — `science_posts_unique.parquet`

One row per unique original Bluesky post. Only includes posts where the DOI was confirmed by OpenAlex. Replies and reposts are excluded — every row is an original authored post.

| Column | Type | Description |
|--------|------|-------------|
| `post_uri` | string | AT-URI of the Bluesky post |
| `bsky_url` | string | Public Bluesky URL |
| `user_did` | string | Author's decentralized identifier (DID) |
| `author_handle` | string | Author's Bluesky handle |
| `author_display_name` | string | Author's display name |
| `created_at` | datetime | Post timestamp (UTC) |
| `text` | string | Full post text |
| `langs` | string | Detected post language(s) |
| `doi` | string | Normalized DOI of referenced paper |
| `doi_source_field` | string | Where DOI was found (`facets_uris`, `text`, or both) |
| `openalex_id` | string | OpenAlex work identifier |
| `paper_title` | string | Paper title from OpenAlex |
| `publication_year` | int | Year paper was published |
| `work_type` | string | Article / review / preprint etc. |
| `venue_name` | string | Journal or repository name |
| `venue_type` | string | Journal / conference / repository |
| `is_oa` | bool | Open access flag from OpenAlex |
| `cited_by_count` | int | Citation count at collection time |
| `topics_top3` | string | Top 3 OpenAlex research topics |
| `like_count` | int | Likes at time of collection |
| `reply_count` | int | Replies at time of collection |
| `repost_count` | int | Reposts at time of collection |
| `quote_count` | int | Quotes at time of collection |
| `engagement_total` | int | Sum of all engagement types |

> **Note on engagement counts:** These are snapshot counts fetched from the Bluesky AppView API at the time Stage 4 was run — not live counts. To refresh them, delete `cache/bluesky_engagement_cache.parquet` and rerun Stage 4.

---

### Engagement edge files

Produced by `hydrate_engagement_ids.py`. Each file stores full interaction records — who interacted with whom — enabling network construction and thread analysis.

**`engagement/like_edges.parquet`** — 555,582 records

| Column | Description |
|--------|-------------|
| `source_post_uri` | Post that was liked |
| `actor_did` | DID of user who liked |
| `actor_handle` | Handle of user who liked |
| `actor_display_name` | Display name |
| `like_created_at` | Timestamp of like |
| `like_indexed_at` | When Bluesky indexed the like |

**`engagement/repost_edges.parquet`** — 174,867 records

| Column | Description |
|--------|-------------|
| `source_post_uri` | Post that was reposted |
| `actor_did` | DID of user who reposted |
| `actor_handle` | Handle of user who reposted |
| `actor_display_name` | Display name |

**`engagement/reply_edges.parquet`** — 53,893 records

| Column | Description |
|--------|-------------|
| `source_post_uri` | Root science post being replied to |
| `reply_uri` | AT-URI of this reply |
| `reply_author_did` | Replier's DID |
| `reply_author_handle` | Replier's handle |
| `reply_text` | Text of the reply |
| `reply_depth` | Thread depth (1 = direct reply to root) |
| `reply_path_uris` | Full URI chain from root to this reply (JSON) |
| `reply_parent_uri` | Immediate parent post URI |
| `reply_root_uri` | Root post URI |
| `reply_like_count` | Likes on this reply |

---

### Which dataset to use per analysis task

| Task | Dataset |
|------|---------|
| RQ1 — temporal trends, growth curve | `science_posts_unique.parquet` |
| RQ1 — domain and field distribution | `science_posts_unique.parquet` + OpenAlex cache |
| RQ1 — cross-platform benchmark | `altmetrics.parquet` *(pending)* |
| RQ2 — passive vs active engagement | `science_posts_unique.parquet` |
| RQ2 — thread depth | `reply_edges.parquet` |
| RQ2 — NLP content classification | `science_posts_unique.parquet` (English subset) |
| RQ2 — user role analysis | `science_posts_unique.parquet` |
| RQ3 — network construction (Part 1) | *(pending)* |
| RQ3 — centrality analysis (Part 2) |  *(pending)* |
| RQ3 — community detection (Part 3) |  *(pending)* |

---

## Analysis Notebooks

Open `analysis/Dataset_overview.ipynb` first for a full dataset summary and statistics. Then run each RQ notebook in order.

### RQ1 — Dissemination Landscape

`analysis/RQ1/RQ1_dissemination_landscape_revised.ipynb`

- Part 1: Scale and speed — daily growth curve, monthly breakdown, top venues
- Part 2: Subject breadth — OpenAlex domain and field distribution, domain trends over time
- Part 3: Benchmark comparison — cross-platform comparison with Altmetric data *(runs when `altmetrics.parquet` is available)*

### RQ2 — Engagement Dynamics

`analysis/RQ2/RQ2_final_complete.ipynb`

- Part 1: Interaction structure — passive (likes + reposts) vs active (replies + quotes) engagement ratio
- Part 2: Content attributes — zero-shot NLI classification into Science Popularization / Peer Criticism / Promotional, with VADER sentiment cross-tabulation
- Part 3: User roles — institutional vs individual account dissemination and engagement comparison

### RQ3 — Architectural Influence

`analysis/RQ3/RQ3_Architectural_Influence.ipynb`

- Part 1: Network construction — weighted directed repost and reply networks, degree concentration (CCDF + Gini) ✓ **Included in current analysis**
- Part 2: Decentralized identity — institutional account centrality analysis ⏳ **Pending — to be confirmed in next meeting**
- Part 3: Network topology — Louvain community detection with null model comparison ⏳ **Pending — to be confirmed in next meeting**

---

## Data Collection Pipeline — How It Works

The pipeline runs in six stages:

```
Bluesky Firehose (raw parquets)
        │
        ▼
Stage 1 — extract
  Scan all Firehose parquet files
  Extract posts containing doi.org links or known publisher URLs
  Filter to original posts only (no replies, no reposts)
        │
        ▼
Stage 2 — validate
  Send unique DOIs to OpenAlex API in batches
  Confirm each DOI resolves to a real scholarly paper
  Fetch paper metadata: title, venue, domain, field, citation count, OA status
        │
        ▼
Stage 3 — build
  Join validated posts with OpenAlex metadata
  Deduplicate by post_uri
  Keep only posts with confirmed OpenAlex matches
        │
        ▼
Stage 4 — engage
  Call Bluesky AppView API for each post
  Fetch like_count, reply_count, repost_count, quote_count
  Output: final_dataset/science_posts_unique.parquet
        │
        ▼
Stage 5 — hydrate_engagement_ids.py
  For each post, fetch full interaction records
  Who liked it, who reposted it, who replied and at what thread depth
  Output: final_dataset/engagement/
        │
        ▼
Stage 6 — collect_altmetrics.py  [PENDING]
  Fetch cross-platform mention counts from Altmetric API
  Required for RQ1 Part 3 benchmark comparison
  Output: cache/altmetrics/altmetrics.parquet
```

---

## Data Sources

| Source | Purpose | Access |
|--------|---------|--------|
| Bluesky Firehose (AT Protocol) | Raw posts — March to June 2025 | Already collected |
| Bluesky AppView API | Engagement counts + actor URI lists | Public, no authentication required |
| OpenAlex API | Paper metadata, domain/field classification | Free — API key recommended |
| Altmetric API | Cross-platform attention scores | Requires institutional API key approval |

---

## Requirements

```bash
pip install -r requirements.txt
```

**Core pipeline** (required for Stages 1–6):
`pandas`, `pyarrow`, `requests`, `tqdm`, `pyyaml`, `matplotlib`, `tabulate`

**Analysis notebooks** (install separately):
`networkx`, `scikit-learn`, `nltk`, `vaderSentiment`, `transformers`, `torch`, `python-louvain`

> For GPU-accelerated NLI classification (RQ2 Part 2), a CUDA-compatible GPU is recommended. The classifier falls back to CPU automatically but will take significantly longer (~8–14 hours vs ~60–90 minutes on GPU).

---

## Citation

If you use this dataset or pipeline in your research, please cite:

```
[Citation to be added]
```

---

## Contact

[Contact information to be added]
