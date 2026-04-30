# IFDB Interactive Fiction Recommender

A full retrieval-and-ranking pipeline built on the [IFDB](https://ifdb.org) Interactive Fiction Database. Community ratings drive training of an **asymmetric two-tower bi-encoder** for candidate retrieval, followed by a fine-tuned **cross-encoder reranker**, indexed via **FAISS**. The system supports free-text queries, user-based personalised recommendations, and "more like this" seed-game queries, with optional hard filters and post-rerank diversity enforcement.

---

## Dataset

IFDB is a community-curated catalog of interactive fiction works with rich metadata and user reviews. The tables used by this pipeline are from the most recent [IFArchive](https://ifarchive.org/if-archive/info/ifdb/ifdb-archive-20260301.zip) dump (current version was archived on 03-01-2026), and contain the following data:

| Table | Rows | Role |
|---|---|---|
| `games` | 15,342 | Game metadata — title, author, system, genre, tags, description |
| `reviews` | 79,225 | Explicit ratings (1–5★) — primary supervision signal |
| `users` | 19,674 | User accounts — used for profile construction and filtering |
| `playedgames` | 70,088 | Implicit engagement — used to suppress already-played games from recommendations |

---

## Architecture

```
  User profile text ──► [ Query Encoder ]──► query_emb (384d)
  "Systems: inform.       fine-tuned                │
   Tags: mystery,         all-MiniLM-L6-v2    cosine similarity
   parser, …"                                       │
                                                    ▼
  Game document    ──► [ Doc Encoder   ]──► item_emb (384d)
  "Title: … Author:…      fine-tuned          FAISS flat index
   Systems: … Tags: …     all-MiniLM-L6-v2         │
   Description: …"                          all above threshold
                                                    │
                                        ┌── hard filters ──┐
                                        │  year / author   │
                                        │  system / tags   │
                                        │  rating / count  │
                                        └──────────────────┘
                                                    │
                                            author-cap (≤2/author)
                                                    │
                                           top_k_retrieve
                                                    │
                          ──► [ Cross-encoder Reranker ]──► scored list
                               fine-tuned                        │
                               ms-marco-MiniLM-L6-v2    filter by CE score
                                                                 │
                                                       diversifying step
                                                       (all user's top systems included)
                                                                 │
                                                         top_k_rerank results
```

### Two-tower bi-encoder (asymmetric)

The query tower and document tower start from the same `all-MiniLM-L6-v2` checkpoint but are fine-tuned independently with separate weights. This asymmetric design lets each tower specialise for its input domain:

- **Query encoder** — encodes user profiles (`"Systems: X. Tags: a, b, c."`) and free-text queries
- **Doc encoder** — encodes full game documents (`"Title: … Author: … Systems: … Tags: … Description: …"`)

**Loss:** InfoNCE (`MultipleNegativesRankingLoss`) with in-batch negatives. With batch size 64, each step compares a positive pair against 63 in-batch negatives — many of which are semantically close (e.g. other parser games), making the task hard without explicit mining. Temperature is set to 0.07.

### Interaction labelling

Interactions are labelled *relative to each game's quality*, not on an absolute scale:

- **Positive** (label=1): `rating > bayesian_avg + 0.25`
- **Negative** (label=0): `rating < bayesian_avg - 0.25`
- Ratings within the ±0.25 band are discarded as ambiguous

This means a 3★ review of a 2.7★ game counts as a positive, while the same rating for a 3.3★ game counts as a negative. The `bayesian_avg` is computed with a prior weight of 5 reviews toward 3.5★, shrinking low-review games toward the mean.

### Cross-encoder reranker

The reranker scores `(query_text, game_document)` pairs with a `CrossEncoder` model fine-tuned on the same positive/negative interactions. Raw logits are passed through sigmoid to produce a 0–1 relevance probability. Candidates with a sigmoid score below `min_rerank_score` are dropped before final results are returned. Optionally, the cross-encoder score is blended with the game's normalised Bayesian-average rating:

```
final_score = (1 - rating_weight) * ce_score + rating_weight * (bayesian_avg / 5.0)
```

### Diversifying steps

Between and after scoring, results pass through diversifying steps that:
1. Cap any individual author at `max_author_appearances=2` across the result list
2. Ensure coverage of all top systems drawn from the user's profile (for `userid` query mode)

---

## Repository layout

```
if-recommender/
├── config.yaml                   # All tunable parameters
├── pyproject.toml / uv.lock
│
├── src/
│   ├── db/
│   │   └── connector.py          # SQLAlchemy MySQL connection
│   ├── data/
│   │   ├── loader.py             # MySQL → Parquet extraction
│   │   ├── preprocessor.py       # Game docs, user profiles, interactions, splits
│   │   └── dataset.py            # PairDataset for bi-encoder training
│   ├── models/
│   │   └── two_tower.py          # TwoTowerModel wrapper
│   ├── index/
│   │   └── faiss_index.py        # GameIndex: build / search / save / load
│   ├── pipeline/
│   │   ├── retriever.py          # Stage 1: ANN retrieval + hard filtering + author cap
│   │   └── ranker.py             # Stage 2: cross-encoder reranking, diversity, eval metrics
│   └── utils/
│       └── env.py                # Logging setup, .env loading
│
└── scripts/
    ├── 01_extract.py             # MySQL → data/*.parquet
    ├── 02_preprocess.py          # Game docs (strict + retrieval), interactions, user profiles
    ├── 03_train_two_tower.py     # Fine-tune asymmetric bi-encoder (query + doc towers)
    ├── 04_train_reranker.py      # Fine-tune cross-encoder reranker
    ├── 05_build_index.py         # Encode all games → FAISS index + embedding files
    └── 06_run_pipeline.py        # Interactive demo + offline evaluation
```

---

## Quickstart

### Prerequisites

```bash
# Python 3.10+, uv recommended
uv sync

# MySQL running locally with the IFDB dump loaded
# Default: root@localhost:3306/ifarchive — no password
# Override via .env or environment variable:
export IFDB_DB_PASSWORD=yourpassword

# Optional: HuggingFace token for model downloads
# Create a .env file and set HF_TOKEN
```

### Run the pipeline

```bash
# 1. Extract tables from MySQL → data/
uv run scripts/01_extract.py

# 2. Build game documents, interaction matrix, and user profiles
uv run scripts/02_preprocess.py

# 3. Fine-tune the bi-encoder (query + doc encoders)
uv run scripts/03_train_two_tower.py

# 4. Fine-tune the cross-encoder reranker
uv run scripts/04_train_reranker.py

# 5. Encode all games and build the FAISS retrieval index
uv run scripts/05_build_index.py

# 6a. Offline evaluation on the test split
uv run scripts/06_run_pipeline.py --mode evaluate

# 6b. Interactive demo
uv run scripts/06_run_pipeline.py
```


---

## Evaluation

`scripts/06_run_pipeline.py --mode evaluate` reports Recall@K, NDCG@K, and MRR against held-out test-split positives for each user. Evaluation uses raw retrieval (no reranking) for speed, computed at K ∈ {1, 5, 10, 20, 50}.

---

## Interactive demo

The interactive demo supports three query types (selected at startup via `--query-type`):

| Mode | Input | Description |
|---|---|---|
| `text` (default) | `"Systems: twine. Tags: fantasy, slice of life, choice-based, graphics"` | Recommendations based on formatted text in the style of a user taste profile, which should include "Systems:" and "Tags:" lists. |
| `userid` | `"9rwstoqhvjcff8hf"` | Personalised recommendations from a user profile based on the user's positive rating history |
| `game_id` | a game ID string | "More like this" recomendations based on the seed game's profile |

```bash
uv run scripts/06_run_pipeline.py --query-type userid
uv run scripts/06_run_pipeline.py --query-type game_id
```

In `userid` mode, games the user has already reviewed or played are automatically suppressed from results. The user's display name and taste profile are shown before results.

### Hard filters

After entering a query, an optional `Filters >` prompt appears and accepts semicolon-separated constraints. All filters are AND-ed together and applied before the reranker:

| Key | Example | Behaviour |
|---|---|---|
| `year` | `year:2010-2020` | Publication year within range (inclusive) |
| `author` | `author:emily short` | Substring match against any individual author name |
| `system` | `system:inform` | Substring match against any individual system name |
| `tags` | `tags:fantasy, horror` | All listed tags must be present in the game's tag set |
| `rating` | `rating:3.5` | Bayesian-average rating ≥ value |
| `count` | `count:10` | Number of ratings ≥ value |

Press Enter to skip filtering.

---

## Data preparation details

### Game documents

Two document sets are built during preprocessing:

- **Strict set** (`game_docs.parquet`) — games with ≥ `min_reviews_per_game` reviews; used for training interactions
- **Retrieval set** (`game_docs_retrieval.parquet`) — all valid games (including zero-review games); used for indexing and serving

A game is included only if its title, author, system, description, and tags are all non-empty. Genre values are folded into the tags field, and all tags are subjected to a competition-tag filter and lowercased. System names are also cleaned: parentheticals and version numbers are stripped, values are lowercased and split. Author names are similarly split and canonicalized.

Each game produces two text representations:
- **`doc_text`** — full document for the doc encoder: `"Title: … Author: … Systems: … Tags: … Description: …"`
- **`query_text`** — profile-format for game-ID queries: `"Systems: … Tags: …"` (no title/author/description)

### User profiles

Each user's profile text is built by aggregating system and tag values from their positively-rated games (relative-positive interactions + any game rated ≥ 4★ absolutely). Format: `"Systems: X, Y. Tags: a, b, c, …"` — up to 3 systems and 20 tags, ranked by frequency. Two profile sets are built: one from training-split positives only (for training), and a broader one from all positive interactions (for retrieval serving).

---

## Configuration

All parameters live in `config.yaml`. Key settings:

```yaml
preprocessing:
  bayesian_prior_mean: 3.5       # prior rating for games with few reviews, 3.5 is the empirical average on IFDB
  bayesian_prior_weight: 5       # shrinks toward prior until this many real reviews
  rating_deviation_threshold: 0.25  # positive if rating > bayesian_avg + threshold

training:
  epochs: 3
  batch_size: 64                 # larger = more in-batch negatives = harder task
  temperature: 0.07              # InfoNCE softmax temperature

retrieval:
  min_retrieval_score: 0.25      # cosine similarity threshold for initial candidate set
  min_rerank_score: 0.25         # cross-encoder sigmoid score threshold (applied before blending)
  top_k_retrieve: 50             # max candidates fed to the reranker after filtering
  top_k_rerank: 10               # final output size
  use_rating_reranking: true     # blend bayesian_avg rating into final score
  rating_weight: 0.5             # weight for bayesian_avg in the blend
  use_diversity: true            # enforce system coverage and author cap in output
```

---

## Experimental run

The following summarises the full end-to-end experimental run on the IFDB dataset.

### Step 1 — Extract

```
games=15,342  reviews=79,225  users=19,674  playedgames=70,088
```

### Step 2 — Preprocess

```
Strict game docs (min_reviews=2):    5,974 games
Retrieval game docs (min_reviews=0): 9,774 games

Interactions: 53,272 total — 25,491 positive / 27,781 negative
User split:   2,237 users total / 1,960 with test items
              2,886 val interactions / 3,306 test interactions

Training user profiles:   2,124 users
Retrieval user profiles:  3,122 users
Training positives:      19,299 pairs across 2,237 users and 6,281 games
```

### Step 3 — Train bi-encoder (3 epochs, batch=64, MiniLM-L6-v2)

| Epoch | Loss | Val Recall@10 | Val MRR |
|---|---|---|---|
| 1 | 3.3281 | 0.1814 | 0.0839 |
| 2 | 3.0411 | 0.1881 | 0.0843 |
| 3 | 2.9544 | 0.1933 | 0.0860 |

Training time: ~15 min (Apple M-series GPU via MPS).

### Step 4 — Train reranker (2 epochs, batch=16, ms-marco-MiniLM-L6-v2)

```
Examples: 40,577 total — 17,402 positive / 23,175 negative
Training loss: 0.6307  (started at 0.8616, converged steadily)
Training time: ~23 min (Apple M-series GPU via MPS).
```

### Step 5 — Build index

```
Doc encoder:   encoded 9,774 games → embeddings shape (9774, 384)
Query encoder: encoded 9,774 game profile texts → game_query_embs.npy
FAISS index:   flat inner-product (exact search), saved to outputs/
```

### Step 6 — Evaluate (retrieval only, 1,856 test users)

| Metric | Score |
|---|---|
| MRR | 0.2586 |
| Recall@1 | 0.1883 |
| Recall@5 | 0.3154 |
| Recall@10 | 0.3661 |
| Recall@20 | 0.4096 |
| Recall@50 | 0.4806 |
| NDCG@1 | 0.1923 |
| NDCG@5 | 0.2591 |
| NDCG@10 | 0.2758 |
| NDCG@20 | 0.2873 |
| NDCG@50 | 0.3024 |

---

## Design notes

**Why asymmetric encoders?**
The query side (user profiles: short, tag-heavy) and the document side (game descriptions: longer, prose-heavy) have different distributional properties. Separate fine-tuned weights let each tower specialise while both start from the same strong pre-trained checkpoint.

**Why relative interaction labelling?**
Absolute thresholds (e.g. "≥ 4★ = positive") ignore game quality. A 3★ rating for a 2.5★ game signals that the game appealed to the user more than expected; the same 3★ rating for a 3.5★ game signals that the game appealed to the user less than expected. Bayesian-weighted rating labels help make the signal consistent across games with different rating distributions.

**FAISS index choice:**
`IndexFlatIP` gives exact cosine search with millisecond latency for ~10k games. Switching to `IndexHNSWFlat` (`index_type: hnsw` in config) would trade a small accuracy drop for sub-linear query time, useful if the corpus grows to hundreds of thousands of items.

**Why two game/profile sets?**
The strict training set (min_reviews ≥ 2) ensures enough signal per game for meaningful interaction labels. The broader retrieval set (min_reviews = 0) maximises coverage for users, including newly-added games with no ratings yet.

---

## Possible extensions

- **Larger base models** — improve quality by swapping out `all-MiniLM-L6-v2` for a stronger bi-encoder like `all-mpnet-base-v2` and/or swapping out `ms-marco-MiniLM-L6-v2` for a stronger cross-encoder like `DeBERTa-v3-base` (the tradeoff is computational cost and latency, which matters if the goal is to release this as a lightweight deployment)
- **Session-based data** — sliding window over a user's recent positive ratings rather than a full profile aggregate
