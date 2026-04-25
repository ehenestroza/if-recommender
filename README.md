# IFDB Two-Tower Retrieval + Reranking Pipeline

A full retrieval-and-ranking system built on the [IFDB](https://ifdb.org)
Interactive Fiction Database. Users, games, and community ratings serve as
the training signal for a **sentence-transformers two-tower bi-encoder**
with a **cross-encoder reranker** second stage, indexed via **FAISS**.

---

## Why this dataset?

IFDB is a community-curated catalog of ~20 000 interactive fiction works with
~90 000 user reviews and ratings (1–5 stars). It provides:

| Signal | Table | Use |
|---|---|---|
| Explicit ratings | `reviews` | Primary supervision (positives ≥ 4★, negatives ≤ 2★) |
| Rich text | `games.desc`, `gametags` | Item-tower document construction |
| Implicit engagement | `wishlists`, `playedgames` | Supplemental positive signal |

---

## Architecture

```
                        ┌─────────────────────────────────┐
  User profile text ──► │  Query Tower (bi-encoder)       │
  (derived from their   │  all-MiniLM-L6-v2               │──► query_emb (384d)
   liked games' tags)   └─────────────────────────────────┘
                                          │
                                    cosine similarity
                                          │
                        ┌─────────────────────────────────┐
   Game document   ──►  │  Item Tower (shared weights)    │──► item_emb (384d)
   (title + author +    │  all-MiniLM-L6-v2               │
    genre + tags + desc)└─────────────────────────────────┘
                                          │
                              FAISS ANN (top-100)
                                          │
                        ┌─────────────────────────────────┐
  (query_text,     ──►  │  Cross-encoder Reranker         │──► top-10 results
   candidate_doc)       │  ms-marco-MiniLM-L-6-v2         │
                        └─────────────────────────────────┘
```

### Training

**Loss:** `MultipleNegativesRankingLoss` (symmetric InfoNCE with in-batch negatives)

Each batch of B positive `(user_profile, game_doc)` pairs is treated as B×B
pairwise comparisons — every other positive in the batch acts as a hard
negative. This is the same paradigm used in DPR, SimCSE, and E5.

**Query representation at training:**  
> *"A player who enjoys: mystery, puzzle, historical, detective, Victorian"*  
(Aggregated top-20 tags from the user's positively-rated games.)

This shared text-query representation means the model generalises to
**cold-start text queries** at inference with no additional engineering.

---

## Repository layout

```
ifdb-retrieval/
├── config.yaml                  # All tunable knobs
├── requirements.txt
│
├── src/
│   ├── db/
│   │   └── connector.py         # SQLAlchemy MySQL connection
│   ├── data/
│   │   ├── loader.py            # MySQL → Parquet extraction
│   │   ├── preprocessor.py      # Game docs, user profiles, interactions, splits
│   │   └── dataset.py           # PyTorch PairDataset / TripletDataset
│   ├── models/
│   │   └── two_tower.py         # TwoTowerModel wrapper (encode / save / load)
│   ├── index/
│   │   └── faiss_index.py       # GameIndex: build / search / save / load
│   └── pipeline/
│       ├── retriever.py         # Stage 1: ANN retrieval (text / userid / game_ids)
│       └── ranker.py            # Stage 2: cross-encoder reranking + eval metrics
│
└── scripts/
    ├── 01_extract.py             # MySQL → Parquet
    ├── 02_preprocess.py          # Game docs, interactions, profiles, popularity
    ├── 03_train_two_tower.py     # Fine-tune bi-encoder
    ├── 04_build_index.py         # Encode all games → FAISS index
    └── 05_run_pipeline.py        # Interactive demo + offline evaluation
```

---

## Quickstart

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# MySQL running locally with the IFDB dump loaded
# (default: root@localhost:3306/ifarchive, no password)
# Override password via env var: export IFDB_DB_PASSWORD=yourpassword
```

### Run the pipeline

```bash
# 1. Pull relevant tables from MySQL → data/
python scripts/01_extract.py

# 2. Build game docs, interaction matrix, user profiles, popularity
python scripts/02_preprocess.py

# 3. Fine-tune the bi-encoder (3 epochs, ~10-30 min on CPU)
python scripts/03_train_two_tower.py

# 4. Encode all games + build FAISS index → outputs/
python scripts/04_build_index.py

# 5a. Interactive query demo
python scripts/05_run_pipeline.py

# 5b. Offline evaluation on test split
python scripts/05_run_pipeline.py --mode evaluate
```

---

## Query types

The pipeline supports three query modes (set at runtime):

| Mode | Example input | When to use |
|---|---|---|
| `text` (default) | `"short puzzle game set in space"` | Cold-start / semantic search |
| `userid` | `"nufzrftl37o9rw5t"` | Personalised recommendation |
| `game_ids` | `["ju778uv5xaswnlpl", "4glrrfh7wrp9zz7b"]` | "More like these" |

```bash
# Text query (default)
python scripts/05_run_pipeline.py --query-type text

# User-based personalised recommendation
python scripts/05_run_pipeline.py --query-type userid
```

---

## Configuration

All tunable parameters live in `config.yaml`. Key settings:

```yaml
model:
  base_model: "sentence-transformers/all-MiniLM-L6-v2"  # swap for larger models
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

training:
  epochs: 3
  batch_size: 64            # larger = more in-batch negatives = harder training
  min_rating_positive: 4    # reviews ≥ 4★ are treated as positives
  max_rating_negative: 2    # reviews ≤ 2★ are treated as negatives

retrieval:
  top_k_retrieve: 100       # FAISS candidates
  top_k_rerank: 10          # cross-encoder output
  index_type: "flat"        # "hnsw" for faster approximate search at scale
```

---

## Evaluation metrics

`scripts/05_run_pipeline.py --mode evaluate` reports:

| Metric | Description |
|---|---|
| Recall@K | Fraction of held-out positive games found in top-K |
| NDCG@K | Position-weighted quality of the ranking |
| MRR | Mean Reciprocal Rank of the first relevant item |

Evaluated at K ∈ {1, 5, 10, 20, 50} against the test-split held-out users.

---

## Design notes

**Why shared encoder weights?**  
At this scale (queries and items are both natural-language text), sharing
weights is more parameter-efficient and avoids the need to warm-start two
separate towers. The query representation is derived from game tag vocabulary,
which is the same distributional space as the item documents.

**Why in-batch negatives vs. explicit hard negatives?**  
With batch size 64 each step sees 63 in-batch negatives — many of which are
semantically close (e.g. other mystery games), making the task genuinely hard
without any explicit mining. The `TripletDataset` class in `src/data/dataset.py`
supports an alternative hard-negative regime using explicit low-rated interactions.

**FAISS index choice:**  
`IndexFlatIP` gives exact search with millisecond latency for ~20k games.
Switching to `IndexHNSWFlat` (set `index_type: hnsw` in config) trades a small
accuracy drop for sub-linear query time — relevant when the corpus grows.

---

## Possible extensions

- **Multi-vector retrieval** (ColBERT-style) — one embedding per game sentence
- **Session-based retrieval** — sliding window over a user's recent plays
- **MLflow / W&B experiment tracking** — log training runs, eval metrics, embeddings
- **Spark preprocessing** — parallelize game-document construction for larger corpora
