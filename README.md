# IFDB Interactive Fiction Recommender

A full retrieval-and-ranking pipeline built on the [IFDB](https://ifdb.org) Interactive Fiction Database. Community ratings drive training of an **asymmetric two-tower bi-encoder** for candidate retrieval, followed by a fine-tuned **cross-encoder reranker**, indexed via **FAISS**. Four ways in — by game, by author, by reviewer, or by picking systems and tags — with hard filters and post-rerank diversity. Ships with a Gradio app; the three enumerable modes are precomputed and served as lookups.

---

## Dataset

IFDB is a community-curated catalog of interactive fiction works with rich metadata and user reviews. The tables used by this pipeline are from the most recent [IFArchive](https://ifarchive.org/if-archive/info/ifdb/ifdb-archive-20260301.zip) dump (current version was archived on 03-01-2026), and contain the following data:

| Table | Rows | Role |
|---|---|---|
| `games` | 15,544 | Game metadata — title, author, system, genre, tags, description |
| `reviews` | 80,244 | Explicit ratings (1–5★) — primary supervision signal |
| `users` | 20,024 | User accounts — used for profile construction and filtering |
| `playedgames` | 71,033 | Implicit engagement — used to suppress already-played games from recommendations |

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
                          ──► [ Cross-encoder Reranker ]──► scored pool
                               fine-tuned                        │
                               ms-marco-MiniLM-L6-v2    filter by CE score
                                                                 │
                                                          cached per query
                                                                 │
                                                    ┌── hard filters ──┐
                                                    │  year / author   │
                                                    │  system / tags   │
                                                    │  rating / count  │
                                                    └──────────────────┘
                                                                 │
                                                       diversifying step
                                                       (≤2 per author; all
                                                        user's top systems)
                                                                 │
                                                         top_k_rerank results
```

The whole candidate pool is reranked, not a truncated slice of it. Cosine rank and cross-encoder rank correlate only weakly (Spearman ρ ≈ 0.22 on a sample query), so truncating first would discard most of what the reranker would have chosen — and, because filtering used to run before that truncation, would let a filter silently change which candidates were scored at all. See [Why the whole pool is reranked](#why-the-whole-pool-is-reranked).

### Two-tower bi-encoder (asymmetric)

The query tower and document tower start from the same `all-MiniLM-L6-v2` checkpoint but are fine-tuned independently with separate weights. This asymmetric design lets each tower specialise for its input domain:

- **Query encoder** — encodes profile-format text (`"Systems: X. Tags: a, b, c"`), whether from a user profile, a game, or a UI's pickers
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
final_score = (1 - rating_weight) * relevance + rating_weight * (bayesian_avg / 5.0)
```

Relevance answers "does this match the query?" and knows nothing about quality; the rating term supplies the quality signal the cross-encoder cannot see. Both components are shown in the results table (`Relev.` and `Rating`) so their trade-off is visible.

Two properties of this formula are deliberate, and both were arrived at by measurement:

**The terms stay on their own absolute scales.** Rescaling them within each query's candidate pool would push the top candidate toward 1.0 even for a niche query with nothing genuinely relevant in it. Absolute scales let a weak pool score like a weak pool.

**The rating term is deliberately the weaker one.** Bayesian smoothing squeezes ratings into roughly 0.54–0.81 while relevance spans nearly 0–1, so rating moves the score about 3× less than `rating_weight` implies. That imbalance is load-bearing rather than a bug: equalising the two terms costs −0.015 NDCG@10 (95% CI [−0.024, −0.006]) over 300 held-out users, and pushing rating harder still (`rating_weight` 0.7) costs −0.028. Rating is the noisier signal and earns less influence than its weight suggests. Measured weights 0.3 and 0.5 are statistically indistinguishable; above 0.5 quality degrades.

### Diversifying steps

After scoring, results pass through diversifying steps that:
1. Cap any individual author at `max_author_appearances=2` across the result list
2. Ensure coverage of all top systems drawn from the user's profile (for `userid` query mode)

Both run on the scored list, so an author's *best* games survive the cap rather than whichever of their games happened to rank highest by cosine.

---

## Repository layout

```
if-recommender/
├── app.py                        # Gradio front-end
├── config.yaml                   # All tunable parameters
├── pyproject.toml / uv.lock
│
├── dump/                         # Raw IFDB .sql.gz dump (git-ignored)
├── data/                         # Extracted + preprocessed Parquet files
├── models/                       # Fine-tuned model weights: query_encoder, doc_encoder, reranker
├── outputs/                      # FAISS index, embedding arrays, and ID maps
│
├── src/
│   ├── db/
│   │   ├── connector.py          # SQLAlchemy MySQL connection
│   │   └── container.py          # Serves the IFDB .sql.gz dump from a throwaway MariaDB container
│   ├── data/
│   │   ├── columns.py            # Original vs. `_clean` column naming convention
│   │   ├── loader.py             # MySQL → Parquet extraction
│   │   ├── preprocessor.py       # Game docs, user/author profiles, display normalisation, splits
│   │   └── dataset.py            # PairDataset for bi-encoder training
│   ├── index/
│   │   └── faiss_index.py        # GameIndex: build / search / save / load
│   ├── pipeline/
│   │   ├── retriever.py          # Stage 1: ANN retrieval + hard filtering + author cap
│   │   └── ranker.py             # Stage 2: cross-encoder reranking, diversity, eval metrics
│   └── utils/
│       └── env.py                # Logging setup, .env loading
│
└── scripts/
    ├── 01_extract.py             # IFDB dump (or live MySQL) → data/*.parquet
    ├── 02_preprocess.py          # Game docs (strict + retrieval), interactions, user profiles
    ├── 03_train_two_tower.py     # Fine-tune asymmetric bi-encoder (query + doc towers)
    ├── 04_train_reranker.py      # Fine-tune cross-encoder reranker
    ├── 05_build_index.py         # Encode all games → FAISS index + embedding files
    ├── 06_run_pipeline.py        # Interactive demo + offline evaluation
    └── 07_precompute.py          # Precompute userid / game_id / author_id rankings
```

---

## Quickstart

### Prerequisites

```bash
# Python 3.10+, uv recommended
uv sync

# The IFDB database dump, saved as dump/ifdb-archive.sql.gz (git-ignored).
# Path is configurable via database.dump.path in config.yaml.

# Docker (or podman) on PATH — step 1 loads the dump into a temporary
# MariaDB container and removes it again once the tables are extracted.

# Optional: HuggingFace token for model downloads
# Create a .env file and set HF_TOKEN
```

No permanent database server is needed. If you would rather read from a MySQL
you already run, see [Reading from a live server](#reading-from-a-live-server).

### Run the pipeline

```bash
# 1. Extract tables from the dump → data/
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

# 6b. Interactive demo (terminal)
uv run scripts/06_run_pipeline.py

# 7. Precompute rankings for the lookup-served modes (~5 h)
uv run scripts/07_precompute.py --mode all --top-n 500

# 8. Web app
python app.py
```

---

## Data extraction

`scripts/01_extract.py` is the only step that touches a database. It reads the raw IFDB mysqldump without asking you to install or maintain a server:

1. Starts a detached MariaDB container (`mariadb:10.5.26`, pinned to the dump's server version) on a runtime-assigned port bound to `127.0.0.1`, datadir on tmpfs, durability settings relaxed for a fast one-shot import.
2. Streams `dump/ifdb-archive.sql.gz` into the container's client, decompressing on the fly — the dump is never expanded to disk.
3. Extracts `games`, `reviews`, `users`, and `playedgames` into `data/*.parquet` over SQLAlchemy/PyMySQL.
4. Writes `data/manifest.json`, then removes the container and its volumes.

The whole run takes about 20 seconds. Nothing survives it: with the datadir in RAM there is no state to leak between runs, which is also what makes this safe to run in CI. While iterating, keep the loaded server around:

```bash
uv run scripts/01_extract.py --keep-container   # container survives the run
uv run scripts/01_extract.py --overwrite        # reuses it, skips the import
docker rm -f ifdb-extract                       # done — clean up
```

Dump location, image, container name, tmpfs size, and timeouts live under `database.dump` in `config.yaml`. A plain `.sql` file works as well as `.sql.gz`.

### Reproducibility

Re-extracting the same dump produces byte-identical Parquet files, so "did the data change?" is a digest comparison rather than a diff you can't read. Two things buy that: every table is read with an `ORDER BY` on its primary key (`playedgames` has none, so it sorts on all columns), and the pandas/pyarrow version stamp is stripped from the Parquet schema metadata before writing. Byte-stability holds for a given pyarrow version — an upgrade can still change the encoding.

`data/manifest.json` records the dump's SHA-256 alongside the row count, columns, and SHA-256 of each extracted table. It carries no timestamp, so two manifests compare equal exactly when the datasets do.

```json
{
  "source": { "path": "dump/ifdb-archive.sql.gz", "sha256": "b7ce4b69f2b8e747…" },
  "tables": { "games": { "file": "games.parquet", "rows": 15544, "sha256": "55580c800cfe…", "columns": [...] } }
}
```

### Reading from a live server

```bash
export IFDB_DB_PASSWORD=yourpassword   # or set database.password in config.yaml
uv run scripts/01_extract.py --source mysql
```

This path uses the `database.host` / `port` / `user` / `database` settings (default `root@localhost:3306/ifarchive`) and needs no container runtime.

---

## Evaluation

`scripts/06_run_pipeline.py --mode evaluate` reports Recall@K, NDCG@K, and MRR against held-out test-split positives for each user, at K ∈ {1, 5, 10, 20, 50}.

By default it measures raw retrieval only, which takes seconds but understates the system. Add `--rerank` to score candidates exactly as the interactive pipeline does — slower (~51 min over 1,887 users), and the number you should quote. Both are reported in [Step 6](#step-6--evaluate-1887-test-users-5-skipped-for-missing-profiles).

---

## Interactive demo

The interactive demo supports three query types (selected at startup via `--query-type`):

| Mode | Input | Description |
|---|---|---|
| `text` (default) | `"Systems: twine. Tags: fantasy, slice of life, choice-based, graphics"` | Recommendations based on formatted text in the style of a user taste profile, which should include "Systems:" and "Tags:" lists. |
| `userid` | `"9rwstoqhvjcff8hf"` | Personalised recommendations from a user profile based on the user's positive rating history |
| `game_id` | a game ID string | "More like this" recommendations based on the seed game's profile |
| `author_id` | an author name | Games in the spirit of an author's catalogue, with their own games excluded |

```bash
uv run scripts/06_run_pipeline.py --query-type userid
uv run scripts/06_run_pipeline.py --query-type game_id
```

In `userid` mode, games the user has already reviewed or played are automatically suppressed from results. The user's display name and taste profile are shown before results.

### Query, then refine

A query is scored once, then refined as many times as you like:

```
Query > 9rwstoqhvjcff8hf
        … top 25 results …
  Refine these results with filters, e.g.  year:2020-2026; rating:3.5; count:2
  Keys: year, author, system, tags, rating, count   (each entry replaces the last)
  'clear' show all again  ·  'back' new query  ·  'quit' exit
Filter > year:2020-2026; rating:3.5
        … the same ranking, narrowed …
Filter > back
Query >
```

Filters apply to the **already-scored** ranking, so refining never re-runs the search. Each entry replaces the previous one rather than stacking, `clear` returns to the unfiltered list, and a filter matching nothing says so instead of printing an empty table.

The practical consequence is that scores mean the same thing across refinements. A game's score is identical filtered or not, ordering is preserved, and filtering can only remove entries — though games ranked below the visible top 25 are promoted into view as higher-ranked ones are filtered out.

### Reading the results table

| Column | Meaning |
|---|---|
| `Score` | The blended ranking score — what the list is sorted by |
| `Relev.` | Cross-encoder relevance, 0–1: how well the game matches the query |
| `Rating` | Raw community average and review count, or `—` when nobody has rated it |

`Score` is not reproducible from the other two columns by hand: the blend uses the *smoothed* `bayesian_avg`, while `Rating` displays the raw average that IFDB shows. The two columns explain why something ranked where it did — a high-relevance/low-rating game versus a well-loved but looser match — rather than restating the arithmetic.

`Relev.` is a ranking score, not a probability: 0.85 does not mean "85% likely to suit you". It is strongly monotonic — the top band contains 11× more genuine matches than average — but see [How much does the score actually mean?](#design-notes) before reading it as a confidence level.

### Hard filters

Filters narrow the ranking you are already looking at. They are AND-ed together and applied after scoring, so they never change which candidates were ranked.

| Key | Example | Matches against |
|---|---|---|
| `year` | `year:2010-2020` | Publication year, inclusive |
| `author` | `author:emily short` | Any individual author name |
| `system` | `system:inform` | Any individual system name |
| `genre` | `genre:horror` | Any genre value |
| `tags` | `tags:IFComp 2025` | Game tags; each listed tag must match |
| `rating` | `rating:3.5` | Raw community average; unrated games excluded |
| `count` | `count:10` | Number of ratings |

**Filters match the original IFDB values, not the normalised `_clean` ones**, so a filter only ever matches something visible in the results. Two consequences motivated that:

- `tags:IFComp 2025` works. Competition tags are stripped from `tags_clean`, so filtering the cleaned values could never match them.
- `tags:slice of life` no longer returns games that merely have *genre* "Slice of life". Genre is folded into `tags_clean` but is not shown in the tags column, which made those look like false positives. Use `genre:` for that field.

Matching is case-insensitive and by substring, so `system:inform` finds `Inform 7` and `tags:xyzzy` finds all 44 XYZZY tags. Quotes are accepted and ignored.

`rating` compares the raw community average rather than `bayesian_avg`. Smoothing pulls low-review games toward a 3.5 prior, so filtering on it would return games whose actual average is below what was asked for, and would admit unrated games on the strength of the prior alone. The results table shows that same raw average, with `—` for unrated games.

The trade-off: a single 5★ review reads as 5.0 and clears any threshold. Pair `rating:` with `count:` for a floor backed by a real sample.

---

## Data preparation details

### Game documents

Two document sets are built during preprocessing:

- **Strict set** (`game_docs.parquet`) — games with ≥ `min_reviews_per_game` reviews; used for training interactions
- **Retrieval set** (`game_docs_retrieval.parquet`) — all valid games (including zero-review games); used for indexing and serving

A game is included only if its title, author, system, description, and tags are all non-empty.

Each game produces two text representations:
- **`doc_text`** — full document for the doc encoder: `"Title: … Author: … Systems: … Tags: … Description: …"`
- **`query_text`** — profile-format for game-ID queries: `"Systems: … Tags: …"` (no title/author/description)

#### Original vs. normalised columns

Normalisation helps the encoders and hurts the reader: a game IFDB lists as `Inform 7` normalises to `inform`, and its tags lose their casing and competition entries. Showing that back to a user makes the recommender look wrong.

So originals keep their own names and every normalised variant is written beside them with a `_clean` suffix ([src/data/columns.py](src/data/columns.py) owns the convention). **Display reads the plain names; models, filters, and profile building read the `_clean` ones.**

| Column | Displayed | Model-facing | Normalisation |
|---|---|---|---|
| author | `author` | `author_clean` | split on `,` `/` `and`, deduplicated, rejoined |
| system | `system` | `system_clean` | parentheticals and version numbers stripped, lowercased, split |
| tags | `tags` | `tags_clean` | genre folded in, competition tags dropped, lowercased, capped at 20 |
| genre | `genre` | — | folded into `tags_clean` |
| published | `published` | `year` | four-digit year extracted, used for `year:` range filters |

`title` is never transformed. Hard filters match on the normalised values, so `system:inform` still finds a game displayed as `Inform 7`. Consumers fall back to the original column when a `_clean` one is absent, so `game_docs` files written before this split keep working.

One deliberate exception: the **User Profile** panel shows normalised text, because it *is* the encoder input — it shows what was actually searched for, not an IFDB record.

### Author profiles

Each author gets a profile in the same shape, aggregated over the games they wrote rather than the games a user rated. That makes "something in this author's spirit" just another profile query through the same encoders, with their own catalogue excluded from results.

Single-game authors are included: their profile is byte-identical to that game's `query_text` for 4,711 of 4,713 cases, so the mode duplicates `game_id` for them — which is the point. Someone picking an author has no idea how many games they wrote, and being told to switch modes and look up a game ID would be a poor answer.

### Display normalisation

IFDB's free-text fields are inconsistent: the same tag appears in many casings, systems and genres use both commas and slashes as separators, and values repeat within one field. Before display, each of `tags`, `system` and `genre` is split, deduplicated, mapped to the community's dominant casing, and ordered by how many games use each value.

```
Educational, Slice of life   ->  Slice of life, Educational      (640 games vs 135)
Fantasy, mystery, romance    ->  Fantasy, Mystery, Romance
Drama / Political            ->  Drama, Political
dendry                       ->  Dendry
```

Tags split on commas only — `gay/queer protagonist` is a single tag — while systems and genres split on both. This is presentation only: filters still match the stored values, so a filter never misses a game because its casing differs from the canonical form.

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
  min_rerank_score: 0.10         # relevance floor; low enough for filter headroom, high enough to display
  top_k_retrieve: 50             # raw-retrieval evaluation only — the pipeline reranks the whole pool
  top_k_rerank: 25               # results per page
  rerank_pool_cap: 0             # max candidates scored live; 0 = no cap
  prefilter_by_tag: true         # drop candidates sharing no tag with the query (measured free)
  use_rating_reranking: true     # blend bayesian_avg rating into final score
  rating_weight: 0.5             # weight on bayesian_avg/5; 0.3-0.5 equivalent, above 0.5 hurts
  use_diversity: true            # enforce system coverage and author cap in output
```

---

## Experimental run

The following summarises the full end-to-end experimental run on the IFDB dataset.

### Step 1 — Extract

```
games=15,544  reviews=80,244  users=20,024  playedgames=71,033
```

### Step 2 — Preprocess

```
Strict game docs (min_reviews=2):    6,103 games
Retrieval game docs (min_reviews=0): 10,087 games

Interactions: 53,842 total — 25,784 positive / 28,058 negative
User split:   2,271 users total / 1,991 with test items
              2,913 val interactions / 3,343 test interactions

Training user profiles:   2,152 users
Retrieval user profiles:  3,175 users
Training positives:      19,528 pairs across 2,271 users and 6,348 games
```

### Step 3 — Train bi-encoder (3 epochs, batch=64, MiniLM-L6-v2)

| Epoch | Loss | Val Recall@10 | Val MRR | |
|---|---|---|---|---|
| 1 | 3.3233 | 0.1804 | 0.0857 | |
| 2 | 3.0308 | **0.2084** | **0.0923** | ← saved |
| 3 | 2.9514 | 0.1971 | 0.0901 | |

Training time: ~32 min (Apple M-series GPU via MPS). The training epochs dominate
at 6–16 min each; the between-epoch validation pass costs only 15–50 s despite
re-encoding all 10,087 game documents.

Validation peaked at epoch 2 while training loss kept falling through epoch 3, so
`03_train_two_tower.py` restored and saved the epoch-2 weights — worth +0.0113
Recall@10 (+5.7%) over the final epoch it would otherwise have shipped. Selection
is on Recall@10, tie-broken on MRR; `--save-last` restores the old behaviour.

The epoch that wins moves between runs (epoch 1 in an earlier run, epoch 2 here),
which is the argument for selecting a checkpoint rather than tuning the epoch count.

### Step 4 — Train reranker (2 epochs, batch=16, ms-marco-MiniLM-L6-v2)

```
Examples: 41,261 total — 17,697 positive / 23,564 negative
          (6,325 skipped — missing profile/doc)
Training loss: 0.6272
Training time: ~24 min (Apple M-series GPU via MPS).
```

### Step 5 — Build index

```
Doc encoder:   encoded 10,087 games → embeddings shape (10087, 384)
Query encoder: encoded 10,087 game profile texts → game_query_embs.npy
FAISS index:   flat inner-product (exact search), saved to outputs/
```

### Step 6 — Evaluate (1,887 test users, 5 skipped for missing profiles)

```bash
uv run scripts/06_run_pipeline.py --mode evaluate            # raw retrieval only (~25 s)
uv run scripts/06_run_pipeline.py --mode evaluate --rerank   # full pipeline (~51 min)
```

| Metric | Raw retrieval | Retrieval + reranking |
|---|---|---|
| MRR | 0.2574 | **0.2847** |
| Recall@1 | 0.1918 | 0.1909 |
| Recall@5 | 0.3119 | **0.3687** |
| Recall@10 | 0.3606 | **0.4315** |
| Recall@20 | 0.4099 | **0.4777** |
| Recall@50 | 0.4816 | **0.5545** |
| NDCG@1 | 0.1934 | 0.1955 |
| NDCG@5 | 0.2582 | **0.2889** |
| NDCG@10 | 0.2743 | **0.3097** |
| NDCG@20 | 0.2875 | **0.3221** |
| NDCG@50 | 0.3028 | **0.3392** |

The reranker is worth +20% Recall@10 and +13% NDCG@10 over raw retrieval — the figures in the right-hand column are what the interactive pipeline actually delivers. `Recall@1` and `NDCG@1` barely move, which is consistent with the reranker reordering the body of the list rather than changing which single game lands on top.

Earlier runs of this table reported raw retrieval only, understating the system throughout.

Against the April run (raw retrieval: MRR 0.2586, Recall@10 0.3661, NDCG@10 0.2758) the numbers are not directly comparable: the newer dump adds 313 games to the retrieval pool (9,774 → 10,087), so each query ranks against more distractors, and the test split itself differs.

---

## Web app

```bash
python app.py
```

A Gradio front-end with four entry points, styled as a terminal to suit the audience. Pick a search type, optionally narrow with filters, and page through results.

| Search type | Picker shows | Served from |
|---|---|---|
| game | `Title — Author (Year)`, 10,087 | precomputed |
| author | name and game count, 6,298 | precomputed |
| user | name and review count, 3,175 | precomputed |
| browse | systems and tags, multi-select | scored live, cached |

Pickers are typeaheads ordered by frequency, so opening one cold shows the most common entries first. Games carry author and year because 119 titles are shared by up to five different games — a bare title would make all but one unreachable.

Result titles link to `ifdb.org/viewgame?id=…`. The `genre`, `system`, `author` and `tags` filters accept free text as well as menu choices, so typing `xyzzy` and pressing return matches all 44 XYZZY tags by substring.

Three author entries are hidden from the *search* picker — both `Anonymous` variants and `Failbetter Games`. They sort to the top by game count but make poor seeds: an aggregate of 61 unrelated games, or a catalogue so system-specific that little outside it matches. They remain available as *filters*, which is a different job.

---

## Deployment

Sized for a CPU-only host such as a Hugging Face Space on the free tier (2 vCPU, 16 GB RAM).

### Precompute the enumerable modes

`userid`, `game_id`, and `author_id` queries draw from fixed key sets — every user with a profile, every game, every author — so their rankings are computed once offline and served as a lookup, with no cross-encoder work at request time:

```bash
uv run scripts/07_precompute.py --mode all --top-n 500
```

| Artefact | Rows | Keys | Median depth | Size |
|---|---|---|---|---|
| `data/precomputed_userid.parquet` | 816,351 | 3,170 | 222 | 14.4 MB |
| `data/precomputed_gameid.parquet` | 2,269,909 | 10,076 | 198 | 39.2 MB |
| `data/precomputed_authorid.parquet` | 1,436,646 | 6,291 | 202 | 24.0 MB |

Depth is what makes filtering usable: a `userid` list of 222 survives `year:2020-2026; rating:3.5` with 205 entries left. Two settings produce it, and both were measured free — `min_rerank_score: 0.10` (a relevance floor low enough to leave headroom, high enough that the displayed `Relev.` column never reads `0.0X`) and `prefilter_by_tag`, which keeps only candidates sharing at least one tag with the query. Together they give 1.87x the depth of a 0.30 floor at **identical** Recall/NDCG@10/25 over 300 held-out users.

The tag rule also earns its place on legibility: every stored result shares a visible tag with the query, so a recommendation always has a reason a user can check. Note it matches on `tags_clean`, which folds in `genre` — a UI should show both fields so the matched term is always visible.

Not everything is deep: 3.2% of users, 8.7% of games, and 9.1% of authors have fewer than 25 stored results, so a UI should report the true count rather than implying a full page.

The pipeline loads these at startup and uses them automatically; if a file is missing, or a key is absent, that query falls back to scoring live. Verified: an uncapped live run reproduces the precomputed ranking exactly, so the cache is a shortcut rather than a second code path.

**Re-run this after retraining the reranker or bi-encoder, or rebuilding the index** — the rankings are tied to the models that produced them.

Full run is ~5 hours on Apple M-series GPU (1 h users, 2.4 h games, 1.6 h authors). Rows stream to Parquet in batches and publish by atomic rename, so a reader never sees a half-written file and a failed run cannot destroy the previous artefact. Rows stream to Parquet in batches, so memory stays flat regardless of job size.

### Resource envelope

**Memory — ~1,500 MB** with everything resident: dataframes, FAISS index, both embedding arrays, both models, and all three precomputed tables. Comfortable against 16 GB.

The browse cache adds ~85 KB per entry (measured over retained objects, not RSS), so the 2,048-entry cap is ~171 MB. The number worth watching in production is neither of those: PyTorch's allocator grew ~470 MB across 28 scorings in testing and had not clearly plateaued, and it does so whether or not results are cached. A larger cache *reduces* that pressure, since every hit is a scoring that never runs.

**CPU** is the real constraint. The cross-encoder runs at ~135 pairs/s on 2 threads of Apple silicon; a free-tier x86 vCPU is plausibly 1.5–3× slower. That budget is fixed and shared, so concurrent requests divide it — three simultaneous users each wait roughly three times as long. Consider serialising inference (`concurrency_limit=1` in Gradio) so contention becomes a visible queue rather than everyone slowing down at once.

Precomputed modes bypass this entirely; only `browse` queries consume CPU, and those are cached two ways — per session while a user narrows filters, and process-wide across users by (systems, tags). A repeat browse query costs 0.02 s against 4 s cold.

Scoring depends only on the query, never on the filters, so changing a filter re-filters an existing ranking rather than re-scoring it. `rerank_pool_cap` bounds their cost (0 = no cap). Capping to 200 makes queries 2–4× faster and is quality-neutral against held-out positives, but returns only ~37% the same games as the uncapped ranking, so text mode would diverge from the precomputed modes. It ships uncapped for consistency.

### Building queries from a UI

Free text is **not** supported: both encoders train only on `(profile_text, doc_text)` pairs, and prose lands measurably outside that distribution (best cosine 0.447 versus 0.670 for profile format). A UI should offer system and tag pickers and assemble the query:

```python
from src.data.preprocessor import format_profile_text, profile_vocabulary

systems, tags = profile_vocabulary(game_docs)          # options, by frequency
query = format_profile_text(["twine"], ["fantasy", "horror"])
# "Systems: twine. Tags: fantasy, horror"
```

Both helpers read the `_clean` columns, so the strings a user picks are exactly the ones the encoders were trained on — IFDB's own casing (`Inform 7`, `IFComp 2019`) is not. `format_profile_text` is the same function that builds user and game profiles during preprocessing, so all three query sources are identical by construction.

### Choice of cross-encoder

Lighter rerankers were fine-tuned and evaluated on identical candidate pools across 300 held-out users. None matched `MiniLM-L6`:

| Model | Params | pairs/s | Recall@10 | vs L6 (95% CI) |
|---|---|---|---|---|
| **MiniLM-L6** (current) | 22.7M | 140 | **0.4447** | — |
| MiniLM-L4 | 19.2M | 221 | 0.4267 | −0.0180 [−0.040, +0.003] n.s. |
| MiniLM-L2 | 15.6M | 416 | 0.3952 | −0.0495 [−0.080, −0.018] |
| TinyBERT-L2 | 4.4M | 1,374 | 0.4127 | −0.0319 [−0.061, −0.003] |

L2 and TinyBERT degrade significantly despite being 3× and 10× faster. L4's interval spans zero, but that is insufficient power at n=300 rather than proven equivalence, and both point estimates lean negative for only 1.58×. Not worth the trade.

---

## Design notes

**Why asymmetric encoders?**
The query side (user profiles: short, tag-heavy) and the document side (game descriptions: longer, prose-heavy) have different distributional properties. Separate fine-tuned weights let each tower specialise while both start from the same strong pre-trained checkpoint.

**Why relative interaction labelling?**
Absolute thresholds (e.g. "≥ 4★ = positive") ignore game quality. A 3★ rating for a 2.5★ game signals that the game appealed to the user more than expected; the same 3★ rating for a 3.5★ game signals that the game appealed to the user less than expected. Bayesian-weighted rating labels help make the signal consistent across games with different rating distributions.

**FAISS index choice:**
`IndexFlatIP` gives exact cosine search with millisecond latency for ~10k games. Switching to `IndexHNSWFlat` (`index_type: hnsw` in config) would trade a small accuracy drop for sub-linear query time, useful if the corpus grows to hundreds of thousands of items.

**Why the whole pool is reranked**
The bi-encoder is a recall device and the cross-encoder is the precision stage, but on this data they barely agree on ordering — Spearman ρ ≈ 0.22 between cosine rank and cross-encoder rank. Reranking only the top 50 by cosine therefore captured just 1 of the 25 results a full rerank would return.

Reranking is clearly worth doing: over the full test split it adds +21% Recall@10 and +14% NDCG@10 (see [Step 6](#step-6--evaluate-1887-test-users-5-skipped-for-missing-profiles)). Reranking *deeper*, however, is worth nothing. On a 200-user sample with paired bootstrap CIs, 50 → 200 candidates moves NDCG@25 by +0.002 (interval spans zero) and 200 → 800 is flat to slightly negative.

Depth is therefore not a quality lever — it is a consistency one. Scoring the entire pool (bounded near 1,500 candidates by `min_retrieval_score`, ~8 s per query) makes a filter narrow a fixed ranking rather than change which candidates get scored at all, at no measurable cost in quality.

**How much does the score actually mean?**
Measured on 300 held-out users — 150,432 (user, candidate) pairs, base rate 0.22%:

| Relevance bucket | Pairs | Genuine held-out positives | Lift |
|---|---|---|---|
| 0.0–0.1 | 15,275 | 0.026% | 0.1× |
| 0.3–0.4 | 22,766 | 0.053% | 0.2× |
| 0.5–0.6 | 14,896 | 0.175% | 0.8× |
| 0.7–0.8 | 7,306 | 0.602% | 2.7× |
| 0.8–1.0 | 7,151 | 2.489% | **11.3×** |

Three conclusions, in decreasing order of confidence:

*Within a query the score is strongly discriminative.* Hit rate rises monotonically across every bucket, ending at an 11.3× lift. The reranker is doing real work.

*It is not a probability.* A game scoring 0.85 is not 85% likely to be a hit — it is about 2.5% likely. The sigmoid is monotonic, not calibrated, because the base rate is 0.22%. Read the number as a rank, never as a confidence.

*Across queries it transfers only loosely.* Splitting users by their score ceiling, the same bucket means very different things: in the 0.5–0.6 band, candidates belonging to low-ceiling users are hit 0.235% of the time versus 0.041% for high-ceiling users — nearly 6× apart. What does carry across queries is the ceiling itself; users in the top ceiling quartile genuinely get roughly twice the Precision@10 of the bottom quartile (0.055 vs 0.028), though the correlation is weak (Spearman ≈ +0.2) and flattens at the very top. So a low-scoring result set is a real signal that this query has few good matches — which is exactly why the blend keeps absolute scales instead of rescaling each pool to fill 0–1.

**Why two game/profile sets?**
The strict training set (min_reviews ≥ 2) ensures enough signal per game for meaningful interaction labels. The broader retrieval set (min_reviews = 0) maximises coverage for users, including newly-added games with no ratings yet.

---

## Possible extensions

- **Larger base models** — improve quality by swapping out `all-MiniLM-L6-v2` for a stronger bi-encoder like `all-mpnet-base-v2` and/or swapping out `ms-marco-MiniLM-L6-v2` for a stronger cross-encoder like `DeBERTa-v3-base` (the tradeoff is computational cost and latency, which matters if the goal is to release this as a lightweight deployment)
- **Session-based data** — sliding window over a user's recent positive ratings rather than a full profile aggregate
- **Faster bi-encoder training** — wall time is dominated by the training epochs themselves (6–16 min each); the between-epoch validation pass costs only 15–50 s despite re-encoding all 10,087 game documents. Gains would have to come from the training loop — larger batches, mixed precision, or gradient accumulation — not from cheaper validation
