# IF recommender

Game recommendations for interactive fiction, built on the [Interactive Fiction Database](https://ifdb.org).

Pick a game you liked, an author, a reviewer whose taste you trust, or just the systems and tags you're in the mood for. You get back a ranked list of games that feel similar — not a keyword search.

---

## Four ways to search

| Mode | You pick | You get |
|---|---|---|
| **game** | a game | games with a similar feel |
| **author** | an author | games in their spirit, excluding their own |
| **reviewer** | an IFDB reviewer | what suits their taste, from their rating history |
| **vibe** | systems and tags | games matching that combination |

All four work the same way underneath: your choice becomes a short profile like `twine, ink // mystery, surreal`, and games are ranked by how well they match it.

Results can then be narrowed by genre, system, author, tags, rating, review count and year. Filters accept typed fragments, so `xyzzy` matches every XYZZY tag and `inform` matches every version of Inform.

Two filters start switched on: rating ≥ 3.0 and at least one rating. Ranking by relevance alone surfaces plenty of obscure games, and a page led by unrated ones asks you to gamble with no information. Drop the rating filter to 0.0 to open up the genuinely unknown tail.

---

## Running it

```bash
uv sync
python app.py
```

That's enough if the trained models and data are already in place. To build everything from scratch you need the [IFDB archive dump](https://ifarchive.org/indexes/if-archive/info/ifdb/) saved as `dump/ifdb-archive.sql.gz`, plus Docker:

```bash
uv run scripts/01_extract.py           # dump → data/*.parquet  (~20 s)
uv run scripts/02_preprocess.py        # game docs, profiles, train/test splits
uv run scripts/03_train_two_tower.py   # ~32 min
uv run scripts/04_train_reranker.py    # ~24 min
uv run scripts/05_build_index.py       # FAISS index
uv run scripts/07_precompute.py        # ~5 h, optional but makes the app instant
```

No database server needed — step 1 loads the dump into a throwaway MariaDB container and removes it afterwards.

There's also a terminal version with the same four modes, if you prefer it:

```bash
uv run scripts/06_run_recommender.py
```

---

## How it works

Two stages, both fine-tuned on IFDB's own review data:

1. **Retrieval** — a two-tower bi-encoder embeds your query and every game, and FAISS returns everything above a similarity threshold. Fast and broad.
2. **Reranking** — a cross-encoder scores each candidate against your query directly. Slower and more accurate, so it only sees the shortlist.

A blended score — relevance plus community rating — decides which candidates survive, but it is not what you see. Results are **ordered by relevance alone**, with the raw community rating shown beside it.

That split is deliberate. One number combining match and popularity can't be read as either, and ordering by it pushes already well-loved games to the top — the opposite of what a discovery tool is for. On one test query, ordering by score gave a top-25 averaging 4.39 stars against 3.67 for relevance: the same games, but the familiar ones no longer crowd out the rest.

Reranking is worth about **+20% Recall@10** over retrieval alone, measured against held-out reviews.

Because `game`, `author` and `reviewer` searches come from a fixed list, their rankings are computed ahead of time and served as a lookup — those are instant. Only `vibe` is scored live.

---

## Layout

```
app.py            Gradio web app
config.yaml       All tunable parameters
data/             Parquet tables and precomputed rankings
models/           Fine-tuned encoders and reranker
outputs/          FAISS index and embeddings
src/              Library code
scripts/          The pipeline, numbered in order
deploy/           cloud-init for a single-VM deployment
```

---

## Configuration

Everything tunable lives in `config.yaml`. The settings worth knowing:

| Setting | Default | What it does |
|---|---|---|
| `min_retrieval_score` | 0.25 | How similar a game must be to make the shortlist |
| `min_rerank_score` | 0.10 | Relevance floor for what gets stored and shown |
| `rerank_pool_cap` | 1000 | Most candidates scored live, per `vibe` query |
| `rating_weight` | 0.5 | How much rating counts toward *selecting* candidates — never toward their displayed order |
| `use_diversity` | true | Caps repeat authors and covers your top systems |
| `top_k_rerank` | 25 | Results per page in the terminal app (the web app has its own 10/20/50 picker) |
| `quantize_reranker` | `"auto"` | int8 cross-encoder where it helps — 2× on x86, skipped on ARM |

---

## More detail

[NOTES.md](NOTES.md) has the longer write-ups: how the data is prepared, full evaluation numbers, deployment notes, and the experiments behind the settings above. This includes several experiments that changed the design, and a few that showed a change wasn't worth making.

---

## License and data

The **code** is MIT licensed — see [LICENSE](LICENSE).

The **data is not mine to relicense**, and two other sets of terms come with this repository:

| Path | Origin | Terms |
|---|---|---|
| `data/` | The IFDB database dump published by the [IF Archive](https://ifarchive.org/indexes/if-archive/info/ifdb/) | [CC BY 3.0 US](https://creativecommons.org/licenses/by/3.0/us/) |
| `models/` | Fine-tuned from `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L6-v2` on that data | Apache-2.0 upstream |

Game titles, authors, tags, ratings and reviews all come from [IFDB](https://ifdb.org), whose content is licensed CC BY 3.0 US. That license permits redistribution and derivative works — which is what the Parquet extracts, the trained encoders and the precomputed rankings are — provided IFDB is credited. The app credits it in the footer; anything built on top of this should keep doing so.

CC BY carries no share-alike requirement, which is why the code can be MIT while the data stays under its own terms.
