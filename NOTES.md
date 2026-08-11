# interactive fiction recs — technical notes

Supplementary details for [interactive fiction recs](README.md). The README covers what you need to run the thing, while this covers how it works and why it is built the way it is.

Most of the design decisions below were settled by measurement rather than judgement, so each one records what was measured and what it showed, including the cases where the answer was "leave it alone".

---

## Contents

- [The data](#the-data)
- [Pipeline](#pipeline)
- [How queries are built](#how-queries-are-built)
- [Filtering and display](#filtering-and-display)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [Deployment](#deployment)
- [Possible extensions](#possible-extensions)

---

## The data

From the [IFArchive](https://ifarchive.org/indexes/if-archive/info/ifdb/) dump of the IFDB database:

| Table | Rows | Role |
|---|---|---|
| `games` | 15,544 | Title, author, system, genre, tags, description |
| `reviews` | 80,244 | Explicit 1–5★ ratings — the supervision signal |
| `users` | 20,024 | Accounts, for profiles and filtering |
| `playedgames` | 71,033 | Implicit engagement, used to suppress already-played games |

### Extraction

`scripts/01_extract.py` starts a disposable MariaDB container (`mariadb:10.5.26`, matching the dump's server version), streams the gzipped dump straight into it without expanding to disk, extracts the four tables to Parquet, and removes the container. Nothing persists between runs, which is also what makes it safe to run in CI.

Re-extracting the same dump produces byte-identical Parquet files. Two things buy that: every table is read with an `ORDER BY` on its primary key, and the pandas/pyarrow version stamp is stripped from the schema metadata. `data/manifest.json` records the dump's SHA-256 alongside each table's row count and checksum, with no timestamp — so two manifests compare equal exactly when the data does.

If you would rather read from a MySQL you already run, `--source mysql` skips the container entirely.

### Preparation

Each game becomes a document like `Title: … Author: … Systems: … Tags: … Description: …`, and each user a profile like `Systems: twine, inform. Tags: parser, fantasy, …` built from the games they rated highly.

Interactions are labelled *relative to each game's quality* rather than on an absolute scale. A 3★ review of a 2.7★ game is a positive; the same rating on a 3.3★ game is a negative. Ratings within ±0.25 of the game's smoothed average are discarded as ambiguous. This keeps the signal consistent across games with very different rating distributions.

Author profiles are built the same way, aggregated over an author's own catalogue. Single-game authors are included: their profile comes out byte-identical to that game's query text in 4,711 of 4,713 cases, so the mode duplicates `game` search for them — which is the point, since nobody picking an author knows or cares how many games they wrote.

### Original vs. normalised columns

Values are normalised before they reach the encoders — lowercased, deduplicated, competition tags dropped, version numbers stripped. But anything shown to a reader has to match what ifdb.org shows, or results look wrong.

So originals keep their own names and every normalised variant sits beside them with a `_clean` suffix. **Display reads the plain names; models and retrieval read the `_clean` ones.**

| Column | Displayed | Model-facing |
|---|---|---|
| author | `author` | `author_clean` — split on `,` `/` `and`, deduplicated |
| system | `system` | `system_clean` — parentheticals and versions stripped |
| tags | `tags` | `tags_clean` — genre folded in, competition tags dropped, capped at 20 |
| genre | `genre` | folded into `tags_clean` |
| published | `published` | `year`, for range filters |

A game IFDB lists as `Inform 7` normalises to `inform`; showing that back would look like a bug.

---

## Pipeline

```
query profile ──► [ query encoder ]──► 384d ─┐
                                             ├─► FAISS, everything above threshold
game document ──► [ doc encoder   ]──► 384d ─┘
                                                    │
                          ──► [ cross-encoder reranker ]──► scored pool (cached)
                                                    │
                                        reordered by relevance
                                        (score selected the pool)
                                                    │
                                              hard filters
                                                    │
                                        diversity: ≤2 per author,
                                        cover the query's top systems
                                                    │
                                              page of results
```

**Two-tower bi-encoder.** Query and document towers start from the same `all-MiniLM-L6-v2` checkpoint but are fine-tuned with separate weights, so each specialises for its own input shape — short tag-heavy profiles on one side, longer prose documents on the other. Trained with InfoNCE and in-batch negatives at batch size 64, so each positive competes against 63 negatives, many of them genuinely close.

**Cross-encoder reranker.** `ms-marco-MiniLM-L6-v2`, fine-tuned on the same labelled interactions. Sigmoid over the raw logit gives a 0–1 relevance score.

**The blend.**

```
final_score = (1 - rating_weight) * relevance + rating_weight * (bayesian_avg / 5)
```

Relevance answers "does this match?" and knows nothing about quality; the rating term supplies what the cross-encoder cannot see.

**The blend selects; relevance orders.** `final_score` decides which candidates survive — it is what `min_rerank_score` filters on and what every top-N truncation is applied to, including the precomputed tables. It is not shown, and results are not ordered by it. The displayed ranking is relevance alone, with the raw community rating beside it.

Two reasons. A single number combining match and popularity could not be read as either, so nobody could tell why a game placed where it did. And ordering by it worked against the point of the tool: measured on `twine + horror, romance`, ordering by score gives a top-25 averaging 4.39 stars, while ordering by relevance gives 3.67 — the same candidates, but the well-loved ones no longer float to the top of a list whose job is to surface things you have not played.

`order_by_relevance` in `src/pipeline/ranker.py` does the re-keying, and both front-ends call it, so the CLI and the web app cannot show different orders. It swaps the value carried in each pair rather than sorting afterwards, so `diversify_results` reads relevance too — the author cap keeps an author's most relevant game rather than their best-rated one.

Two properties are deliberate:

*The terms stay on their own absolute scales.* Rescaling within each query's candidate pool would push the top result toward 1.0 even for a niche query with nothing good in it. Absolute scales let a weak pool look weak.

*The rating term is deliberately the weaker one.* Bayesian smoothing squeezes ratings into roughly 0.54–0.81 while relevance spans nearly 0–1, so rating moves the score about 3× less than its weight implies. That imbalance is load-bearing — see [the blend experiment](#the-rating-blend).

**Deep reranking, capped at the tail.** The reranker scores every candidate above the retrieval threshold up to `rerank_pool_cap` (1,000), applied after the tag pre-filter. Cosine rank and cross-encoder rank correlate only weakly (Spearman ρ ≈ 0.22), so truncating aggressively discards most of what the reranker would have picked — but a median vibe pool is 580 candidates, so the cap binds only on the long tail and leaves the typical query scored end to end. See [the cap experiment](#where-the-cap-belongs-and-what-it-costs) for what it costs, which is nothing measurable.

The bi-encoder is doing the pruning either way: a threshold rather than a top-K, but it still takes 10,087 games down to a median of 927 before the cross-encoder sees anything.

---

## How queries are built

Every mode resolves to the same profile format the encoders were trained on:

```python
from src.data.preprocessor import format_profile_text, profile_vocabulary

systems, tags = profile_vocabulary(game_docs)      # options, by frequency
query = format_profile_text(["twine"], ["fantasy", "horror"])
# "Systems: twine. Tags: fantasy, horror"
```

Both helpers read the `_clean` columns, so the values a UI offers are exactly the ones the encoders saw. `format_profile_text` is the same function that builds user and game profiles during preprocessing, so all query sources are identical by construction.

**Free text is not supported.** Both towers train only on `(profile, document)` pairs, and prose lands measurably outside that distribution — best cosine 0.447 against 0.670 for profile format. A UI should offer pickers, not a text box.

---

## Filtering and display

Filters narrow a ranking you are already looking at. They run after scoring, so they never change which candidates were ranked — the scores you see are identical filtered or not.

| Key | Matches |
|---|---|
| `year` | publication year, inclusive |
| `author`, `system`, `genre` | any listed term (OR within the field) |
| `tags` | every listed tag must match (AND) |
| `rating` | raw community average; unrated games excluded |
| `count` | number of ratings |

Two decisions worth knowing:

**Filters match the original IFDB values, not the `_clean` ones**, so a filter only ever matches something visible in the results. `tags:IFComp 2025` works, even though competition tags are stripped from `tags_clean`. And `tags:slice of life` no longer returns games that merely have *genre* "Slice of life" — that used to look like false positives, since genre isn't shown in the tags column.

**`rating` compares the raw average, not the smoothed one.** Filtering on `bayesian_avg` would return games whose actual average is below what you asked for, and would admit unrated games on the strength of the 3.5 prior alone. The trade-off is that one 5★ review reads as 5.0, so pair `rating:` with `count:` when you want a floor backed by a real sample.

### Display normalisation

IFDB's free-text fields are inconsistent — the same tag in many casings, systems and genres separated by either commas or slashes, values repeated within one field. Before display, each of `tags`, `system` and `genre` is split, deduplicated, mapped to the community's dominant casing, and ordered by how many games use each value.

```
Educational, Slice of life  ->  Slice of life, Educational     (640 games vs 135)
Fantasy, mystery, romance   ->  Fantasy, Mystery, Romance
Drama / Political           ->  Drama, Political
dendry                      ->  Dendry
```

Tags split on commas only — `gay/queer protagonist` is one tag — while systems and genres split on both. This is presentation only; filters still match the stored values, so nothing is missed because its casing differs.

---

## Evaluation

Recall@K, NDCG@K and MRR against held-out test-split positives, 1,887 users.

```bash
uv run scripts/06_run_recommender.py --mode evaluate            # raw retrieval  (~25 s)
uv run scripts/06_run_recommender.py --mode evaluate --rerank   # full pipeline  (~51 min)
```

| Metric | Raw retrieval | + Reranking |
|---|---|---|
| MRR | 0.2574 | **0.2847** |
| Recall@5 | 0.3119 | **0.3687** |
| Recall@10 | 0.3606 | **0.4315** |
| Recall@20 | 0.4099 | **0.4777** |
| Recall@50 | 0.4816 | **0.5545** |
| NDCG@5 | 0.2582 | **0.2889** |
| NDCG@10 | 0.2743 | **0.3097** |
| NDCG@50 | 0.3028 | **0.3392** |

Reranking is worth +20% Recall@10 and +13% NDCG@10. `Recall@1` and `NDCG@1` barely move, which fits: the reranker reorders the body of the list rather than changing which single game lands on top.

### Training runs

**Bi-encoder** — 3 epochs, batch 64, ~32 min on an Apple M-series GPU.

| Epoch | Loss | Val Recall@10 | Val MRR | |
|---|---|---|---|---|
| 1 | 3.3233 | 0.1804 | 0.0857 | |
| 2 | 3.0308 | **0.2084** | **0.0923** | ← saved |
| 3 | 2.9514 | 0.1971 | 0.0901 | |

Validation peaked at epoch 2 while training loss kept falling, so the script restores the best-validating weights rather than the last — worth +0.0113 Recall@10 here. The winning epoch moves between runs, which is the argument for selecting a checkpoint rather than tuning the epoch count.

**Reranker** — 2 epochs, batch 16, ~24 min. 41,261 examples (17,697 positive, 23,564 negative), final training loss 0.6272.

---

## Experiments

### Depth: how many candidates should the reranker see?

Cosine rank and cross-encoder rank barely agree (Spearman ρ ≈ 0.22), so reranking only the top 50 by cosine captured **1 of the 25** results a full rerank returns.

But depth turns out not to be a quality lever. Over 200 held-out users with paired bootstrap CIs, 50 → 200 candidates moves NDCG@25 by +0.002 (interval spans zero) and 200 → 800 is flat to slightly negative. What depth buys is *consistency*: with the whole pool scored, a filter narrows a fixed ranking instead of changing which candidates were scored at all.

### Where the cap belongs, and what it costs

Depth was revisited when `vibe` turned out to be the only mode a visitor waits on, and revisiting it settled where `rerank_pool_cap` should sit.

Vibe queries have no ground truth of their own, so 300 held-out users' profiles were truncated to menu-style picks — the shape the UI produces — and scored against their test-split positives at every cap from 50 to 1,500. Because a cross-encoder score depends only on its own (query, document) pair, every cap is derived from one full scoring per query rather than rescored; `scripts/verify_cap_equivalence.py` checks that derivation against genuinely capped runs, ID for ID.

Quality confirms the earlier finding and extends it — nothing above 300 is distinguishable from scoring everything:

| Cap | Recall@10 | NDCG@10 | ΔNDCG@10 (95% CI) |
|---|---|---|---|
| 100 | 0.1779 | 0.1149 | −0.0165 [−0.0375, +0.0031] |
| 200 | 0.1799 | 0.1158 | −0.0155 [−0.0339, +0.0007] |
| 300 | 0.1918 | 0.1269 | −0.0045 [−0.0149, +0.0040] |
| 500 | 0.1979 | 0.1317 | +0.0003 [−0.0084, +0.0072] |
| 1,000 | 0.2005 | 0.1314 | +0.0000 [+0.0000, +0.0000] |
| whole pool | 0.2005 | 0.1314 | — |

Consistency is where the cap is actually paid for, and it degrades far faster than quality does:

| Cap | overlap@25 vs whole pool | top-1 same | identical pages |
|---|---|---|---|
| 200 | 0.558 | 56% | 13% |
| 300 | 0.697 | 72% | 25% |
| 500 | 0.854 | 84% | 49% |
| 750 | 0.957 | 94% | 72% |
| 1,000 | 0.992 | 99% | 95% |

**1,000 is a tail control, not a throughput control.** The pool is smaller than it looks — a median of 927 candidates clear the cosine floor and 580 survive the tag pre-filter — so capping at 1,000 leaves the median query untouched and cuts only 2% of mean scoring work. What it does is bound the worst case: p99 pool 1,387 → 1,000, which on the deployment VM is 32 s → 24 s before quantization. It costs nothing measurable on either axis, which is the whole argument for it.

Going lower does buy real time (500 puts every query under 12.5 s), but 0.854 overlap means about 15% of the page moves. Since depth was never a quality lever, that cost is entirely in reproducibility — worth knowing before trading it for latency.

The cap is applied **after** the tag pre-filter in both front-ends. At the same K that dominates capping the raw cosine list: it spends the budget on candidates that survived the filter, and measured better at every cap (0.854 vs 0.750 overlap@25 at 500).

### int8 quantization of the cross-encoder

Dynamic quantization is the only lever here that shortens the *median* query rather than the tail: it scales all scoring by a constant, so unlike a cap it does not work by discarding candidates. Weights are stored as int8 and activations quantized per batch, with no calibration set and no retraining.

Quality is unaffected. Over 150 queries and 147,619 scored pairs, with both models scoring the same pools so the comparison is paired:

| Metric | fp32 | int8 | Δ (95% CI) |
|---|---|---|---|
| Recall@10 | 0.2025 | 0.2083 | +0.0058 [−0.0025, +0.0200] |
| Recall@25 | 0.2654 | 0.2570 | −0.0083 [−0.0267, +0.0033] |
| NDCG@10 | 0.1215 | 0.1224 | +0.0010 [−0.0062, +0.0078] |
| NDCG@25 | 0.1381 | 0.1360 | −0.0020 [−0.0093, +0.0042] |
| MRR | 0.1054 | 0.1044 | −0.0010 [−0.0098, +0.0060] |

Every interval spans zero and the point estimates fall on both sides of it — the signature of a change that perturbs scores without systematically degrading the ranking. Relevance scores move 0.0245 on average (max 0.242), enough to reshuffle near-ties and not enough to reorder anything that matters. The visible cost is page churn: **0.936 overlap@25**, top-1 unchanged for 92% of queries, mean displacement 1.84 positions. That is roughly what capping at 750 costs, except that a cap discards candidates while this only jitters the order of ones it kept.

**Speed is a property of the host, not the model**, by a factor large enough to invert the decision:

| Backend | Machine | pairs/s | |
|---|---|---|---|
| fbgemm | deployment VM (x86, 2 vCPU) | 45 → 92 | **2.02×** |
| qnnpack | Apple M-series laptop | 149 → 36 | **0.24×** |

Hence `model.quantize_reranker: "auto"`, which consults the backend rather than trusting a boolean: it enables quantization on fbgemm and skips it on qnnpack, so one config.yaml is correct on an E-series instance and on an Ampere A1. Quantized kernels are also CPU-only, so it skips (loudly) when the model has landed on MPS or CUDA.

One trap is worth recording, because it fails silently. `CrossEncoder.model` is a property proxying to `ce[0].auto_model`, and `nn.Module.__setattr__` intercepts Module assignments before the property setter runs — so assigning to either registers an unused second child and leaves the module `forward()` calls in fp32. Inference keeps working and returns *bit-identical* scores, which reads as "quantization changed nothing" rather than as a bug. The correct target is `ce[0].model`, and `src/pipeline/quantize.py` asserts the live module converted so it cannot recur.

Two caveats on adopting it. The quality numbers above were measured under qnnpack; fbgemm applies `reduce_range` on x86 and its error profile differs slightly, so re-running `scripts/exp_quantized_quality.py` on the target host gives the figure that actually applies. And `torch.ao.quantization` is deprecated as of torch 2.11 with removal signalled — the migration path is torchao.

int8 also pushes about 3% more candidates below the `min_rerank_score` floor, trimming filter headroom. Harmless against a median stored depth of ~510, but it compounds with a cap.

### The rating blend

Two variants tested over 300 users:

| Scheme | Recall@10 | NDCG@10 |
|---|---|---|
| raw, weight 0.3 | **0.4454** | **0.3238** |
| raw, weight 0.5 | 0.4438 | 0.3223 |
| raw, weight 0.7 | 0.4263 | 0.2945 |
| pool-rescaled, weight 0.5 | 0.4390 | 0.3076 |

Rescaling the two terms so they contribute equally is **worse** at every weight (−0.015 NDCG@10, CI [−0.024, −0.006]). Every change that strengthens the rating signal hurts. The 3:1 relevance dominance you get "accidentally" from dividing by 5 is close to optimal — rating is the noisier signal and deserves less influence than its weight implies. Weights 0.3 and 0.5 are statistically indistinguishable; above 0.5 degrades.

### The tag pre-filter

Dropping candidates that share no tag with the query removes 11% of the pool for full profiles and 39% for short menu-style queries, and changes Recall@10/25 and NDCG@10/25 by **exactly zero** across 300 users. The highest-ranked candidate it removes sits around rank 100–400, well below anything displayed.

Unlike truncating by cosine rank, it prunes on something the query is actually made of — which is why it costs nothing. It also means every stored result shares a visible tag with the query, so a recommendation always has a reason you can check.

### Relevance floor vs. stored depth

Filters exhaust a short list fast — `tags:horror` used to leave 78% of users with fewer than 25 results. The instinct is to store more per key, but `top_n` was the wrong knob: for most keys the list was short because of the relevance floor, not truncation.

| `min_rerank_score` | Median list |
|---|---|
| 0.30 | 96 |
| 0.20 | 130 |
| 0.10 | 210 |
| 0.00 | 262 |

Lowering the floor to 0.10 with a tag-overlap rule roughly doubles usable depth at **identical** Recall/NDCG, and lifts the worst decile from 8 candidates to 37 — the floor was cutting hardest exactly where headroom was scarcest.

### Lighter cross-encoders

Three smaller rerankers were fine-tuned and evaluated on identical candidate pools:

| Model | Params | pairs/s | Recall@10 | vs L6 (95% CI) |
|---|---|---|---|---|
| **MiniLM-L6** (current) | 22.7M | 140 | **0.4447** | — |
| MiniLM-L4 | 19.2M | 221 | 0.4267 | −0.018 [−0.040, +0.003] n.s. |
| MiniLM-L2 | 15.6M | 416 | 0.3952 | −0.050 [−0.080, −0.018] |
| TinyBERT-L2 | 4.4M | 1,374 | 0.4127 | −0.032 [−0.061, −0.003] |

L2 and TinyBERT degrade significantly despite being 3× and 10× faster. L4's interval spans zero, but that's insufficient power at n=300 rather than proven equivalence, and both point estimates lean negative for only 1.58×. Not worth the trade.

### Does tag order inside a query matter?

Transformers are not permutation-invariant, so reordering tags provably changes scores. It does not change quality: across six orderings over 150 users — as-is, reversed, shuffled, alphabetical, corpus-frequency, rarest-first — **every confidence interval spans zero**. Even fully shuffling costs only −0.020 Recall@10, CI [−0.053, +0.013].

A null result rather than proof of invariance: at n=150 an effect smaller than ~4% would be invisible. But it rules out anything large enough to justify a retrain, so display-side reordering is safe and the stored query text is left alone.

### Is the relevance score meaningful?

Over 150,432 (user, candidate) pairs, base rate 0.22%:

| Relevance | Genuine held-out positives | Lift |
|---|---|---|
| 0.0–0.1 | 0.026% | 0.1× |
| 0.5–0.6 | 0.175% | 0.8× |
| 0.7–0.8 | 0.602% | 2.7× |
| 0.8–1.0 | 2.489% | **11.3×** |

Three conclusions in decreasing order of confidence. *Within a query the score is strongly discriminative* — monotonic across every bucket. *It is not a probability* — 0.85 means about 2.5% likely, not 85%, because the base rate is 0.22%. *Across queries it transfers only loosely* — in the 0.5–0.6 band, candidates belonging to low-ceiling users are hit 6× more often than high-ceiling ones. What does carry across is the ceiling itself: top-quartile users get roughly twice the Precision@10 of the bottom quartile. So a low-scoring result set genuinely signals a query with few good matches, which is why the blend keeps absolute scales.

---

## Deployment

Sized for a CPU-only host such as a free-tier Hugging Face Space (2 vCPU, 16 GB).

### Precomputed rankings

`game`, `author` and `reviewer` draw from fixed key sets, so their rankings are computed once and served as a lookup:

```bash
uv run scripts/07_precompute.py --mode all --top-n 500     # ~5 h
```

| Artefact | Rows | Keys | Median depth | Size |
|---|---|---|---|---|
| `precomputed_userid.parquet` | 816,351 | 3,170 | 222 | 14.4 MB |
| `precomputed_gameid.parquet` | 2,269,909 | 10,076 | 198 | 39.2 MB |
| `precomputed_authorid.parquet` | 1,436,646 | 6,291 | 202 | 24.0 MB |

Rows stream to Parquet in batches and publish by atomic rename, so a reader never sees a half-written file and a failed run cannot destroy the previous artefact. If a file is missing or unreadable, that mode falls back to live scoring — slower, never wrong.

**Re-run this after retraining anything**, or the app serves stale rankings.

Precompute ignores `rerank_pool_cap` — an offline job has no latency budget to protect, so these tables are built over the whole pool. It does pick up `quantize_reranker` through the shared loader, which on an x86 host halves the run. The tables currently shipped were built fp32; regenerating them under int8 is optional rather than required, since quality is unchanged either way.

Not everything is deep: 3.2% of users, 8.7% of games and 9.1% of authors have fewer than 25 stored results, so a UI should report the true count rather than implying a full page.

### Resources

**Memory ~1.5 GB** with everything resident — dataframes, FAISS index, embeddings, both models and all three lookup tables.

**CPU is the real constraint**, and it is worth measuring rather than estimating: an Apple M-series laptop runs the cross-encoder at ~145 pairs/s on two threads, while the 2-vCPU OCI instance runs it at 45 — a 3.2× gap that no amount of reasoning about "a free-tier vCPU" would have pinned down. `scripts/measure_latency.py` reports it for a given host. That budget is fixed and shared, so concurrent requests divide it. Serialising inference (`concurrency_limit=1`) makes contention a visible queue rather than everyone slowing down at once.

Latency is close to linear in candidates scored, which makes it projectable. On the deployment VM:

```
fp32   t = 1.45 s + pairs / 45.2
int8   t = 0.82 s + pairs / 91.5
```

Both fit within 0.5 s across pools from 174 to 1,390 pairs. Note the fixed term: ~1 s of every query is setup that no amount of pool trimming touches, which is why capping below ~300 candidates stops paying for itself.

With `rerank_pool_cap: 1000` and int8 on fbgemm, a cold vibe query runs **6–7 s typical and 11.7 s worst case**, against 12–14 s / 38 s uncapped and unquantized. The cap does the tail and the quantization does the median; neither substitutes for the other.

Only `vibe` consumes CPU, and it is cached two ways: per session while a user narrows filters, and process-wide across users keyed by (systems, tags). A repeat vibe query costs 0.02 s against several seconds cold. The cache holds 2,048 entries at ~85 KB each, about 171 MB.

The number worth watching in production is neither the cache nor the models: PyTorch's allocator grew ~470 MB across 28 scorings in testing and had not clearly plateaued, and it does so whether or not results are cached. A larger cache *reduces* that pressure, since every hit is a scoring that never runs.

---

## Possible extensions

- **Larger base models** — `all-mpnet-base-v2` for retrieval, DeBERTa-v3 for reranking. Better quality, worse latency, which matters for a lightweight deployment but could be useful in situations where everything would be precomputed.
- **Session-based profiles** — a sliding window over recent positive ratings rather than a full-history aggregate.
- **Faster bi-encoder training** — wall time is dominated by the training epochs (6–16 min each), not the validation passes (15–50 s). Gains would come from larger batches or mixed precision.
- **Author identity** — `gameprofilelinks` maps games to IFDB user accounts, covering 38.5% of games. Too sparse to key author search on, but enough to deep-link authors to their profiles and to merge pen names that name matching cannot.
