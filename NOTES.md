# IF recommender — technical notes

Supplementary details for [IF recommender](README.md). The README covers what you need to run the thing, while this covers how it works and why it is built the way it is.

Most of the design decisions below were settled by measurement rather than judgement, so each one records what was measured and what it showed, including the cases where the answer was "leave it alone".

---

## Contents

- [The data](#the-data)
- [Pipeline](#pipeline)
- [How queries are built](#how-queries-are-built)
- [Filtering and display](#filtering-and-display)
- [Evaluation](#evaluation)
- [Item-to-item modes](#item-to-item-modes)
- [Experiments](#experiments)
- [Deployment](#deployment)
- [Possible extensions](#possible-extensions)

---

## The data

From the [IFArchive](https://ifarchive.org/indexes/if-archive/info/ifdb/) dump of the IFDB database (`ifdb-archive-20260901`):

| Table | Rows | Role |
|---|---|---|
| `games` | 15,747 | Title, author, system, genre, tags, description, year, language |
| `reviews` | 81,200 | Explicit 1–5★ ratings — the supervision signal |
| `users` | 20,377 | Accounts, for profiles and filtering |
| `playedgames` | 71,928 | Implicit engagement, used to suppress already-played games |
| `wishlists` | 51,269 | A user's wishlist — item co-occurrence |
| `polls`, `pollvotes` | 704 / 22,827 | Games voted into one poll ("Best short games") — item co-occurrence |
| `reclists`, `reclistitems` | 578 / 6,970 | A member's recommended list — item co-occurrence |
| `crossrecs`, `gamexrefs`, `gamexreftypes` | 399 / 964 / 9 | Explicit "if you liked X" links and sequel/remake relations; too few to train on, kept for sanity checks |

The last four groups feed the item-to-item encoder (see [Item-to-item modes](#item-to-item-modes)). They were chosen after an audit of every table in the dump; what was looked at and left out is recorded under [What the dump does and does not offer](#what-the-dump-does-and-does-not-offer).

### Extraction

`scripts/01_extract.py` starts a disposable MariaDB container (`mariadb:10.5.26`, matching the dump's server version), streams the gzipped dump straight into it without expanding to disk, extracts the twelve tables to Parquet, and removes the container. Every table's primary key is `id` in IFDB, so each is renamed on the way out (`gameid`, `userid`, `reviewid`, `listid`, `crossrecid`) rather than guessed at. Nothing persists between runs, which is also what makes it safe to run in CI.

Re-extracting the same dump produces byte-identical Parquet files. Two things buy that: every table is read with an `ORDER BY` on its primary key, and the pandas/pyarrow version stamp is stripped from the schema metadata. `data/manifest.json` records the dump's SHA-256 alongside each table's row count and checksum, with no timestamp — so two manifests compare equal exactly when the data does.

If you would rather read from a MySQL you already run, `--source mysql` skips the container entirely.

### Preparation

Each game becomes a document:

```
Title: … Author: … Systems: … Tags: … Language: english. Description: …
```

and each user a profile built from the games they rated highly:

```
Systems: inform, twine. Tags: parser, fantasy, … Authors: Emily Short, Andrew Plotkin.
Language: spanish. Dislikes: puzzleless, recommended for beginners
```

Only the sections with values are written, so a vibe pick of systems and tags comes out exactly as it did before the later sections existed. Language sits ahead of the description in the document, which is the part the 256-token limit truncates.

The later profile sections came with the 2026-09 dump, and each was chosen from the data rather than on instinct:

- **Authors** — the five authors liked most often. Documents carry `Author:`, so this gives the cross-encoder an exact-match path. It is a real signal — 24.5% of test positives are by an author the user had already liked — and the strongest single section by a wide margin (see [Profile sections](#which-profile-sections-matter)). It also concentrates: 51% of a raw top-10 is by an already-liked author, which the page's ≤2-per-author cap keeps in proportion.
- **Language** — languages meeting the same bar, written only when something other than English qualifies: on its own, "english" is a constant token that says nothing. 13% of games are non-English.
- **Dislikes** — systems and tags that mark the user's *disliked* games out from their liked ones. See below.

Two things were tried and taken out again. **`Era`** — the decades holding a quarter of the liked set — and **`Year:`** in the document. Era measured neutral for retrieval (see [Profile sections](#which-profile-sections-matter)), and the pair of them turned period into a cutoff: a decade token cannot say that 2019 is closer to 2020 than to 2010, and the item encoder, given `Year:` on both sides of every pair, learned to match the year and little else (see [Item-to-item modes](#item-to-item-modes)). The year filter is where a period preference belongs — a filter is allowed to be a cutoff. Whatever temporal closeness the recommendations still show comes from the co-occurrence data itself, softly.

Interactions are labelled *relative to each game's quality* rather than on an absolute scale. A 3★ review of a 2.7★ game is a positive; the same rating on a 3.3★ game is a negative. Ratings within ±0.25 of the game's smoothed average are discarded as ambiguous. This keeps the signal consistent across games with very different rating distributions.

**Dislikes are discriminative, not a mirror of "top tags".** A disliked game carries the same ordinary tags a liked one does — "parser", "fantasy" — so the most common tags across a user's negatives would just restate the most common tags across everything. A value qualifies instead when at least two disliked games carry it *and* it is commoner among the disliked games than among the liked ones, ranked by that gap and capped at ten; anything already in the liked list is excluded, so a profile cannot both like and dislike a tag. 39% of training profiles get a list; the rest have too little or too undifferentiated a negative history, and get nothing rather than noise.

**Negatives are held out too.** Users with three or more negatives keep ~10% of them (at least one) in the test split with `label=0`, so the evaluation can check that disliked games rank *low* — 3,033 held-out negatives against 3,403 held-out positives. Without them the value of a dislike section is invisible to the metrics.

Author profiles are built the same way, aggregated over an author's own catalogue. They are now display text only: `author` search runs in the item space (see [Item-to-item modes](#item-to-item-modes)).

### Original vs. normalised columns

Values are normalised before they reach the encoders — lowercased, deduplicated, competition tags dropped, version numbers stripped. But anything shown to a reader has to match what ifdb.org shows, or results look wrong.

So originals keep their own names and every normalised variant sits beside them with a `_clean` suffix. **Display reads the plain names; models and retrieval read the `_clean` ones.**

| Column | Displayed | Model-facing |
|---|---|---|
| author | `author` | `author_clean` — split on `,` `/` `and`, deduplicated |
| system | `system` | `system_clean` — parentheticals and versions stripped |
| tags | `tags` | `tags_clean` — genre folded in, competition tags dropped, capped at 20 |
| genre | `genre` | folded into `tags_clean` |
| published | `published` | `year`, for range filters and the document |
| language | `language` | `language_clean` — codes and variants resolved to lowercase names |

A game IFDB lists as `Inform 7` normalises to `inform`; showing that back would look like a bug.

**Tags are ordered by corpus frequency before the 20-tag cap.** IFDB stores tags in the order they were voted, and one game carries fifty; cutting at twenty used to keep whichever came first. Frequency order keeps the tags a profile can actually contain — a tag on one game in the corpus matches nothing else — and puts genre values, which were set deliberately, in front. Dropped this way, the 2,411 games with any tag voted by more than one user lose nothing that mattered: nearly every vote count is one, which is also why tag weights were not worth adding.

---

## Pipeline

Two routes into the same tail. `reviewer` and `vibe` are profile queries; `game` and `author` are item-space lookups.

```
reviewer / vibe                                game / author
query profile ──► [ query encoder ]──► 384d ─┐    seed game(s) ──► [ item encoder ] ──► 384d
                                             ├─► FAISS, above threshold        │
game document ──► [ doc encoder   ]──► 384d ─┘        │                  nearest neighbours
                                                      │                  (centroid of seeds)
                            ──► [ cross-encoder reranker ]                     │
                                                      │                        │
                                                      └──────► scored pool ◄───┘
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

**Item encoder.** A third tower, one input shape — a game document — trained so that games which belong together in someone's mind embed close: `game` and `author` searches are nearest-neighbour lookups in its space, with no reranker. See [Item-to-item modes](#item-to-item-modes) for why, and for what it is trained on.

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

**Deep reranking, capped at the tail.** The reranker scores every candidate above the retrieval threshold up to `rerank_pool_cap` (500 since the move to ARM; 1,000 on x86), applied after the tag pre-filter. Cosine rank and cross-encoder rank correlate only weakly (Spearman ρ ≈ 0.22), so truncating aggressively discards most of what the reranker would have picked — but a median vibe pool is 580 candidates, so the cap binds only on the long tail and leaves the typical query scored end to end. See [the cap experiment](#where-the-cap-belongs-and-what-it-costs) for what it costs, which is nothing measurable.

The bi-encoder is doing the pruning either way: a threshold rather than a top-K, but it still takes 10,291 games down to a median of about 1,000 before the cross-encoder sees anything.

**The threshold belongs to the model, not the design.** With the 2026-09 profiles the query tower embeds more specifically and the whole cosine distribution sits about 0.1 lower — the best match for a median user is 0.64 where it was 0.67, and the body of the pool moved further. At the old floor of 0.25 the median pool was 136 candidates and a fifth of users had fewer than 25 stored results; at 0.15 it is 1,010, which is what the cap, the tag policy and the headroom figures below were all measured against. The experiment tables in this file that quote 0.25 as "shipped" predate the change and are left as measured.

---

## How queries are built

Every mode resolves to the same profile format the encoders were trained on:

```python
from src.data.preprocessor import format_profile_text, profile_vocabulary

systems, tags = profile_vocabulary(game_docs)      # options, by frequency
query = format_profile_text(["twine"], ["fantasy", "horror"])
# "Systems: twine. Tags: fantasy, horror"
```

Both helpers read the `_clean` columns, so the values a UI offers are exactly the ones the encoders saw. `format_profile_text` is the same function that builds user profiles during preprocessing, so all query sources are identical by construction. It takes the later sections (`authors`, `languages`, `dislikes`) as optional arguments, and the web app's vibe mode offers a picker for one of them: authors, the strongest section there is. Not dislikes, which measured as doing nothing to the ranking (see [Profile sections](#which-profile-sections-matter)) — a control that does nothing would be a lie. And not language, though the profiles carry it: to a reader it is a constraint rather than a taste, the hardest filter there is for anyone who reads one language, and offering it beside the tags asked them to guess whether it narrowed or merely nudged. It is a filter, where a constraint belongs. Picks are put in corpus order, as systems and tags are, so the same picks make the same query.

`parse_profile_sections` is the inverse, and finds sections by their labels rather than by splitting on `". "` — an author name like "jennifer s. lange" has a period of its own. A label counts only at the start of the text or after a sentence break, so a tag such as "era: victorian" inside a list is not mistaken for the `Era` section.

`game` and `author` queries do not go through this at all: they are seed ids into the item space. The profile text is still built for them, to show in the query panel and to pick the diversity targets.

**Free text is not supported.** Both towers train only on `(profile, document)` pairs, and prose lands measurably outside that distribution — best cosine 0.447 against 0.670 for profile format. A UI should offer pickers, not a text box.

---

## Filtering and display

**The author cap works on the whole ranking, not a page.** `diversify_results` sets aside an author's third and later games; they now go to the tail of the list in score order, so a short page still fills from them but they never outrank another author's first. They used to be merged back by score whenever the cap left fewer than `top_k` — and the web app asks for the whole pool and paginates itself, so `top_k` was the pool size, the page was always "short", and the cap had been doing nothing there since local pagination arrived. It surfaced once the `Authors` section made same-author concentration common: four games by one author in a reviewer's top five.

Filters narrow a ranking you are already looking at. They run after scoring, so they never change which candidates were ranked — the scores you see are identical filtered or not.

| Key | Matches |
|---|---|
| `year` | publication year, inclusive |
| `author`, `system`, `language`, `genre_tags` | any listed term (OR within the field) |
| `rating` | raw community average; unrated games excluded |
| `count` | number of ratings |

Every filter offers only what the results in front of you contain: the list
fields are ordered by frequency, the year dropdowns span that set's own range
(continuously, gaps filled), and the two rating ladders are trimmed to the rungs
that change something — a rung above everything observed returns nothing, and a
rung below the lowest returns what the next one up already returns. Counts of
2, 3, 8, 10, 12 leave `[2, 5, 10]` out of `[0, 1, 2, 5, 10, 25, 50]`.

IFDB records "no authoring system" as the literal string `None` (164 games) and
once as `N/A`. Both are blanked at load time, in the original and `_clean`
columns together, before the display maps and vocabularies are derived from
them — otherwise the absence appears as something to search for in the vibe
picker and the system filter, and prints as "None" on a card where every other
missing field shows an em dash. The `N/A` was worse than it looked: the display
map splits on `/`, so it had also been producing systems called "N" and "A". The
games themselves stay; only the field is emptied. `Other` and `Misc` in the
genre and tag lists are left alone — vague, but real values people applied.

Messages — "pick a game first", "no results match those filters" — render in
their own slab below the filters rather than in the profile block. Folding the
no-match message into the profile replaced it, taking away the one thing that
says what there is to relax. Only a press of the recommend button may prompt for
missing input: the same function runs whenever a filter moves, including the
moves it makes itself when a mode changes, and a prompt appearing then answers a
question nobody asked.

The four numeric filters accept values outside their own choices. Gradio rejects
an incoming value the component no longer lists, and these lists are rewritten on
every query, so a control still reporting the previous render's year failed
validation before any of this code ran — an error toast for a query that had
worked. Values are coerced where they are read instead.

Every control resets on a new query and the block reappears collapsed, so no
setting outlives the search it was made against. Switching mode does the same
immediately rather than waiting for the button, and clears the results with it:
the block's choices were built from the previous mode's results and its values
were applied to them, so both describe something the reader has just navigated
away from.

Two traps live in that, both of which cost results silently rather than
visibly. **The rating ladders are derived from the scored pool, not from the
results on screen.** The results on screen have always had the rating filter
applied to them, so rungs read back off them would start at the 3.0 default —
the filter could only ever be tightened, and the tail below it would be
unreachable. The pool is that same set before the two thresholds touched it.
Defaults are snapped into the rungs *before* filtering rather than after, since
a pool whose games all hold 19 ratings offers only the rung below that. **And a
year span counts as "no constraint" only against the span being offered**, not
the corpus one, because applying a range drops the 184 games with no recorded
year.

Genre and tags are one filter, `genres/tags`, matching either field. The
matching has always pooled them — the `_clean` tag column folds genre in, which
is what the encoders were trained on — so presenting two filters implied a
distinction the data does not keep: "Fantasy" is a genre on one game and a tag
on the next. The dropdown merges both fields' values, deduped case-insensitively and ordered
by how many results carry each. Spelling is fixed corpus-wide by deferring to
the tag column, not by picking the commoner of the two forms: "Horror" outnumbers
"horror" (938 games to 806) while "comedy" beats "Comedy" (196 to 73), so
frequency alone sets a capitalised word beside a lowercase one and the list still
looks unedited. Deferring to tags holds one convention — the same one the vibe
picker shows, so a value reads identically in both.
It offers the raw values, so competition tags (IFComp, XYZZY) are filterable
even though the vibe picker's cleaned vocabulary drops them. Cards still show
genre and tags on separate rows, which is where the distinction earns its place.

Every list-valued field is OR within itself and AND against the others, which is
the faceted-search convention. `tags` was the exception until the filter lists
started being built from the result set: with dynamic choices, OR also means a
combination the UI offers can never come back empty. The cost is that requiring
two tags at once is no longer expressible — narrowing now comes from combining
*different* facets rather than stacking one.

Two more decisions worth knowing:

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

Recall@K, NDCG@K and MRR against held-out test-split positives, 2,020 users with a test item (1,920 of them with a profile), plus NegHit@K — the share of a user's held-out *disliked* games that reach their top K, which should be low — over the 908 users who have any.

```bash
uv run scripts/06_run_recommender.py --mode evaluate            # raw retrieval  (~25 s)
uv run scripts/06_run_recommender.py --mode evaluate --rerank   # full pipeline  (~50 min)
uv run scripts/eval_item_modes.py --rerank-sample 300           # game and author modes
```

**Queries are the training profiles, not the serving ones.** The profiles the app serves (`user_profiles_retrieval.parquet`) are built from every rating a user made — including the test positives being looked for — and the evaluation used to query with them. That leaked: the held-out game's own tags sat in the query, and with the `Authors` section its author did too. On the 2026-09 data the leak was worth Recall@10 0.387 against an honest 0.229 for the unchanged pipeline, and 0.537 against 0.363 with the new profiles. Every figure below is leak-free (`--profiles train`, the default now); the earlier write-ups' numbers were not, and are superseded rather than restated.

Baseline is the previous pipeline retrained on the 2026-09 dump, so the only difference between the columns is what went into the documents and profiles.

The baseline retrieves at cosine ≥ 0.25 (its own floor) and the shipped models at 0.15 (see [Pipeline](#pipeline) for why the floor moved); both lists are in the blended order that selects the pool.

| Metric | Baseline, retrieval | Baseline + rerank | Shipped, retrieval | Shipped + rerank |
|---|---|---|---|---|
| MRR | 0.1117 | 0.1656 | 0.1695 | **0.2109** |
| Recall@5 | 0.1740 | 0.2472 | 0.2847 | **0.3114** |
| Recall@10 | 0.2286 | 0.3161 | 0.3663 | **0.4191** |
| Recall@20 | 0.2877 | 0.3870 | 0.4426 | **0.5159** |
| Recall@50 | 0.3698 | 0.4646 | 0.5201 | **0.6197** |
| NDCG@10 | 0.1314 | 0.1902 | 0.2058 | **0.2458** |
| NegHit@10 | — | — | 0.0474 | 0.0987 |
| NegHit@50 | — | — | 0.1301 | 0.1949 |

The new documents and profiles are worth about a third on every headline metric at the same stage — retrieval alone now beats the baseline's full pipeline — and reranking still earns its place on top of them: +14% Recall@10, +19% NDCG@10, +24% MRR. The baseline has no NegHit because its split held out no negatives.

Disliked games do rank low: 5% of them reach a raw top-10 against 37% of liked ones. The reranker separates them less sharply than retrieval does (10% against 42%), which is the one number here that moves the wrong way; see [Profile sections](#which-profile-sections-matter) for what the `Dislikes` section itself contributes.

**Ordering.** Lists are scored as the page shows them — by relevance — as well as by the blended score that selected them:

| Shipped + rerank | blend order | **relevance order (the page)** |
|---|---|---|
| MRR | 0.2109 | 0.2002 |
| Recall@10 | 0.4191 | 0.4375 |
| Recall@50 | 0.6197 | 0.6104 |
| NDCG@10 | 0.2458 | 0.2435 |
| NegHit@10 | 0.0987 | 0.0565 |

Recall@10 is a little better by relevance and MRR a little worse — the rating term is worth something at the very top — and relevance ordering nearly halves how often a disliked game reaches the top ten. The reasons for ordering by relevance are unchanged (see [The blend](#pipeline)).

An earlier round of these models carried an `Era` section and `Year:` in the documents. Removing both (see [Preparation](#preparation)) moved every user-mode number by less than a point in either direction — Recall@10 0.4294 → 0.4375 by relevance, MRR 0.1988 → 0.2002 — while fixing the item modes outright.

### Training runs

**Bi-encoder** — 3 epochs, batch 64, 18,005 pairs, ~20 min per epoch on an Apple M-series GPU. Validation uses training profiles and is leak-free.

| Epoch | Val Recall@10 | Val MRR | |
|---|---|---|---|
| 1 | 0.2957 | 0.1295 | |
| 2 | 0.3006 | 0.1274 | |
| 3 | **0.3091** | 0.1299 | ← saved |

The baseline run on the same data peaked at 0.1963. The best epoch still moves between runs (the previous round peaked at epoch 2), which remains the argument for selecting a checkpoint rather than tuning the epoch count.

**Reranker** — 2 epochs, batch 16, 39,276 examples (18,005 positive, 21,271 negative).

**Item encoder** — 3 epochs, batch 64, 34,289 sampled pairs per epoch, ~16 min per epoch, initialised from the fine-tuned doc encoder. Best epoch 2, val item Recall@10 **0.0759** (the round with `Year:` in the documents peaked at 0.0722). The validation task is the item-to-item one described below, which is far harder than the profile task, so the two Recall columns are not comparable.

---

## Item-to-item modes

`game` and `author` used to be profile queries in disguise: a game's systems and tags were formatted as though they were a user's profile and put through the query encoder, which had only ever seen aggregates over many games. That can find tag-alike games and nothing else — never "people who loved this also loved that" — and it was never evaluated, because nothing in the data said what a game's neighbours should be.

The co-occurrence tables in the dump say exactly that. Four sources, each a group of games that belong together in someone's mind:

| Source | Sets | Rows | Games | Weight |
|---|---|---|---|---|
| A user's liked games | 1,653 | 28,095 | 4,780 | 1.0 |
| A user's wishlist | 1,412 | 45,813 | 5,654 | 0.5 |
| Games voted into one poll | 688 | 9,389 | 2,800 | 1.0 |
| One member's recommended list | 534 | 5,883 | 2,409 | 0.7 |

7,208 of the 10,291 games in the retrieval set appear in at least one; the rest are still embedded from their text, which is the reason for a text tower rather than id embeddings. Every source is keyed by a user — a poll vote and a list have an owner — so the (user, game) pairs held out for validation and testing are dropped from all four before any pair is drawn. Otherwise a user's wishlist could hand the encoder the very pair the evaluation asks it to find.

**Sampling, not enumeration.** A user with 1,500 liked games would supply a million pairs and drown everyone else, so each epoch draws at most `pairs_per_set_cap` (20) random pairs from each set, scaled by the source weight. Symmetric InfoNCE with in-batch negatives, masking the ones that are not negatives — the same game on both sides, or two pairs from one set.

**No reranker.** The cross-encoder was trained on (profile, document) pairs and has nothing to say about two documents; an item-to-item bi-encoder is ordinarily the whole model, and a second cross-encoder would have cost another training run and turned a seconds-long precompute back into hours. It can be added if the numbers ever call for it.

**No `Year:` in the documents.** The first item encoder was trained with it, and its neighbours were the seed's contemporaries and nothing else — the median year gap between a seed and its top 25 was zero, and for a 2020 seed 93% of them were from 2020–21:

| seed year | <2010 | 2010–14 | 2015–17 | 2018–19 | 2020–21 | 2022+ |
|---|---|---|---|---|---|---|
| 2016 | 0% | 2% | **96%** | 2% | 0% | 0% |
| 2019 | 0% | 0% | 1% | **90%** | 8% | 1% |
| 2020 | 0% | 1% | 0% | 5% | **93%** | 0% |
| 2023 | 1% | 0% | 0% | 0% | 0% | **99%** |

The data invites it: reviewers rate whole competition cohorts, so co-liked games share a year, and a year token is the cheapest way to say so. The metrics did not object, because a user's held-out positives share that year too — which is a reminder that the item-mode numbers reward "same cohort" as readily as "same taste", and the qualitative check matters. With the token gone the encoder has to find the taste, and period survives only as far as the co-occurrence carries it — a median gap of three to four years, spread smoothly, with 2019 and 2020 seeds drawing near-identical distributions:

| seed year | <2010 | 2010–14 | 2015–17 | 2018–19 | 2020–21 | 2022+ | median gap |
|---|---|---|---|---|---|---|---|
| 2016 | 7% | 13% | 31% | 13% | 12% | 25% | 3 |
| 2019 | 8% | 15% | 17% | 16% | 16% | 28% | 3 |
| 2020 | 13% | 12% | 16% | 14% | 18% | 28% | 4 |
| 2023 | 5% | 5% | 8% | 6% | 9% | 67% | 2 |

Photopia still gets *9:05*, *Shade*, *I-0*, *Aisle*, *Galatea* and *Varicella* — its actual contemporaries in taste — while Counterfeit Monkey's neighbours now run from *Savoir-Faire* (2002) to *Never Gives Up Her Dead* (2023). Validation item Recall@10 went *up*, 0.072 to 0.076, on a metric that rewards same-cohort matches.

**Relevance is rescaled cosine.** The space is narrow — random pairs sit at cosine 0.75 and a game's 500th neighbour well above 0.8 — so raw values would read as "everything matches" and a cosine floor would never bite. Relevance is `(cos − baseline) / (1 − baseline)`, clipped to [0, 1], with the baseline — 0.751 for the shipped encoder — measured over random pairs when the index is built. Affine and corpus-wide, so it stays absolute: a first neighbour lands around 0.8, the 500th around 0.4, and a seed with only weak neighbours still looks weak. `min_item_score: 0.30` on that scale leaves a median of ~1,000 candidates, and rarely fewer than 50.

The neighbours read as a person's would. Photopia's are *9:05*, *I-0*, *Shade*, *Galatea*, *Violet* and *For a Change*; Counterfeit Monkey's are *Hadean Lands*, *City of Secrets* and *Savoir-Faire*. The tag route had given Photopia *BYOD [es]*, *Broken* and *A Normal Lost Phone* — games that share its tags and nothing else.

### Evaluating the item modes

Same test split, read the other way round. For each user with test positives, one of their *training* positives is the seed game, and the test positives are what a good "more like this" list should contain. For `author`, the seed is the author of one of the training positives, the author's own games are excluded (as the mode does), and the targets are the user's test positives by anyone else. Seeds are drawn with a fixed generator so every method scores the same queries, and the shipped approach is measured on the same protocol.

**Game mode** (1,742 queries; the reranked baseline on a 300-query sample, with the others re-scored on that sample for a paired comparison):

| | profile | profile + rerank | item encoder |
|---|---|---|---|
| Recall@10, same 300 | 0.0458 | 0.0569 | **0.0928** |
| Recall@25, same 300 | 0.0733 | 0.0853 | **0.1489** |
| Recall@50, same 300 | 0.0866 | 0.1261 | **0.1861** |
| MRR, same 300 | 0.0347 | 0.0337 | **0.0637** |
| Recall@10, all 1,742 | 0.0337 | — | **0.0899** |
| Recall@50, all 1,742 | 0.0702 | — | **0.2026** |

+63% Recall@10 and +75% Recall@25 over the full shipped pipeline, reranker included, and the gap widens with depth — the shape a co-occurrence model should have against a tag-similarity one. Absolute numbers are low because the question is hard: from one seed game, find the specific games this user went on to like, among ten thousand.

**Author mode** (1,629 queries; the reranked baseline on a 300-query sample, with the item methods re-scored on that sample):

| | profile | profile + rerank | item, centroid | item, max |
|---|---|---|---|---|
| Recall@10, same 300 | 0.0115 | 0.0214 | **0.0248** | 0.0181 |
| NDCG@10, same 300 | 0.0078 | 0.0124 | **0.0151** | 0.0131 |
| Recall@25, same 300 | 0.0270 | 0.0469 | **0.0495** | 0.0384 |
| Recall@50, same 300 | 0.0414 | **0.0627** | 0.0572 | 0.0689 |
| MRR, same 300 | 0.0091 | 0.0147 | **0.0171** | 0.0169 |
| Recall@10, all 1,629 | 0.0149 | — | 0.0261 | **0.0268** |
| Recall@25, all 1,629 | 0.0271 | — | **0.0578** | 0.0519 |
| Recall@50, all 1,629 | 0.0406 | — | 0.0762 | **0.0788** |

A smaller and less even win than game mode's: +16% Recall@10 and +22% NDCG@10 over the old route on the paired sample, a tie at 25, and the reranked route ahead at 50. The task is harder — the targets exclude the author's own games, so what is left is "what else did this author's fans like", a step removed from the seed — and an author of one game is a game query in all but name.

Centroid — the author's games averaged into one seed — and max-over-games are close, but centroid is ahead where it matters, the first page (Recall@10 0.0248 against 0.0181 on the paired sample), and max only pulls level at depth; the worry that a centroid blurs an eclectic catalogue did not materialise. `author_merge: centroid` is the default. Single-game authors (4,713 of 6,291) collapse to `game` mode either way.

## Experiments

### Which profile sections matter

Raw-retrieval ablation over the 1,920 test users: one section stripped from every profile at query time, encoders untouched. That measures what the trained query tower gets from a section, not what a tower trained without it would do — enough to rank them, not to price them exactly. Measured on the first 2026-09 models, which still carried `Era` in profiles and `Year:` in documents; that is why the table has a row the shipped profile no longer has.

| Profile | MRR | Recall@10 | Recall@50 | NDCG@10 | NegHit@10 |
|---|---|---|---|---|---|
| full | 0.1715 | 0.3634 | 0.5171 | 0.2063 | 0.0415 |
| − Authors | 0.0885 | 0.1552 | 0.2809 | 0.0970 | 0.0377 |
| − Era | 0.1728 | 0.3729 | 0.5253 | 0.2096 | 0.0474 |
| − Language | 0.1718 | 0.3644 | 0.5172 | 0.2069 | 0.0404 |
| − Dislikes | 0.1739 | 0.3673 | 0.5263 | 0.2087 | 0.0444 |
| systems + tags only | 0.0926 | 0.1630 | 0.2911 | 0.1011 | 0.0438 |

**Authors carries the retrieval gain.** Everything else is within noise of the full profile at this stage, and a query tower trained with authors present is lost without them — 0.155, below even the baseline's 0.229, because it was never asked to work from tags alone. The honest reading is that authorship is the strongest taste signal the data has, which matches the 24.5% of test positives that are by an author the user had already liked.

**Era and Language are neutral for the bi-encoder** — removing either even nudges recall up. Language stays: it costs nothing and it is the only way a Spanish-speaking reviewer's profile can say so. Era was removed, for the reasons under [Preparation](#preparation): neutral here, and a cutoff in effect.

**Dislikes are neutral for the bi-encoder, as predicted.** Mean-pooled cosine cannot negate: appending "Dislikes: puzzles" adds the token "puzzles" to the query vector and pulls it toward puzzle games, and fine-tuning did not teach the tower otherwise — NegHit@10 is 0.0415 with the section and 0.0444 without, a difference of two games in a thousand. The section's case rested on the reranker, which attends across both texts — and it does not make it either. With `Dislikes` stripped from every profile, the reranked pipeline scores NegHit@10 0.0906 against 0.0892 with it, Recall@10 0.4260 against 0.4292, MRR 0.2206 against 0.2263: a game or two in a thousand on the negative side, a hair on the positive one. Stripping at query time is a slight mismatch against what the cross-encoder trained on, so this is not a verdict on a reranker trained without the section, but it rules out anything large.

So the dislikes stay in the profile for what they are — an honest line in the query panel about what a reviewer rates low — not for anything they do to the ranking. Moving disliked games down measurably would need something other than more text: a hard demotion at ranking time on the tags in the list, or explicit negatives in the bi-encoder's loss, both untested.

### What the dump does and does not offer

Every table in the 2026-09 dump was looked at before the sections above were chosen. The ones not used, and why:

| Field or table | Coverage | Verdict |
|---|---|---|
| `games.forgiveness` | 26%, five real values and ~45 joke ones | Cheap to fold into tags; left out for now |
| `games.seriesname` | 14% (Eamon 281, Fallen London 111) | Tags carry it; same-series recommendations are either obvious or trivial |
| `games.license` | 80% | A filter attribute (free/commercial), not taste |
| `playertimes` | 1,661 games, 360 users | Length matters in IF, coverage is too thin; tags partly cover it |
| `compgames` (placement) | 4,957 games | A quality signal, already carried by the rating blend |
| `gametags` (per-user votes) | `games.tags` is exactly its aggregate | Nearly every vote count is one; weights add nothing |
| `reviews.review` (text) | 7,879 games with a ≥200-character review | The largest untapped signal: descriptions are author blurbs, reviews describe the experience. Deferred — it competes with the 256-token budget, so it means a longer sequence and a slower reranker everywhere |
| `users.gender` | M 963 / F 365 / unknown 2,572 among raters | Left out: a third covered, steers by demographic rather than taste, and opaque to the reader |
| `users.location`, `users.profile` | 17% / 720 raters, free text | Too sparse and out of distribution for the encoders |
| age | not recorded | — |
| `unwishlists` | 15,158 games, 131 users | One account marks nearly every game; no signal |

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

### Precomputing the common vibe picks

`vibe` was the only mode still scored at request time, which is why it was the
only one the move to ARM made slower — the other three have been table lookups
since `07_precompute.py` existed. It now has a table too: `precomputed_vibe.
parquet`, 1,050 keys over the top 5 systems paired with each of the top 20 tags
and each unordered pair of them, 375k rows, 7 MB, 26 minutes to build.

Deliberately one and two tags, which is the inverse of where the work is. Picks
of three or more tags are already cheap because `prefilter_tag_matches` requires
two matches of them; it is the one- and two-tag picks that neither that policy
nor the cap can help, and both of the slowest queries measured on the A1 were of
that shape. Both are now lookups.

A hit is also *better* than the live path, not merely faster: the offline job has
no latency budget, so it scores the whole pool uncapped, where a live query would
have stopped at `rerank_pool_cap` and paid 0.854 overlap@25 for it.

**Click order had to be collapsed first.** A multiselect reports values in the
order they were clicked, that order reaches the encoder as text, and
"Tags: horror, romance" against "Tags: romance, horror" returned pages differing
by 4-12% of their entries — two people wanting the same thing getting different
answers, and a table that would have had to store both spellings to catch
either. `canonical_vibe` orders picks by corpus frequency, which is both what
the pickers display and how the profiles the encoder trained on were built, so a
canonical query stays in the distribution the model saw.

**Coverage is the weak part, and it is not close to complete.** The top 5
systems carry 62.7% of all system assignments but the top 20 tags only 25.7% of
tag assignments, the tail being 5,237 tags long. If picks followed corpus
frequency a one-tag query would hit about 16% of the time and a two-tag query
about 4%. Corpus frequency is a poor stand-in for what someone picks off a menu
— the pickers list options in that same order, so real picks concentrate at the
top far harder than the corpus does — but the honest position is that the true
hit rate is unknown.

Widening has sharply diminishing returns, because the tag tail is long:

| systems x tags | keys | build | 1-tag hit (corpus proxy) |
|---|---|---|---|
| 5 x 20 (shipped) | 1,050 | 18 min | 16.1% |
| 5 x 40 | 4,100 | 68 min | 22.1% |
| 8 x 40 | 6,560 | 109 min | 25.5% |
| 10 x 50 | 12,750 | 212 min | 29.8% |

Twelve times the keys buys less than double the coverage, so the table should be
sized from evidence instead. Every vibe query already logs its picks, and a hit
logs `Precomputed vibe:` against `Live scoring:` for a miss — so after real
traffic, counting those two lines gives the actual hit rate, and the observed
picks give a far better candidate list than the corpus ordering does.

### Pruning the pool on tags rather than on rank

Moving to an Ampere A1 made `vibe` roughly 4.8x slower — 19.1 pairs/s against
91.5 on the quantized x86 box — which put the cap under pressure from the wrong
direction. Capping harder was the obvious response and the wrong one: the median
post-filter pool is 580, so a cap of 500 bites the *typical* query rather than
the tail, and it prunes on cosine rank, which predicts the reranker's order only
weakly (rho ~ 0.22). Two alternatives prune on the query's own tags instead.
`scripts/exp_vibe_prefilter.py` measures both over the same 300 held-out users
and query shapes as the cap sweep, deriving every variant from one full scoring
per query so the comparisons are exact.

**Raising the cosine floor is the worse lever**, and past 0.30 it stops being
free at all:

| floor | pairs | overlap@25 | Recall@10 |
|---|---|---|---|
| 0.25 (shipped) | 577 | 1.000 | 0.2005 |
| 0.30 | 358 | 0.752 | 0.2094 |
| 0.35 | 205 | 0.501 | 0.1790 |
| 0.45 | 60 | 0.210 | 0.0996 |

**Requiring two shared tags once the query offers three is the better one**, and
beats every cap on both axes at once (narrow = 1 system + 3 tags, broad = 2 + 6):

| policy | shape | pairs | overlap@25 | ΔNDCG@10 |
|---|---|---|---|---|
| tags>=1 (was shipped) | narrow | 577 | 1.000 | — |
| tags>=1 & system>=1 | narrow | 385 | 0.877 | +0.0017 |
| **tags>=2 of 3+** | **narrow** | **164** | **0.864** | **+0.0034** |
| **tags>=2 of 3+** | **broad** | **260** | **0.952** | **−0.0010** |

72% fewer pairs at higher fidelity than `rerank_pool_cap: 500` manages (0.854)
for a 14% cut. Quality does not move; the deltas are an order of magnitude
inside the intervals the cap sweep produced on the same 300 users, though
paired CIs were not recomputed for this run.

Three things this does *not* do, all worth knowing before reading the win as
larger than it is:

**It only reaches queries offering three or more tags.** A one-tag query cannot
be asked for two matches, and a two-tag query is left alone deliberately —
requiring both turns a vibe into a conjunction, cutting one measured pool from
818 candidates to 45, which is a different product and a page that may not fill.
Untested here, since both experiment shapes carry three tags or more.

**Fewer tags does not mean a smaller pool** — the opposite, which is what makes
the gap awkward. The broad shape retrieves *less* before filtering (792 against
1,011) because a more specific query embeds more specifically. Pool size tracks
how large the chosen system is: the slowest queries measured were `twine` ones.

**Requirements relax rather than empty.** A pool where nothing shares two tags
falls back to one, and then to no filtering — a strict rule allowed to return
nothing would post excellent latency by rendering a blank page.

Live paths only. Precompute keeps one shared tag for the same reason it ignores
`rerank_pool_cap`: no one is waiting on an offline job, and whole-profile
queries carry far more tags than the menu-style ones measured here.

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

Selecting the backend is the part that bit. `torch.backends.quantized.supported_engines`
lists what the wheel was compiled with, not what the CPU can execute, and the
two diverge on exactly the host this deploys to: the Linux aarch64 wheel
advertises `fbgemm`, so choosing from that list alone force-set an x86 backend
on an Ampere A1 and the first `linear_prepack` raised `RuntimeError: unknown
architecure`, exiting the service on every restart. macOS arm64 advertises only
`qnnpack`, which is why development never reproduced it. The engine is now
gated on `platform.machine()`, and `apply` catches anything the backend throws
and continues in fp32 — this is a speed optimization on a model that is correct
without it, so it may cost latency but never availability.

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
uv run scripts/07_precompute.py --mode all --top-n 500     # ~1.2 h
```

Nearly all of that is `reviewer` and `vibe`, the two modes that cross-encode. `game` and `author` are nearest-neighbour lookups now and take about forty seconds between them for 16,700 keys, where they used to take four hours; they are also stored at the full 500 for almost every key, since the item space rarely runs out of candidates above its floor.

| Artefact | Rows | Keys | Median depth | Size |
|---|---|---|---|---|
| `precomputed_userid.parquet` | 1,002,138 | 3,242 | 316 | 18.5 MB |
| `precomputed_gameid.parquet` | 4,246,018 | 10,282 | 500 | 76.5 MB |
| `precomputed_authorid.parquet` | 2,765,311 | 6,422 | 500 | 47.6 MB |
| `precomputed_vibe.parquet` | 369,897 | 1,050 | 382 | 6.7 MB |

Rows stream to Parquet in batches and publish by atomic rename, so a reader never sees a half-written file and a failed run cannot destroy the previous artefact. If a file is missing or unreadable, that mode falls back to live scoring — slower, never wrong.

**Re-run this after retraining anything**, or the app serves stale rankings.

Precompute ignores `rerank_pool_cap` — an offline job has no latency budget to protect, so these tables are built over the whole pool. It does pick up `quantize_reranker` through the shared loader, which on an x86 host halves the run. The tables currently shipped were built fp32; regenerating them under int8 is optional rather than required, since quality is unchanged either way.

Not everything is deep: 2.9% of users have fewer than 25 stored results (0.5% of games and 0.9% of authors), so a UI should report the true count rather than implying a full page.

### Resources

**Memory ~1.7 GB** with everything resident — dataframes, both FAISS indexes, embeddings, all three encoders and the reranker, and the four lookup tables. The item encoder and its index add about 120 MB.

**CPU is the real constraint**, and it is worth measuring rather than estimating: an Apple M-series laptop runs the cross-encoder at ~145 pairs/s on two threads, while the 2-vCPU OCI instance runs it at 45 — a 3.2× gap that no amount of reasoning about "a free-tier vCPU" would have pinned down. `scripts/measure_latency.py` reports it for a given host. That budget is fixed and shared, so concurrent requests divide it. Serialising inference (`concurrency_limit=1`) makes contention a visible queue rather than everyone slowing down at once.

Latency is close to linear in candidates scored, which makes it projectable. On the deployment VM:

```
fp32   t = 1.45 s + pairs / 45.2
int8   t = 0.82 s + pairs / 91.5
```

Both fit within 0.5 s across pools from 174 to 1,390 pairs. Note the fixed term: ~1 s of every query is setup that no amount of pool trimming touches, which is why capping below ~300 candidates stops paying for itself.

With `rerank_pool_cap: 1000` and int8 on fbgemm, a cold vibe query runs **6–7 s typical and 11.7 s worst case**, against 12–14 s / 38 s uncapped and unquantized. The cap does the tail and the quantization does the median; neither substitutes for the other.

On the A1 free-tier host none of that holds: int8 is skipped, the core scores 19.1 pairs/s, and the cap moved to 500 to compensate. The tag policy carries queries of three or more tags (a measured 56.4 s → 12.8 s), and the cap carries the one- and two-tag queries it cannot reach, at roughly 30 s. A one-tag query on a large system is the remaining worst case and neither lever touches it.

Only `vibe` consumes CPU, and it is cached two ways: per session while a user narrows filters, and process-wide across users keyed by (systems, tags). A repeat vibe query costs 0.02 s against several seconds cold. The cache holds 2,048 entries at ~85 KB each, about 171 MB.

The number worth watching in production is neither the cache nor the models: PyTorch's allocator grew ~470 MB across 28 scorings in testing and had not clearly plateaued, and it does so whether or not results are cached. A larger cache *reduces* that pressure, since every hit is a scoring that never runs.

---

## Possible extensions

- **Larger base models** — `all-mpnet-base-v2` for retrieval, DeBERTa-v3 for reranking. Better quality, worse latency, which matters for a lightweight deployment but could be useful in situations where everything would be precomputed.
- **Review text in the documents** — 7,879 games have a substantial review; the description is the author's pitch, a review is what playing it was like. Needs a longer `max_seq_length`, so it is a measured trade rather than a free addition.
- **A longer item-encoder run** — validation was still rising at the third epoch, and a second cross-encoder over (document, document) pairs is the natural next step if the item modes ever need one.
- **Session-based profiles** — a sliding window over recent positive ratings rather than a full-history aggregate.
- **Faster bi-encoder training** — wall time is dominated by the training epochs (6–16 min each), not the validation passes (15–50 s). Gains would come from larger batches or mixed precision.
- **Author identity** — `gameprofilelinks` maps games to IFDB user accounts, covering 38.5% of games. Too sparse to key author search on, but enough to deep-link authors to their profiles and to merge pen names that name matching cannot.
