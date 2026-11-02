# Labelling guide

These labels measure how far the LLM judge can be trusted. Each row is a GOV.UK guidance
page and a one-sentence summary of it. The model that wrote each summary is hidden.

## What to do

1. Open `human_labels.csv` in a spreadsheet app.
2. For each row, read `page` and `summary`. `reference` is GOV.UK's own description of
   the page, shown for context. It is not a perfect answer.
3. Put a whole number from 0 to 10 in `score`. Add anything useful in `notes`.
4. Label all 15 on your own, before looking at `review.csv`. That file is created
   afterwards and holds Claude's draft labels.

## The scale

This follows the judge's rubric (`rubrics/summary_quality.yaml`).

| Score | Meaning |
|---|---|
| 9-10 | States the page's main point accurately and completely, in one sentence, with no unsupported claim. |
| 7-8 | Accurate main point; misses a minor key fact, or is slightly long. |
| 5-6 | Broadly right, but misses an important fact, is vague, or runs past one sentence. |
| 3-4 | Wrong emphasis, notable omission, or a claim the page does not support. |
| 0-2 | Wrong, invented, or not a summary at all. |

Penalise a summary for claims the page doesn't support, for being more than one
sentence, and for preamble or filler. The task asked for at most 30 words, so a much
longer summary should lose a point or two even if it is accurate.

## Steps after your 15

1. `evalverdict labels export-review` creates `review.csv` with Claude's drafts for the
   other 35 items. Change `reviewed_score` wherever you disagree.
2. `evalverdict labels import` combines everything and records what changed in review.

The file `private/key.json`, which says which model wrote each summary, is gitignored
and must not be opened until labelling is finished.
