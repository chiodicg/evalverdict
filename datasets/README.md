# Datasets

Every dataset committed here has a confirmed licence and a recorded build. Hashes must
match the files exactly; tests check this.

| File | Rows | SHA-256 | Licence |
|---|---|---|---|
| summaries.jsonl | 200 | 42ee63f300e118a33570f1f6108a8a8afadd64d62f7d836399640b4e4ee5c403 | Open Government Licence v3.0 |

## summaries.jsonl

- **Source:** GOV.UK detailed guidance pages, fetched through the GOV.UK Content API
  (https://www.gov.uk/api/content) on 2026-11-02.
- **Licence:** Open Government Licence v3.0
  (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- **Attribution:** Contains public sector information licensed under the Open Government Licence v3.0.
- **Task:** summarise a guidance page in one sentence. `input` is the page title and body
  text; `reference` is the page's own description, written by GOV.UK editors. The demo
  measures one-sentence summarisation, not long-form summaries.
- **Build:** `python scripts/build_govuk_dataset.py --out datasets/summaries.jsonl`
  (seed 42). From the first 1,500 guidance links, pages are kept if their body has
  150-1,200 words and their description at least 8 words, until 200 are found.
- **Changes from the source:** HTML is converted to plain text, and email addresses and
  phone numbers are replaced with `[email address]` and `[phone number]`, because
  contact details are not needed for summarising and some may name individuals.
- **Rebuilding:** GOV.UK pages change, so a rebuild may differ. The committed file is the
  reference, and its hash is recorded above.
- **Approval:** Carla chose this source and licence on 2026-11-02.
