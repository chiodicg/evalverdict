"""Build the GOV.UK guidance summarisation dataset.

Email addresses and phone numbers are replaced with placeholders.

Each row pairs a GOV.UK guidance page (title and body text) with the page's own
one-sentence description, written by GOV.UK editors, as the reference summary.

Content is reused under the Open Government Licence v3.0:
"Contains public sector information licensed under the Open Government Licence v3.0."

Run from the repository root:
    python scripts/build_govuk_dataset.py --out datasets/summaries.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import urllib.parse
import urllib.request
from typing import Any

SEARCH = "https://www.gov.uk/api/search.json"
CONTENT = "https://www.gov.uk/api/content"
USER_AGENT = "evalverdict-dataset-builder (github.com/chiodicg/evalverdict)"
PAUSE_S = 0.25
MIN_WORDS, MAX_WORDS = 150, 1200
MIN_DESCRIPTION_WORDS = 8
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?:\+44\s?|\b0)(?:\d[\s-]?){9,10}\b")


def redact_contacts(text: str) -> str:
    """Contact details are not needed for summarising and may name individuals."""
    return PHONE.sub("[phone number]", EMAIL.sub("[email address]", text))


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def guidance_links(limit: int) -> list[str]:
    links: list[str] = []
    start = 0
    while len(links) < limit:
        query = urllib.parse.urlencode(
            {
                "filter_format": "detailed_guide",
                "count": 500,
                "start": start,
                "fields": "link",
                "order": "public_timestamp",
            }
        )
        results = fetch_json(f"{SEARCH}?{query}")["results"]
        if not results:
            break
        links.extend(result["link"] for result in results)
        start += len(results)
        time.sleep(PAUSE_S)
    return sorted(set(links))[:limit]


def plain_text(body_html: str) -> str:
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", body_html, flags=re.S)
    text = re.sub(r"</(p|h[1-6]|li|tr|div)>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def page_row(link: str) -> dict[str, Any] | None:
    page = fetch_json(CONTENT + link)
    body = (page.get("details") or {}).get("body") or ""
    if isinstance(body, list):
        body = " ".join(part.get("content", "") for part in body)
    text = redact_contacts(plain_text(body))
    description = redact_contacts((page.get("description") or "").strip())
    words = len(text.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        return None
    if len(description.split()) < MIN_DESCRIPTION_WORDS:
        return None
    return {
        "id": link.removeprefix("/guidance/"),
        "input": f"{page['title']}\n\n{text}",
        "reference": description,
        "metadata": {"source_url": f"https://www.gov.uk{link}", "words": words},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--pool", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    links = guidance_links(args.pool)
    random.Random(args.seed).shuffle(links)
    rows: list[dict[str, Any]] = []
    for link in links:
        if len(rows) >= args.size:
            break
        try:
            row = page_row(link)
        except (OSError, ValueError, KeyError):
            row = None
        if row is not None:
            rows.append(row)
        time.sleep(PAUSE_S)
    rows.sort(key=lambda row: row["id"])
    with open(args.out, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows from a pool of {len(links)} guidance pages")


if __name__ == "__main__":
    main()
