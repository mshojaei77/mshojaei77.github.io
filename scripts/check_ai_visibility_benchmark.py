"""Validate the manual AI-search query inventory and its observation ledger."""
import csv
from pathlib import Path

root = Path("scripts")
with (root / "ai_visibility_queries.csv").open(encoding="utf-8-sig", newline="") as source:
    queries = list(csv.DictReader(source))
assert 30 <= len(queries) <= 60, len(queries)
query_ids = {row["query_id"] for row in queries}
assert len(query_ids) == len(queries)
assert {row["query_language"] for row in queries} >= {"en", "fa", "mixed"}
assert {row["brand_status"] for row in queries} == {"branded", "nonbranded"}
assert sum(row["brand_status"] == "branded" for row in queries) >= 3
assert all(row["query_text"].strip() and row["topic"].strip() and row["intent"].strip() for row in queries)
assert all(row["query_status"] in {"untested", "observed"} for row in queries)

expected = ["query_id", "engine", "product_or_model", "search_enabled", "measurement_timestamp",
            "article_cited", "citation_url", "citation_position_if_available", "answer_relevance",
            "competing_sources", "observations"]
with (root / "ai_visibility_observations.csv").open(encoding="utf-8-sig", newline="") as source:
    reader = csv.DictReader(source)
    assert reader.fieldnames == expected
    observations = list(reader)
seen = set()
for row in observations:
    assert row["query_id"] in query_ids, row["query_id"]
    assert row["query_id"] not in seen, row["query_id"]
    assert row["engine"].strip() and row["product_or_model"].strip() and row["measurement_timestamp"].strip()
    assert row["search_enabled"] in {"true", "false", "unknown"}
    assert row["article_cited"] in {"yes", "no", "unknown"}
    assert row["observations"].strip()
    seen.add(row["query_id"])
for row in queries:
    assert (row["query_status"] == "observed") == (row["query_id"] in seen), row["query_id"]

untested = sum(row["query_status"] == "untested" for row in queries)
print(f"Validated {len(queries)} bilingual AI-search queries: {len(observations)} observed, {untested} untested; no citations are inferred.")
