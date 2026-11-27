"""Validate the manual AI-search visibility query set and empty observation template."""
import csv
from pathlib import Path

root = Path("scripts")
with (root / "ai_visibility_queries.csv").open(encoding="utf-8-sig", newline="") as source:
    queries = list(csv.DictReader(source))
assert 30 <= len(queries) <= 50, len(queries)
assert len({row["query_id"] for row in queries}) == len(queries)
assert {row["query_language"] for row in queries} >= {"en", "fa", "mixed"}
assert {row["brand_status"] for row in queries} == {"branded", "nonbranded"}
assert sum(row["brand_status"] == "branded" for row in queries) >= 3
assert all(row["query_text"].strip() and row["topic"].strip() and row["intent"].strip() for row in queries)
expected = ["query_id", "engine", "product_or_model", "search_enabled", "measurement_timestamp",
            "article_cited", "citation_url", "citation_position_if_available", "answer_relevance",
            "competing_sources", "observations"]
with (root / "ai_visibility_observations.csv").open(encoding="utf-8-sig", newline="") as source:
    observations = csv.reader(source)
    assert next(observations) == expected
    assert next(observations, None) is None, "Observation template must stay empty until real manual measurements exist."
print(f"Validated {len(queries)} bilingual AI-search queries; no answer/citation results are fabricated.")
