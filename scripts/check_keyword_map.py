"""Advisory coverage checks for article research, cluster and related-link maps.

Missing research rows must not block publication. Use --strict only for an
explicit inventory review where complete coverage is the acceptance target.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from publication_routes import FRONT_MATTER, PERMALINK

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://mshojaei77.github.io"
LANG = re.compile(r"^lang:\s*['\"]?([a-z]{2})['\"]?\s*$", re.M)


def live_routes() -> dict[str, tuple[str, str]]:
    routes: dict[str, tuple[str, str]] = {}
    for source in sorted((ROOT / "_posts").glob("*")):
        if not source.is_file():
            continue
        front = FRONT_MATTER.match(source.read_text(encoding="utf-8"))
        if not front:
            continue
        permalink = PERMALINK.search(front.group(1))
        language = LANG.search(front.group(1))
        if permalink and language:
            route = permalink.group(1)
            routes[BASE_URL + route] = (route, language.group(1))
    return routes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="scripts/reports/keyword-page-map.csv")
    parser.add_argument("--strict", action="store_true", help="fail when advisory inventories are incomplete")
    args = parser.parse_args()

    with (ROOT / args.map).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    routes = live_routes()
    active_urls = set(routes)
    mapped_urls = [row.get("canonical_url", "").strip() for row in rows]
    mapped_set = set(mapped_urls)
    duplicates = sorted({url for url in mapped_urls if mapped_urls.count(url) > 1})
    missing = sorted(active_urls - mapped_set)
    stale = sorted(mapped_set - active_urls)
    malformed = [
        row.get("canonical_url", "")
        for row in rows
        if not row.get("primary_topic", "").strip()
        or not row.get("primary_query_candidate", "").strip()
        or not row.get("research_status", "").strip()
    ]

    topics = json.loads((ROOT / "_data/topics.json").read_text(encoding="utf-8"))
    active_slugs = {urlsplit(url).path.rstrip("/").split("/")[-1] for url in active_urls}
    article_clusters = topics.get("article_clusters", {})
    missing_clusters = sorted(active_slugs - set(article_clusters))
    stale_clusters = sorted(set(article_clusters) - active_slugs)
    known_clusters = set(topics.get("clusters", {}))
    invalid_cluster_ids = sorted({
        cluster_id
        for membership in article_clusters.values()
        for cluster_id in membership
        if cluster_id not in known_clusters
    })
    invalid_cluster_members = sorted({
        slug
        for cluster in topics.get("clusters", {}).values()
        for slug in cluster.get("articles", [])
        if slug not in active_slugs
    })
    active_route_lang = {route: lang for route, lang in routes.values()}
    slug_languages: dict[str, set[str]] = {}
    for route, lang in routes.values():
        slug = route.rstrip("/").split("/")[-1]
        slug_languages.setdefault(slug, set()).add(lang)
    related_map = topics.get("related_by_route", {})
    missing_related = sorted(route for route in active_route_lang if route not in related_map)
    invalid_related = []
    for source_route, suggestions in related_map.items():
        source_lang = active_route_lang.get(source_route)
        source_slug = source_route.rstrip("/").split("/")[-1]
        if source_lang is None:
            invalid_related.append(f"unknown source {source_route}")
            continue
        seen = set()
        for suggestion in suggestions:
            slug = suggestion.get("slug", "")
            if not slug or slug == source_slug or slug in seen or source_lang not in slug_languages.get(slug, set()):
                invalid_related.append(f"{source_route} -> {slug or '(missing slug)'}")
            seen.add(slug)

    counts = {
        "map_missing": len(missing),
        "map_stale": len(stale),
        "map_duplicate": len(duplicates),
        "map_malformed": len(malformed),
        "cluster_missing": len(missing_clusters),
        "cluster_stale": len(stale_clusters),
        "cluster_invalid": len(invalid_cluster_ids) + len(invalid_cluster_members),
        "related_missing": len(missing_related),
        "related_invalid": len(invalid_related),
    }
    covered = len(active_urls & mapped_set)
    print(f"Search inventory: {covered}/{len(active_urls)} routes mapped; {counts}")
    warnings = [
        (missing, "active route(s) lack a keyword-map row"),
        (stale, "keyword-map row(s) reference an inactive route"),
        (duplicates, "canonical URL(s) appear more than once in the map"),
        (malformed, "map row(s) lack a topic, candidate query or evidence status"),
        (missing_clusters, "active article slug(s) lack a topical cluster"),
        (stale_clusters, "cluster map entry(ies) reference an inactive article slug"),
        (invalid_cluster_ids or invalid_cluster_members, "cluster definitions contain invalid IDs or inactive slugs"),
        (missing_related, "active route(s) lack curated related-article ordering"),
        (invalid_related, "related-article suggestions contain an invalid or cross-language edge"),
    ]
    for values, message in warnings:
        if values:
            print(f"::warning::{len(values)} {message}. Advisory only; this does not block publication.")

    return 1 if args.strict and any(counts.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
