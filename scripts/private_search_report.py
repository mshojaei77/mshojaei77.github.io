"""Build a private, source-stamped SEO operations report from real local evidence.

The script reads public production-audit data and the optional private distribution
registry. It writes only outside the repository and never fills provider metrics
that were not supplied by an authorized export.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from private_metric_imports import ImportErrorDetail, import_manifest, summarize_imports

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = (
    Path.home() / "AppData" / "Local" / "MohammadShojaei" /
    "PublicationOperations" / "distribution-registry.json"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path, help="strict production audit JSON")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="private publication registry")
    parser.add_argument("--checkpoint", type=Path, help="optional private copy of the IndexNow checkpoint")
    parser.add_argument("--external-links", type=Path, help="optional sampled external-reference audit CSV")
    parser.add_argument("--browser-audit", type=Path, help="optional production Playwright/axe lab results JSON")
    parser.add_argument("--imports-manifest", type=Path, help="optional private manifest for official provider exports")
    parser.add_argument("--output-dir", required=True, type=Path, help="private directory outside the repository")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir == ROOT.resolve() or ROOT.resolve() in output_dir.parents:
        parser.error("--output-dir must be outside the public repository")

    audit = read_json(args.audit.resolve())
    if not audit.get("strict_check", {}).get("passed"):
        parser.error("production audit is not a passing strict audit")
    rows = audit["urls"] if isinstance(audit["urls"], list) else list(audit["urls"].values())
    article_rows = [row for row in rows if row.get("rss_membership") and "/writing/" in row.get("url", "")]
    map_path = ROOT / "scripts" / "reports" / "keyword-page-map.csv"
    with map_path.open(encoding="utf-8-sig", newline="") as source:
        keyword_rows = list(csv.DictReader(source))
    active_urls = {row["canonical_url"] for row in keyword_rows}
    with (ROOT / "scripts" / "ai_visibility_queries.csv").open(encoding="utf-8-sig", newline="") as source:
        ai_queries = list(csv.DictReader(source))
    with (ROOT / "scripts" / "ai_visibility_observations.csv").open(encoding="utf-8-sig", newline="") as source:
        ai_observations = list(csv.DictReader(source))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_revision": audit.get("deployment_revision"),
        "audit_checked_at_utc": audit.get("checked_at_utc"),
        "production_audit": {
            "strict_passed": audit["strict_check"]["passed"],
            "urls_crawled": audit["inventory_counts"]["production_urls_crawled"],
            "sitemap_urls": audit["inventory_counts"]["sitemap_urls"],
            "sitemap_indexable_pages": audit["inventory_counts"]["sitemap_indexable_pages"],
            "articles_en": audit["inventory_counts"]["articles_en"],
            "articles_fa": audit["inventory_counts"]["articles_fa"],
            "rss_article_urls": audit["inventory_counts"]["rss_article_urls"],
            "issue_counts": audit["issue_counts"],
            "sitemap_http_status": audit["discovery_endpoints"]["/sitemap.xml"]["http_status"],
            "sitemap_xml_valid": audit["discovery_endpoints"]["/sitemap.xml"]["xml_valid"],
        },
        "keyword_inventory": {
            "mapped_article_routes": len(active_urls),
            "unique_mapped_routes": len({row["canonical_url"] for row in keyword_rows}),
            "coverage_against_current_article_inventory": (
                len(active_urls & {row["url"] for row in article_rows}) / len(article_rows)
                if article_rows else None
            ),
            "measured_search_volume_or_rank_rows": 0,
            "research_status": "candidate_queries_are_not_provider_measurements",
        },
        "internal_article_graph": {
            "article_routes": len(article_rows),
            "article_to_article_outlinks": sum(len(row.get("article_outlinks", [])) for row in article_rows),
            "articles_with_no_article_inlinks": sum(not row.get("article_inlinks") for row in article_rows),
            "articles_with_no_contextual_article_inlinks": sum(not row.get("contextual_article_inlinks") for row in article_rows),
            "articles_with_no_related_component_inlinks": sum(not row.get("related_article_inlinks") for row in article_rows),
        },
        "provider_measurements": {
            "google_search_console": "UNKNOWN_NOT_AUTHORIZED_OR_NO_EXPORT",
            "bing_webmaster": "UNKNOWN_NOT_AUTHORIZED_OR_NO_EXPORT",
            "bing_ai_performance": "UNKNOWN_NOT_AUTHORIZED_OR_NO_EXPORT",
            "goatcounter": "UNKNOWN_NOT_AUTHORIZED_OR_NO_EXPORT",
        },
        "ai_visibility": {
            "query_count": len(ai_queries),
            "untested_query_count": sum(row.get("query_status") == "untested" for row in ai_queries),
            "observations_count": len(ai_observations),
            "citation_count": "UNKNOWN",
            "status": "untested" if not ai_observations else "observations_available",
        },
    }

    if args.imports_manifest:
        manifest_path = args.imports_manifest.expanduser().resolve()
        if manifest_path == ROOT.resolve() or ROOT.resolve() in manifest_path.parents:
            parser.error("--imports-manifest must be outside the public repository")
        ledger_path = output_dir / "provider-import-ledger.json"
        known_fingerprints = set()
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            if ledger.get("schema_version") != 1 or not isinstance(ledger.get("fingerprints"), list):
                parser.error("private provider import ledger has an unsupported or invalid schema")
            known_fingerprints = {str(value) for value in ledger["fingerprints"]}
        try:
            imported, import_warnings, known_fingerprints = import_manifest(
                manifest_path, output_dir, known_fingerprints
            )
        except (ImportErrorDetail, OSError, json.JSONDecodeError) as exc:
            parser.error(f"provider export import failed: {exc}")
        snapshot_time = datetime.now(timezone.utc)
        generated_at = snapshot_time.isoformat()
        snapshot_path = output_dir / f"provider-metrics-{report['deployment_revision']}-{snapshot_time.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        snapshot = {
            "schema_version": 1,
            "source_revision": report["deployment_revision"],
            "generated_at_utc": generated_at,
            "measurement_semantics": "Provider exports are kept separate; overlapping ranges are not summed. GoatCounter hits are not unique-human counts.",
            "imports": imported,
            "aggregate_summary": summarize_imports(imported),
            "warnings": import_warnings,
        }
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_states = {}
        for entry in imported:
            source = entry["summary"]["source"]
            state = "DATA_AVAILABLE_FROM_EXPORT" if entry["summary"]["import_status"] == "IMPORTED" else "DUPLICATE_EXPORT_NO_NEW_DATA"
            if source_states.get(source) != "DATA_AVAILABLE_FROM_EXPORT":
                source_states[source] = state
        for key, source in (("google_search_console", "google_search_console"), ("bing_webmaster", "bing_webmaster"), ("bing_ai_performance", "bing_ai_performance"), ("goatcounter", "goatcounter")):
            if source in source_states:
                report["provider_measurements"][key] = source_states[source]
        report["provider_imports"] = {
            "manifest_schema_version": 1,
            "import_count": sum(entry["summary"]["import_status"] == "IMPORTED" for entry in imported),
            "duplicate_skipped_count": sum(entry["summary"]["import_status"] == "DUPLICATE_SKIPPED" for entry in imported),
            "sources": dict(Counter(entry["summary"]["source"] for entry in imported)),
            "warnings": import_warnings,
            "normalized_snapshot": snapshot_path.name,
            "private_details_in_report": False,
        }

    if args.registry.exists():
        registry = read_json(args.registry.resolve())
        records = registry.get("records", [])
        report["external_publications"] = {
            "registry_validated_record_count": len(records),
            "platform_counts": dict(Counter(row.get("platform", "unknown") for row in records)),
            "mapped_to_blog_article": sum(bool(row.get("article_url")) for row in records),
            "not_migrated": sum(row.get("article_status") == "not_migrated" for row in records),
            "canonical_conflict_count": sum(row.get("canonical_status") == "conflict" for row in records),
            "canonical_state_counts": dict(Counter(row.get("canonical_status", "unknown") for row in records)),
            "private_detail_included": False,
        }
    else:
        report["external_publications"] = {"status": "registry_not_found"}

    if args.checkpoint and args.checkpoint.exists():
        checkpoint = read_json(args.checkpoint.resolve())
        report["indexnow"] = {
            "last_acknowledged_revision": checkpoint.get("last_acknowledged_revision"),
            "notification_state": checkpoint.get("notification_state"),
            "canonical_url_count": len(checkpoint.get("known_canonical_urls", [])),
            "acknowledged_at": checkpoint.get("last_acknowledged_at"),
        }

    if args.external_links and args.external_links.exists():
        with args.external_links.open(encoding="utf-8-sig", newline="") as source:
            reference_rows = list(csv.DictReader(source))
        report["external_reference_health"] = {
            "sampled_unique_urls": len(reference_rows),
            "classification_counts": dict(Counter(row.get("classification", "unknown") for row in reference_rows)),
            "confirmed_broken_urls": sum(row.get("classification") in {"http_404", "http_410", "dns_failure"} for row in reference_rows),
            "scope": "frequency-ranked sample only; non-success responses are not automatically classified as broken",
        }

    if args.browser_audit and args.browser_audit.exists():
        browser_audit = read_json(args.browser_audit.resolve())
        browser_results = browser_audit.get("results", [])
        violations = [
            {"page": row.get("name"), "device": row.get("device"), **violation}
            for row in browser_results
            for violation in row.get("a11y", {}).get("violations", [])
        ]
        report["browser_lab_audit"] = {
            "measured_at": browser_audit.get("measured_at"),
            "environment": browser_audit.get("environment"),
            "page_device_runs": len(browser_results),
            "http_failures": sum(row.get("status") != 200 for row in browser_results),
            "page_errors": sum(len(row.get("page_errors", [])) for row in browser_results),
            "axe_violations": violations,
            "core_web_vitals": "NOT_MEASURED",
            "field_data": "NOT_AVAILABLE_FROM_THIS_LAB_RUN",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"search-operations-{report['deployment_revision']}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PRIVATE_REPORT={output}")
    print(f"DEPLOYMENT_REVISION={report['deployment_revision']}")
    print("PRIVATE_PROVIDER_METRICS=" + ("STORED_ONLY_IN_PRIVATE_OUTPUT" if args.imports_manifest else "NOT_IMPORTED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
