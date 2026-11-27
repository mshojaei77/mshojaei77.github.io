"""Synthetic fixtures for private provider-export normalization."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_metric_imports import ImportErrorDetail, import_manifest, summarize_imports  # noqa: E402


class PrivateMetricImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="private-metric-import-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, data):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "imports": data}), encoding="utf-8")
        return path

    @staticmethod
    def base(source, report_type, fmt, filename):
        return {
            "source": source,
            "report_type": report_type,
            "format": fmt,
            "file": filename,
            "exported_at": "2026-11-28T10:00:00Z",
            "period_start": "2026-11-01",
            "period_end": "2026-11-27",
            "filters": {"search_type": "web"},
            "scope": {"property": "https://mshojaei77.github.io/"},
        }

    def test_google_search_analytics_api_preserves_fractional_ctr_and_aggregation(self):
        (self.root / "gsc.json").write_text(json.dumps({
            "responseAggregationType": "byPage",
            "rows": [{"keys": ["RAG", "MOBILE"], "clicks": 3, "impressions": 24, "ctr": 0.125, "position": 7.5}],
        }), encoding="utf-8")
        item = self.base("google_search_console", "search_performance", "gsc_api_json", "gsc.json")
        item["dimensions"] = ["query", "device"]
        result, warnings, _ = import_manifest(self.manifest([item]), self.root / "out")
        imported = result[0]
        self.assertEqual(imported["rows"][0]["metrics"]["ctr"], 0.125)
        self.assertEqual(imported["rows"][0]["dimensions"], {"query": "RAG", "device": "MOBILE"})
        self.assertEqual(imported["summary"]["response_aggregation_type"], "byPage")
        self.assertEqual(warnings, [])

    def test_gsc_and_bing_table_import_require_explicit_column_maps(self):
        (self.root / "gsc.csv").write_text("Query,Clicks,Impressions,CTR,Position\nRAG,3,24,12.5%,7.5\n", encoding="utf-8")
        item = self.base("google_search_console", "search_performance", "table_csv", "gsc.csv")
        item["columns"] = {"dimensions": {"query": "Query"}, "metrics": {"clicks": "Clicks", "impressions": "Impressions", "ctr": "CTR", "position": "Position"}}
        result, _, _ = import_manifest(self.manifest([item]), self.root / "out")
        self.assertAlmostEqual(result[0]["rows"][0]["metrics"]["ctr"], 0.125)

        (self.root / "bing.json").write_text(json.dumps([{"Search Query": "RAG", "Clicks": 2, "Impressions": 18}]), encoding="utf-8")
        bing = self.base("bing_webmaster", "search_performance", "mapped_json", "bing.json")
        bing["columns"] = {"dimensions": {"query": "Search Query"}, "metrics": {"clicks": "Clicks", "impressions": "Impressions"}}
        result, _, _ = import_manifest(self.manifest([bing]), self.root / "out")
        self.assertEqual(result[0]["rows"][0]["metrics"], {"clicks": 2, "impressions": 18})

        del bing["columns"]
        with self.assertRaises(ImportErrorDetail):
            import_manifest(self.manifest([bing]), self.root / "out")

    def test_goatcounter_csv_discards_session_and_referrer_query_data(self):
        path = self.root / "goat.csv"
        headers = ["2", "Path", "Title", "Event", "User-Agent", "Browser", "System", "Session", "Bot", "Referrer", "Referrer scheme", "Screen size", "Location", "FirstVisit", "Date"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerow(["/fa/writing/rag/?token=private", "RAG", "false", "", "Chrome", "Windows", "session-secret", "0", "https://example.org/post?secret=private", "h", "1440x900", "US-CA", "true", "2026-11-20T12:00:00Z"])
            writer.writerow(["/fa/writing/rag/", "RAG", "false", "", "Chrome", "Windows", "other-session", "0", "telegram", "g", "1440x900", "US", "false", "2026-11-20T12:30:00Z"])
        item = self.base("goatcounter", "pageviews", "goatcounter_csv", "goat.csv")
        result, _, _ = import_manifest(self.manifest([item]), self.root / "out")
        payload = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("session-secret", payload)
        self.assertNotIn("secret=private", payload)
        self.assertEqual(len(result[0]["rows"]), 2)
        self.assertEqual(result[0]["rows"][0]["dimensions"]["path"], "/fa/writing/rag/")
        self.assertEqual(result[0]["rows"][0]["dimensions"]["referrer_source"], "example.org")

    def test_goatcounter_json_zip_aggregates_paths_and_skips_event_paths(self):
        path = self.root / "goat.zip"
        entries = {
            "info.json": {"export_version": "1.0", "created_at": "2026-11-28T00:00:00Z"},
            "paths.jsonl": [
                {"id": 1, "path": "/fa/writing/document-parsing/"},
                {"id": 2, "event": True, "path": "copy-code"},
            ],
            "refs.jsonl": [{"id": 0, "ref": "", "ref_scheme": "o"}, {"id": 7, "ref": "Google", "ref_scheme": "g"}],
            "hit_stats.jsonl": [
                {"hour": "2026-11-20T12:00:00Z", "path_id": 1, "ref_id": 7, "count": 4},
                {"hour": "2026-11-20T13:00:00Z", "path_id": 1, "ref_id": 7, "count": 2},
                {"hour": "2026-11-20T13:00:00Z", "path_id": 2, "ref_id": 0, "count": 9},
            ],
        }
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in entries.items():
                if name.endswith(".jsonl"):
                    body = "".join(json.dumps(row) + "\n" for row in value)
                else:
                    body = json.dumps(value)
                archive.writestr(name, body)
        item = self.base("goatcounter", "pageviews", "goatcounter_json_zip", "goat.zip")
        result, _, _ = import_manifest(self.manifest([item]), self.root / "out")
        rows = result[0]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"]["hits_total"], 6)
        self.assertEqual(rows[0]["dimensions"]["language"], "fa")
        summary = summarize_imports(result)
        pageviews = summary["sources"]["goatcounter:pageviews"]
        self.assertEqual(pageviews["reported_hits"], 6)
        self.assertEqual(pageviews["by_language"], {"fa": 6})
        self.assertEqual(pageviews["status"], "IMPORTED")

    def test_identical_export_is_skipped_and_overlaps_are_not_summed(self):
        (self.root / "one.json").write_text(json.dumps({"responseAggregationType": "byPage", "rows": []}), encoding="utf-8")
        item = self.base("google_search_console", "search_performance", "gsc_api_json", "one.json")
        item["dimensions"] = ["query"]
        manifest = self.manifest([item, item])
        results, _, known = import_manifest(manifest, self.root / "out")
        self.assertEqual([row["summary"]["import_status"] for row in results], ["IMPORTED", "DUPLICATE_SKIPPED"])
        results_again, _, _ = import_manifest(manifest, self.root / "out", known)
        self.assertTrue(all(row["summary"]["import_status"] == "DUPLICATE_SKIPPED" for row in results_again))
        duplicate_summary = summarize_imports(results_again)["sources"]["google_search_console:search_performance"]
        self.assertEqual(duplicate_summary["status"], "DUPLICATE_EXPORT_NO_NEW_DATA")
        self.assertNotIn("exported_row_sums_only", duplicate_summary)


if __name__ == "__main__":
    unittest.main()
