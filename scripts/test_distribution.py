from __future__ import annotations

import unittest
from datetime import date

import distribution


class TrackingLinkTests(unittest.TestCase):
    def test_clean_canonical_url(self) -> None:
        actual = distribution.tracking_url(
            "https://mshojaei77.github.io/fa/writing/document-parsing/",
            "telegram", "social", "document-parsing",
        )
        self.assertEqual(
            actual,
            "https://mshojaei77.github.io/fa/writing/document-parsing/?utm_source=telegram&utm_medium=social&utm_campaign=document-parsing",
        )

    def test_independent_discovery_campaign_source(self) -> None:
        actual = distribution.tracking_url(
            "https://mshojaei77.github.io/writing/deep-research-agents/",
            "hackernews", "social", "deep-research-agents", "regular-story",
        )
        self.assertEqual(
            actual,
            "https://mshojaei77.github.io/writing/deep-research-agents/?utm_source=hackernews&utm_medium=social&utm_campaign=deep-research-agents&utm_content=regular-story",
        )
        self.assertNotIn("utm_", distribution.urlsplit(actual).path)

    def test_preserves_non_utm_params_unicode_and_fragment_and_replaces_old_utm(self) -> None:
        actual = distribution.tracking_url(
            "https://mshojaei77.github.io/writing/deep-research-agents/?view=full&utm_source=old&utm_campaign=old#references",
            "medium", "syndication", "deep-research-agents", "نسخه ۲",
        )
        parsed = distribution.urlsplit(actual)
        pairs = distribution.parse_qsl(parsed.query, keep_blank_values=True)
        self.assertEqual(parsed.fragment, "references")
        self.assertEqual([(k, v) for k, v in pairs if k == "view"], [("view", "full")])
        self.assertEqual([(k, v) for k, v in pairs if k == "utm_source"], [("utm_source", "medium")])
        self.assertEqual([(k, v) for k, v in pairs if k == "utm_campaign"], [("utm_campaign", "deep-research-agents")])
        self.assertEqual([(k, v) for k, v in pairs if k == "utm_content"], [("utm_content", "نسخه ۲")])

    def test_rejects_external_destinations_and_empty_fields(self) -> None:
        for url in ("javascript:alert(1)", "file:///C:/secret", "https://example.com/writing/a/", "http://mshojaei77.github.io/writing/a/"):
            with self.subTest(url=url), self.assertRaises(distribution.DistributionError):
                distribution.tracking_url(url, "telegram", "social", "article")
        with self.assertRaises(distribution.DistributionError):
            distribution.tracking_url("https://mshojaei77.github.io/writing/a/", "telegram", "social", "")


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "article_id": "en-deep-research-agents",
            "article_url": "https://mshojaei77.github.io/writing/deep-research-agents/",
            "platform": "medium",
            "external_url": "https://medium.com/@mshojaei77/building-deep-research-agents-abcdef123456",
            "relationship": "older_edition",
            "publication_status": "published_owner_confirmed",
            "canonical_status": "correct",
            "last_verified_at": "2026-11-27T12:00:00Z",
            "evidence": "Rendered page canonical inspected.",
        }

    def test_validation_and_duplicate_detection(self) -> None:
        registry = distribution.validate_registry({"schema_version": 1, "records": [self.record]})
        self.assertEqual(len(registry["records"]), 1)
        with self.assertRaisesRegex(distribution.DistributionError, "Duplicate registry record"):
            distribution.validate_registry({"schema_version": 1, "records": [self.record, self.record]})

    def test_rejects_untrusted_urls_bad_status_and_wrong_version(self) -> None:
        malicious = {**self.record, "external_url": "javascript:alert(1)"}
        with self.assertRaises(distribution.DistributionError):
            distribution.validate_registry({"schema_version": 1, "records": [malicious]})
        with self.assertRaises(distribution.DistributionError):
            distribution.validate_registry({"schema_version": 2, "records": []})
        bad_status = {**self.record, "canonical_status": "verified_because_i_said_so"}
        with self.assertRaises(distribution.DistributionError):
            distribution.validate_registry({"schema_version": 1, "records": [bad_status]})

    def test_telegraph_records_are_supported_and_registry_writes_stay_private(self) -> None:
        telegraph = {
            **self.record,
            "article_id": "fa-rl-in-llms",
            "article_url": "https://mshojaei77.github.io/fa/writing/rl-in-llms/",
            "platform": "telegraph",
            "external_url": "https://telegra.ph/RL-in-LLMs-09-14",
            "relationship": "older_edition",
            "canonical_status": "unknown",
        }
        self.assertEqual(len(distribution.validate_registry({"schema_version": 1, "records": [telegraph]})["records"]), 1)
        with self.assertRaisesRegex(distribution.DistributionError, "outside the public repository"):
            distribution.safe_output_dir(distribution.REPO_ROOT / "private.json")

    def test_safe_merge_does_not_downgrade_confirmed_state(self) -> None:
        weak = {**self.record, "publication_status": "unknown", "canonical_status": "unverified", "last_verified_at": None, "evidence": ""}
        merged = distribution.merge_registries(
            {"schema_version": 1, "records": [self.record]},
            {"schema_version": 1, "records": [weak]},
        )
        self.assertEqual(len(merged["records"]), 1)
        self.assertEqual(merged["records"][0]["publication_status"], "published_owner_confirmed")
        self.assertEqual(merged["records"][0]["canonical_status"], "correct")
        self.assertEqual(merged["records"][0]["last_verified_at"], self.record["last_verified_at"])


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.article = {
            "title": "A technical article",
            "description": "A concise description.",
            "lang": "en",
            "route": "/writing/a-technical-article/",
            "url": "https://mshojaei77.github.io/writing/a-technical-article/",
            "article_id": "en-a-technical-article",
            "sha256": "a" * 64,
            "image": "/assets/img/social-card.png",
            "extension": ".md",
            "body": "## Details\n\n[About](/about/)\n\n| A | B |\n| - | - |\n| x | y |\n",
        }

    def test_full_copy_requires_manual_production_confirmation(self) -> None:
        with self.assertRaisesRegex(distribution.DistributionError, "production article"):
            distribution.export_files(self.article, "devto", live_confirmed=False)

    def test_dev_export_is_draft_safe_and_keeps_links_and_table(self) -> None:
        files = distribution.export_files(self.article, "devto", live_confirmed=True)
        body = files["devto.md"]
        self.assertIn("published: false", body)
        self.assertIn("canonical_url: https://mshojaei77.github.io/writing/a-technical-article/", body)
        self.assertIn("https://mshojaei77.github.io/about/", body)
        self.assertIn("| A | B |", body)
        self.assertNotIn("source_url", body)

    def test_medium_instructions_use_blog_first_import(self) -> None:
        files = distribution.export_files(self.article, "medium", live_confirmed=True)
        self.assertIn("Use Medium's official Import a story flow", files["instructions.md"])
        self.assertIn(self.article["url"], files["instructions.md"])

    def test_social_exports_are_authored_templates_not_summaries(self) -> None:
        telegram = distribution.export_files(self.article, "telegram", live_confirmed=False)["telegram-template.md"]
        linkedin = distribution.export_files(self.article, "linkedin", live_confirmed=False)["linkedin-template.md"]
        self.assertIn("[خلاصه را با لحن خودتان بنویسید.]", telegram)
        self.assertIn("[یک نکتهٔ مستند را با محدودیت‌هایش بیان کنید.]", linkedin)
        self.assertNotIn("Details", telegram)

    def test_refuses_jekyll_and_unsafe_html(self) -> None:
        liquid = {**self.article, "body": "{{ site.url }}"}
        with self.assertRaisesRegex(distribution.DistributionError, "Jekyll Liquid"):
            distribution.export_files(liquid, "devto", live_confirmed=True)
        unsafe = {**self.article, "extension": ".html", "body": "<img src=x onerror=alert(1)>"}
        with self.assertRaisesRegex(distribution.DistributionError, "script, event-handler"):
            distribution.export_files(unsafe, "medium", live_confirmed=True)

    def test_blog_posts_are_live_and_keep_filename_dates(self) -> None:
        posts = sorted((distribution.REPO_ROOT / "_posts").glob("*"))
        checked = 0
        for path in posts:
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            fields, _body = distribution.read_front_matter(source)
            self.assertNotEqual(fields.get("published", "true").lower(), "false", path.name)
            self.assertTrue(fields.get("date"), path.name)
            self.assertLessEqual(date.fromisoformat(fields["date"][:10]), date.today(), path.name)
            checked += 1
        self.assertGreaterEqual(checked, 43)

    def test_public_articles_have_no_migration_provenance_fields(self) -> None:
        for path in sorted((distribution.REPO_ROOT / "_posts").glob("*")):
            if not path.is_file():
                continue
            fields, _body = distribution.read_front_matter(path.read_text(encoding="utf-8"))
            self.assertNotIn("source_url", fields, path.name)
            self.assertNotIn("publication_origin", fields, path.name)


if __name__ == "__main__":
    unittest.main()
