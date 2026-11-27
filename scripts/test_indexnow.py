"""Deterministic tests for IndexNow URL change detection."""
import subprocess
import tempfile
from pathlib import Path

from indexnow import BASE, changed_urls, has_successful_notification, pages_run_id_from_url, permalink, requires_full_site_notification


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root: Path, message: str) -> str:
    run(root, "add", "-A")
    run(root, "commit", "-m", message)
    return run(root, "rev-parse", "HEAD")


def add_post(root: Path, language: str, slug: str) -> None:
    path = root / "_posts" / f"2026-11-27-{language}-{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    route = f"/fa/writing/{slug}/" if language == "fa" else f"/writing/{slug}/"
    path.write_text(
        f"---\nlang: {language}\ntitle: {slug}\npermalink: {route}\n---\nFixture\n",
        encoding="utf-8",
    )


def main() -> None:
    assert permalink('---\ntitle: "Example"\npermalink: /fa/writing/example/\n---\nBody') == "/fa/writing/example/"
    assert pages_run_id_from_url("https://github.com/mshojaei77/mshojaei77.github.io/actions/runs/45925179377/job/140264859010") == "45925179377"
    try:
        pages_run_id_from_url("https://example.invalid/actions/runs/123")
    except RuntimeError:
        pass
    else:
        raise AssertionError("untrusted Pages log URLs must be rejected")
    sha = "a" * 40
    prior_runs = [
        {"id": 1, "name": "Notify IndexNow after deployment", "head_sha": sha, "conclusion": "success"},
        {"id": 2, "name": "Notify IndexNow after deployment", "head_sha": "b" * 40, "conclusion": "success"},
        {"id": 3, "name": "Production smoke check", "head_sha": sha, "conclusion": "success"},
    ]
    assert has_successful_notification(prior_runs, sha, "4")
    assert not has_successful_notification(prior_runs, sha, "1"), "the current run is not a previous notification"
    assert not has_successful_notification(prior_runs, "c" * 40, "4")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        run(root, "init", "-b", "main")
        run(root, "config", "user.email", "indexnow-test@example.invalid")
        run(root, "config", "user.name", "IndexNow test")
        (root / "_posts").mkdir()
        article = root / "_posts" / "2026-11-27-fa-example.md"
        article.write_text('---\nlang: fa\ntitle: Example\npermalink: /fa/writing/example/\n---\nDraft body\n', encoding="utf-8")
        first = commit(root, "publish article")
        assert changed_urls(first, first, root) == ([], [])
        article.write_text('---\nlang: fa\ntitle: Example\npermalink: /fa/writing/example/\n---\nUpdated body\n', encoding="utf-8")
        second = commit(root, "update article")
        live, deleted = changed_urls(first, second, root)
        assert live == [BASE + "/fa/writing/", BASE + "/fa/writing/example/"]
        assert deleted == []
        article.unlink()
        retired = root / "retired-posts" / "fa-example.html"
        retired.parent.mkdir()
        retired.write_text('---\nlang: fa\npermalink: /fa/writing/example/\nretired_post: true\n---\nRetired\n', encoding="utf-8")
        third = commit(root, "retire article")
        live, deleted = changed_urls(second, third, root)
        assert live == [BASE + "/fa/writing/"]
        assert deleted == [BASE + "/fa/writing/example/"]
        add_post(root, "en", "roadmap")
        add_post(root, "fa", "roadmap")
        fourth = commit(root, "publish both language fixtures")
        layout = root / "_layouts" / "article.html"
        layout.parent.mkdir()
        layout.write_text("article v1\n", encoding="utf-8")
        fifth = commit(root, "add article layout")
        layout.write_text("article v2\n", encoding="utf-8")
        sixth = commit(root, "change shared article layout")
        assert not requires_full_site_notification(fifth, sixth, root)
        live, deleted = changed_urls(fifth, sixth, root)
        assert live == [
            BASE + "/fa/writing/",
            BASE + "/fa/writing/roadmap/",
            BASE + "/writing/",
            BASE + "/writing/roadmap/",
        ]
        assert deleted == []
        topics = root / "_data" / "topics.json"
        topics.parent.mkdir()
        topics.write_text("{}\n", encoding="utf-8")
        ninth = commit(root, "add topic map")
        topics.write_text('{"posts": {"roadmap": ["systems"]}}\n', encoding="utf-8")
        tenth = commit(root, "update shared topic map")
        live, deleted = changed_urls(ninth, tenth, root)
        assert live == [
            BASE + "/",
            BASE + "/fa/",
            BASE + "/fa/writing/",
            BASE + "/fa/writing/roadmap/",
            BASE + "/writing/",
            BASE + "/writing/roadmap/",
        ]
        assert deleted == []
        author = root / "_data" / "authors.json"
        author.write_text('{"name":"v1"}\n', encoding="utf-8")
        eleventh = commit(root, "add author profile data")
        author.write_text('{"name":"v2"}\n', encoding="utf-8")
        twelfth = commit(root, "update author profile data")
        live, deleted = changed_urls(eleventh, twelfth, root)
        assert live == [
            BASE + "/",
            BASE + "/about.html",
            BASE + "/fa/",
            BASE + "/fa/about/",
            BASE + "/fa/writing/roadmap/",
            BASE + "/writing/roadmap/",
        ], (live, deleted)
        assert deleted == []
        language_data = root / "_data" / "languages.yml"
        language_data.write_text("- en: /\n  fa: /fa/\n", encoding="utf-8")
        thirteenth = commit(root, "add language map")
        language_data.write_text("- en: /\n  fa: /fa/\n  x-default: /\n", encoding="utf-8")
        fourteenth = commit(root, "change language map")
        assert requires_full_site_notification(thirteenth, fourteenth, root)
        add_post(root, "fa", "retire-with-template")
        site_layout = root / "_layouts" / "publication.html"
        site_layout.write_text("publication v1\n", encoding="utf-8")
        seventh = commit(root, "add publication layout")
        retired_post = root / "_posts" / "2026-11-27-fa-retire-with-template.md"
        retired_post.unlink()
        retirement = root / "retired-posts" / "fa-retire-with-template.html"
        retirement.parent.mkdir(exist_ok=True)
        retirement.write_text(
            "---\nlang: fa\npermalink: /fa/writing/retire-with-template/\nretired_post: true\n---\nRetired\n",
            encoding="utf-8",
        )
        site_layout.write_text("publication v2\n", encoding="utf-8")
        eighth = commit(root, "change site-wide publication layout")
        assert requires_full_site_notification(seventh, eighth, root)
        live, deleted = changed_urls(seventh, eighth, root)
        assert BASE + "/fa/writing/" in live
        assert deleted == [BASE + "/fa/writing/retire-with-template/"]
    print("Verified IndexNow add/update/archive/removal URL detection without network requests.")


if __name__ == "__main__":
    main()
