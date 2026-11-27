"""Deterministic tests for IndexNow URL change detection."""
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import indexnow
from indexnow import (
    BASE,
    changed_urls,
    checkpoint_payload,
    deployment_pair,
    force_resync_urls,
    load_checkpoint,
    pages_run_id_from_url,
    parse_checkpoint,
    permalink,
    require_accepted_status,
    requires_full_site_notification,
    save_checkpoint,
    valid_production_redirect,
)


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
    require_accepted_status(200)
    try:
        require_accepted_status(202)
    except RuntimeError as error:
        assert "checkpoint will not advance" in str(error)
    else:
        raise AssertionError("HTTP 202 must not advance the durable checkpoint")
    stamp = datetime(2026, 11, 27, tzinfo=timezone.utc)
    checkpoint = checkpoint_payload("a" * 40, "submission_accepted", [BASE + "/fa/writing/example/"], stamp)
    assert parse_checkpoint(json.dumps(checkpoint)) == checkpoint
    try:
        parse_checkpoint('{"schema_version":1,"last_acknowledged_revision":"x","known_canonical_urls":[]}')
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid persisted checkpoint revisions must fail closed")
    assert force_resync_urls(
        [BASE + "/writing/current/"],
        [BASE + "/writing/current/", BASE + "/writing/removed/"],
    ) == ([BASE + "/writing/current/"], [BASE + "/writing/removed/"])
    redirect_body = (
        b'<meta name="robots" content="noindex,nofollow">'
        b'<link rel="canonical" href="https://mshojaei77.github.io/writing/new/">'
        b'<meta http-equiv="refresh" content="0;url=/writing/new/">'
    )
    assert valid_production_redirect(200, redirect_body, {BASE + "/writing/new/"})
    assert not valid_production_redirect(200, redirect_body, {BASE + "/writing/other/"})
    assert not valid_production_redirect(200, b'<link rel="canonical" href="https://mshojaei77.github.io/writing/new/">', {BASE + "/writing/new/"})
    prior_sha, deployed_sha = "a" * 40, "b" * 40
    pages_runs = [
        {"id": 2, "run_number": 2, "workflow_id": 123, "name": "pages-build-deployment", "status": "completed", "conclusion": "success", "head_branch": "main", "head_sha": deployed_sha},
        {"id": 1, "run_number": 1, "workflow_id": 123, "name": "pages-build-deployment", "status": "completed", "conclusion": "success", "head_branch": "main", "head_sha": prior_sha},
    ]
    original_api_json = indexnow.api_json
    def fake_api_json(url: str, token: str) -> dict:
        if url.endswith("/pages/builds/latest"):
            return {"commit": deployed_sha}
        if url.endswith("/actions/workflows?per_page=100"):
            return {"workflows": [{"id": 123, "name": "pages-build-deployment", "state": "active"}]}
        if "/actions/workflows/123/runs?" in url:
            return {"workflow_runs": pages_runs}
        raise AssertionError(f"Unexpected API fixture URL: {url}")
    indexnow.api_json = fake_api_json
    try:
        assert deployment_pair("mshojaei77/mshojaei77.github.io", "", "test-token", deployed_sha) == (prior_sha, deployed_sha)
    finally:
        indexnow.api_json = original_api_json
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp)
        state_repo = parent / "repo"
        remote = parent / "state-remote.git"
        state_repo.mkdir()
        run(parent, "init", "--bare", str(remote))
        run(state_repo, "init", "-b", "main")
        run(state_repo, "config", "user.email", "indexnow-test@example.invalid")
        run(state_repo, "config", "user.name", "IndexNow test")
        (state_repo / "README.md").write_text("main revision one\n", encoding="utf-8")
        acknowledged = commit(state_repo, "main revision one")
        run(state_repo, "remote", "add", "origin", str(remote))
        run(state_repo, "push", "-u", "origin", "main")
        save_checkpoint(checkpoint_payload(acknowledged, "submission_accepted", [BASE + "/"]), state_repo)
        assert load_checkpoint(state_repo)["last_acknowledged_revision"] == acknowledged
        run(state_repo, "checkout", "main")
        (state_repo / "README.md").write_text("main revision two\n", encoding="utf-8")
        current = commit(state_repo, "main revision two")
        save_checkpoint(checkpoint_payload(current, "no_changes", [BASE + "/", BASE + "/fa/"]), state_repo)
        save_checkpoint(checkpoint_payload(acknowledged, "submission_accepted", [BASE + "/"]), state_repo)
        saved = load_checkpoint(state_repo)
        assert saved["last_acknowledged_revision"] == current, "a stale workflow must not roll back the checkpoint"
        assert saved["known_canonical_urls"] == sorted([BASE + "/", BASE + "/fa/"])
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
        discovery_head = root / "_includes" / "discovery-head.html"
        discovery_head.parent.mkdir(parents=True, exist_ok=True)
        discovery_head.write_text("SEO head v1\n", encoding="utf-8")
        fifteenth = commit(root, "add shared discovery metadata")
        discovery_head.write_text("SEO head v2\n", encoding="utf-8")
        sixteenth = commit(root, "change shared discovery metadata")
        assert requires_full_site_notification(fifteenth, sixteenth, root)
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

        # A failed or skipped notification at B is recovered by diffing from
        # the durable acknowledgement A directly to the next deployed revision C.
        acknowledged = run(root, "rev-parse", "HEAD")
        add_post(root, "en", "missed-deployment-b")
        deployment_b = commit(root, "publish content during a missed notification")
        add_post(root, "fa", "latest-deployment-c")
        deployment_c = commit(root, "publish content in the recovery deployment")
        live, deleted = changed_urls(acknowledged, deployment_c, root)
        assert BASE + "/writing/missed-deployment-b/" in live
        assert BASE + "/fa/writing/latest-deployment-c/" in live
        assert BASE + "/writing/" in live and BASE + "/fa/writing/" in live
        assert deleted == []

        roadmap = root / "roadmap.md"
        roadmap.write_text("Roadmap v1\n", encoding="utf-8")
        roadmap_added = commit(root, "add standalone roadmap fixture")
        roadmap.write_text("Roadmap v2\n", encoding="utf-8")
        roadmap_updated = commit(root, "update standalone roadmap fixture")
        live, deleted = changed_urls(roadmap_added, roadmap_updated, root)
        assert live == [BASE + "/roadmap.html"], (live, deleted)
        assert deleted == []

        add_post(root, "en", "moved-article")
        article_before_move = commit(root, "publish article before moving its route")
        moved_article = root / "_posts" / "2026-11-27-en-moved-article.md"
        moved_article.write_text(
            '---\nlang: en\ntitle: moved-article\npermalink: /writing/new-moved-article/\n---\nFixture\n',
            encoding="utf-8",
        )
        redirect_dir = root / "redirects"
        redirect_dir.mkdir(exist_ok=True)
        (redirect_dir / "moved-article.html").write_text(
            '---\npermalink: /writing/moved-article/\nredirect_to: /writing/new-moved-article/\n---\n',
            encoding="utf-8",
        )
        article_after_move = commit(root, "move article and preserve its old URL")
        live, deleted = changed_urls(article_before_move, article_after_move, root)
        assert BASE + "/writing/new-moved-article/" in live
        assert BASE + "/writing/moved-article/" in live
        assert BASE + "/writing/" in live
        assert deleted == []

        about = root / "about.md"
        about.write_text("---\npermalink: /about.html\n---\nAbout v1\n", encoding="utf-8")
        about_before_move = commit(root, "add standalone page fixture")
        about.write_text("---\npermalink: /about-profile/\n---\nAbout v2\n", encoding="utf-8")
        (redirect_dir / "about.html").write_text(
            '---\npermalink: /about.html\nredirect_to: /about-profile/\n---\n',
            encoding="utf-8",
        )
        about_after_move = commit(root, "move standalone page and preserve its old URL")
        live, deleted = changed_urls(about_before_move, about_after_move, root)
        assert live == [BASE + "/about-profile/", BASE + "/about.html"]
        assert deleted == []
    print("Verified IndexNow add/update/archive/removal URL detection without network requests.")


if __name__ == "__main__":
    main()
