"""Deterministic tests for IndexNow URL change detection."""
import subprocess
import tempfile
from pathlib import Path

from indexnow import BASE, changed_urls, has_successful_notification, pages_run_id_from_url, permalink


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root: Path, message: str) -> str:
    run(root, "add", "-A")
    run(root, "commit", "-m", message)
    return run(root, "rev-parse", "HEAD")


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
    print("Verified IndexNow add/update/archive/removal URL detection without network requests.")


if __name__ == "__main__":
    main()
