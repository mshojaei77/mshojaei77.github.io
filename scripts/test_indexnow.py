"""Deterministic tests for IndexNow URL change detection."""
import subprocess
import tempfile
from pathlib import Path

from indexnow import BASE, changed_urls, permalink


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root: Path, message: str) -> str:
    run(root, "add", "-A")
    run(root, "commit", "-m", message)
    return run(root, "rev-parse", "HEAD")


def main() -> None:
    assert permalink('---\ntitle: "Example"\npermalink: /fa/writing/example/\n---\nBody') == "/fa/writing/example/"
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
