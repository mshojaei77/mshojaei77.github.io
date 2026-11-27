"""Regression tests for active post, retirement and redirect route conflicts."""
from pathlib import Path
from tempfile import TemporaryDirectory

from publication_routes import source_route_collisions


def source(path: Path, route: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: Fixture\npermalink: {route}\n---\n", encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        posts, retired, redirects = root / "_posts", root / "retired-posts", root / "redirects"
        source(posts / "one.md", "/writing/one/")
        source(posts / "two.md", "/fa/writing/two/")
        source(retired / "old.html", "/writing/old/")
        source(redirects / "legacy.html", "/legacy/")
        roots = {"article": posts, "retired": retired, "redirect": redirects}
        assert source_route_collisions(roots) == []

        source(posts / "duplicate.md", "/writing/one/")
        conflicts = source_route_collisions(roots)
        assert len(conflicts) == 1 and "/writing/one/" in conflicts[0]

        (posts / "duplicate.md").unlink()
        source(retired / "collision.html", "/writing/one/")
        conflicts = source_route_collisions(roots)
        assert len(conflicts) == 1 and "article:" in conflicts[0] and "retired:" in conflicts[0]

        (retired / "collision.html").unlink()
        source(redirects / "collision.html", "/writing/one/")
        conflicts = source_route_collisions(roots)
        assert len(conflicts) == 1 and "article:" in conflicts[0] and "redirect:" in conflicts[0]

    print("Verified unique article, retired and compatibility routes, including negative collision cases.")


if __name__ == "__main__":
    main()
