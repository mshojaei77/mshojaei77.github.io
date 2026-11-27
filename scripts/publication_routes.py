"""Shared validation for routes reserved by articles, tombstones and redirects."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

SOURCE_SUFFIXES = {".md", ".markdown", ".html"}
FRONT_MATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.S)
PERMALINK = re.compile(r"^permalink:\s*['\"]?([^\s'\"]+)['\"]?\s*$", re.M)


def source_route_collisions(source_roots: Mapping[str, Path]) -> list[str]:
    """Report any public route declared by more than one source file."""
    owners: dict[str, list[str]] = defaultdict(list)
    for kind, root in source_roots.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            source = path.read_text(encoding="utf-8")
            front = FRONT_MATTER.match(source)
            if not front:
                continue
            route = PERMALINK.search(front.group(1))
            if route:
                owners[route.group(1)].append(f"{kind}:{path.as_posix()}")

    return [
        f"Conflicting source routes for {route}: {', '.join(paths)}"
        for route, paths in sorted(owners.items())
        if len(paths) > 1
    ]
