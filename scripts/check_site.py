"""Check the publication's owned routes and navigational links in a built site."""
import sys
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

root = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
routes = ["index.html", "writing/index.html", "research/index.html", "projects/index.html",
          "newsletter/index.html", "about.html", "roadmap.html", "resources.html",
          "interview.html", "market_analysis.html", "mini_roadmap.html", "open_source_ai.html",
          "vector_dbs.html", "book/index.html", "book/intro.html", "book/ch7.html",
          "404.html", "assets/css/publication.css", "feed.xml", "sitemap.xml"]
routes += [f"book/ch{i}.html" for i in range(1, 7)]
errors = [f"Missing output: {p}" for p in routes if not (root / p).is_file()]

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag in ("a", "link"):
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

indexes = ["index.html", "writing/index.html", "research/index.html", "projects/index.html",
           "newsletter/index.html", "book/index.html", "404.html"]
for name in indexes:
    path = root / name
    if not path.is_file():
        continue
    parser = Links()
    parser.feed(path.read_text(encoding="utf-8"))
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path:
            continue
        target = unquote(url.path)
        dest = (root / target.lstrip("/")) if target.startswith("/") else (path.parent / target)
        if target.endswith("/") or dest.is_dir():
            dest = dest / "index.html"
        if not dest.is_file():
            errors.append(f"{name}: unresolved local link {link}")
if errors:
    print("\n".join(sorted(set(errors))))
    sys.exit(1)
print(f"Verified {len(routes)} required outputs and navigation on {len(indexes)} indexes.")
