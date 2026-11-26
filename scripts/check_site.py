"""Check the publication's owned routes and navigational links in a built site."""
import sys
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

root = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
routes = ["index.html", "writing/index.html", "research/index.html", "projects/index.html",
          "newsletter/index.html", "about.html", "roadmap.html", "resources.html",
          "interview.html", "market_analysis.html", "mini_roadmap.html", "open_source_ai.html",
          "vector_dbs.html",
          "404.html", "assets/css/publication.css", "feed.xml", "sitemap.xml"]
fa_routes = ["fa/index.html"] + [f"fa/{part}/index.html" for part in ("writing", "research", "projects", "newsletter", "roadmap", "about")]
fa_routes += [f"fa/{part}/index.html" for part in ("open-source-ai", "vector-databases", "market-analysis", "resources", "interview", "mini-roadmap")]
fa_routes += ["fa/writing/document-parsing/index.html"]
routes += fa_routes + ["assets/fonts/Vazirmatn-variable.woff2", "assets/fonts/OFL.txt", "assets/img/social-card.png", "assets/js/theme.js", "assets/js/document-parsing.js"]
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
           "newsletter/index.html", "404.html"]
indexes += fa_routes
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
# Validate language metadata and reciprocal switches for publication pairs.
class Locale(HTMLParser):
    def __init__(self):
        super().__init__()
        self.html = {}
        self.alternates = {}
        self.anchors = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.html = a
        if tag == "link" and a.get("rel") == "alternate" and a.get("hreflang"):
            self.alternates[a["hreflang"]] = urlsplit(a.get("href", "")).path
        if tag == "a":
            self.anchors.append(a.get("href"))

pairs = [("/", "/fa/")] + [(f"/{p}/", f"/fa/{p}/") for p in ("writing", "research", "newsletter")]
pairs += [("/roadmap.html", "/fa/roadmap/"), ("/about.html", "/fa/about/"), ("/open_source_ai.html", "/fa/open-source-ai/"), ("/vector_dbs.html", "/fa/vector-databases/"), ("/resources.html", "/fa/resources/"), ("/mini_roadmap.html", "/fa/mini-roadmap/"), ("/interview.html", "/fa/interview/")]
def inspect_route(route):
    p = root / route.lstrip("/")
    if route.endswith("/"):
        p /= "index.html"
    parser = Locale()
    if p.is_file():
        parser.feed(p.read_text(encoding="utf-8"))
    return parser
for en, fa in pairs:
    f, g = inspect_route(en), inspect_route(fa)
    if g.html.get("lang") != "fa" or g.html.get("dir") != "rtl":
        errors.append(f"{fa}: missing Persian RTL metadata")
    if en not in g.anchors or fa not in f.anchors:
        errors.append(f"{en} / {fa}: missing reciprocal language navigation")
    if en not in ("/roadmap.html",) and (g.alternates.get("en") != en or g.alternates.get("fa") != fa):
        errors.append(f"{fa}: missing language alternates")
    if en not in ("/roadmap.html",):
        if f.html.get("lang") != "en" or f.html.get("dir") != "ltr":
            errors.append(f"{en}: missing English LTR metadata")
        if f.alternates.get("fa") != fa or f.alternates.get("en") != en:
            errors.append(f"{en}: missing language alternates")
article_route = "/fa/writing/document-parsing/"
for path in ("fa/index.html", "fa/writing/index.html"):
    parser = Links()
    if (root / path).is_file():
        parser.feed((root / path).read_text(encoding="utf-8"))
        if article_route not in parser.links:
            errors.append(f"{path}: missing published article")
for path in ("feed.xml", "sitemap.xml"):
    if (root / path).is_file() and article_route not in (root / path).read_text(encoding="utf-8"):
        errors.append(f"{path}: missing published article")
for path in ("index.html", "writing/index.html"):
    parser = Links()
    parser.feed((root / path).read_text(encoding="utf-8"))
    if article_route in parser.links:
        errors.append(f"{path}: Persian-only article leaked into English archive")
for path in ("index.html", "fa/index.html", "writing/index.html", "fa/writing/index.html"):
    html = (root / path).read_text(encoding="utf-8")
    if any(x in html for x in ("Living book", "Selected open-source work", "کتاب در حال نگارش", "پروژه‌های متن‌باز منتخب")):
        errors.append(f"{path}: retired content advertised")
    for link in ("https://t.me/LLMEngineers", "https://t.me/AI_LLMs"):
        if link not in html:
            errors.append(f"{path}: missing {link}")
for path in ("book/index.html", "fa/book/index.html", "book/ch1.html"):
    if (root / path).exists():
        errors.append(f"Retired book published: {path}")
if errors:
    print("\n".join(sorted(set(errors))))
    sys.exit(1)
print(f"Verified {len(routes)} required outputs and navigation on {len(indexes)} indexes.")
