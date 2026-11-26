"""Check the publication's owned routes and navigational links in a built site."""
import sys
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

root = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
routes = ["index.html", "writing/index.html", "projects/index.html",
          "about.html", "roadmap.html", "resources.html",
          "interview.html", "market_analysis.html", "mini_roadmap.html", "open_source_ai.html",
          "vector_dbs.html", "writing/document-parsing/index.html",
          "404.html", "assets/css/publication.css", "feed.xml", "sitemap.xml"]
fa_routes = ["fa/index.html"] + [f"fa/{part}/index.html" for part in ("writing", "roadmap", "about")]
fa_routes += [f"fa/{part}/index.html" for part in ("open-source-ai", "vector-databases", "market-analysis", "resources", "interview", "mini-roadmap")]
fa_routes += ["fa/writing/document-parsing/index.html", "fa/writing/market-analysis/index.html", "fa/writing/vector-databases/index.html"]
routes += ["writing/market-analysis/index.html", "writing/vector-databases/index.html"]
routes += fa_routes + ["assets/fonts/Vazirmatn-variable.woff2", "assets/fonts/OFL.txt", "assets/img/social-card.png", "assets/js/theme.js", "assets/js/document-parsing.js"]
routes += ["materials/Data_Preparation/index.html","fa/materials/Data_Preparation/index.html","materials/Embeddings/index.html","fa/materials/Embeddings/index.html","materials/Intro/index.html","fa/materials/Intro/index.html","materials/Neural_Networks/index.html","fa/materials/Neural_Networks/index.html","materials/Tokenization/index.html","fa/materials/Tokenization/index.html","materials/Traditional_LMs/index.html","fa/materials/Traditional_LMs/index.html","materials/Transformers/index.html","fa/materials/Transformers/index.html"]
routes += ["assets/examples/document-parsing/" + name for name in ("parse_pdf.py", "make_demo.py", "requirements.txt")]
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

indexes = ["index.html", "writing/index.html", "projects/index.html",
           "newsletter/index.html", "404.html"]
indexes += [p for p in fa_routes if p in ("fa/index.html", "fa/writing/index.html", "fa/roadmap/index.html", "fa/about/index.html") or p.startswith("fa/writing/")]
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

pairs = [("/", "/fa/")] + [(f"/{p}/", f"/fa/{p}/") for p in ("writing",)]
pairs += [("/roadmap.html", "/fa/roadmap/"), ("/about.html", "/fa/about/"), ("/writing/open-source-ai/", "/fa/writing/open-source-ai/"), ("/writing/vector-databases/", "/fa/writing/vector-databases/"), ("/writing/resources/", "/fa/writing/resources/"), ("/writing/mini-roadmap/", "/fa/writing/mini-roadmap/"), ("/writing/interview/", "/fa/writing/interview/"), ("/writing/market-analysis/", "/fa/writing/market-analysis/")]
pairs.append(("/writing/document-parsing/", "/fa/writing/document-parsing/"))
pairs += [["/writing/data-preparation/","/fa/writing/data-preparation/"],["/writing/embeddings/","/fa/writing/embeddings/"],["/writing/intro-to-llms/","/fa/writing/intro-to-llms/"],["/writing/neural-networks/","/fa/writing/neural-networks/"],["/writing/tokenization/","/fa/writing/tokenization/"],["/writing/traditional-language-models/","/fa/writing/traditional-language-models/"],["/writing/transformers/","/fa/writing/transformers/"]]
def inspect_route(route):
    p = root / route.lstrip("/")
    if route.endswith("/"):
        p /= "index.html"
    parser = Locale()
    if p.is_file():
        parser.feed(p.read_text(encoding="utf-8"))
    return parser
def retired(route):
    page = root / route.lstrip("/") / "index.html"
    return page.is_file() and "data-retired-post" in page.read_text(encoding="utf-8")

for en, fa in pairs:
    if retired(en) or retired(fa):
        for route, other, fallback in ((en, fa, '/fa/'), (fa, en, '/')):
            info = inspect_route(route)
            if info.alternates or fallback not in info.anchors:
                errors.append(f'{route}: removed translation still advertised')
        continue
    f, g = inspect_route(en), inspect_route(fa)
    if g.html.get("lang") != "fa" or g.html.get("dir") != "rtl":
        errors.append(f"{fa}: missing Persian RTL metadata")
    if en not in g.anchors or fa not in f.anchors:
        errors.append(f"{en} / {fa}: missing reciprocal language navigation")
    if fa != "/fa/writing/document-parsing/" and (g.alternates.get("en") != en or g.alternates.get("fa") != fa):
        errors.append(f"{fa}: missing language alternates")
    if True:
        if f.html.get("lang") != "en" or f.html.get("dir") != "ltr":
            errors.append(f"{en}: missing English LTR metadata")
        if fa != "/fa/writing/document-parsing/" and (f.alternates.get("fa") != fa or f.alternates.get("en") != en):
            errors.append(f"{en}: missing language alternates")
article_routes = {"fa": "/fa/writing/document-parsing/", "en": "/writing/document-parsing/"}
def listed_xml(name, route):
    import xml.etree.ElementTree as ET
    tree = ET.parse(root / name)
    ns = "{http://www.w3.org/2005/Atom}"
    if name == "feed.xml":
        urls = [link.get("href", "") for entry in tree.findall(ns + "entry") for link in entry.findall(ns + "link") if link.get("rel", "alternate") == "alternate"]
    else:
        urls = [node.text or "" for node in tree.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return any(urlsplit(url).path == route for url in urls)


for lang, own in article_routes.items():
    other = article_routes["en" if lang == "fa" else "fa"]
    prefix = "fa/" if lang == "fa" else ""
    for path in (prefix + "index.html", prefix + "writing/index.html"):
        parser = Links()
        parser.feed((root / path).read_text(encoding="utf-8"))
        if (path.endswith("writing/index.html") and (own in parser.links) == retired(own)) or (retired(own) and own in parser.links) or other in parser.links:
            errors.append(f"{path}: article language isolation failed")
    for path in ("feed.xml", "sitemap.xml"):
        if (root / path).is_file() and (path != "feed.xml" or retired(own)) and (listed_xml(path, own) == retired(own)):
            errors.append(f"{path}: missing {own}")
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
# Retired publication surfaces must not appear anywhere in the generated site.
import re
for path in root.rglob("*"):
    if not path.is_file() or path.suffix not in (".html", ".xml"):
        continue
    html = path.read_text(encoding="utf-8")
    if re.search(r'https?://(?:[\w-]+\.)*medium\.com(?:[/:?#]|$)', html, re.I):
        errors.append(f"{path.relative_to(root)}: Medium URL remains")
    if re.search(r'(?:(?:https?://mshojaei77\.github\.io)?/(?:fa/)?newsletter/|https?://(?:www\.)?linkedin\.com/newsletters/)', html, re.I):
        errors.append(f"{path.relative_to(root)}: retired newsletter URL remains")
for path in ("newsletter/index.html", "fa/newsletter/index.html"):
    if (root / path).exists():
        errors.append(f"Retired newsletter published: {path}")
# V6 replaces the earlier Persian tutorial at its canonical public URL.
guide_route = "/fa/writing/document-parsing/"
obsolete_route = "/fa/writing/document-parsing-engineering/"
guide_file = root / guide_route.lstrip("/") / "index.html"
class Guide(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.refs = []
        self.figures = 0
        self.svgs = 0
        self.in_article = False
        self.h1 = 0
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "article" and "article-content" in a.get("class", "").split():
            self.in_article = True
        if not self.in_article:
            return
        if a.get("id"):
            self.ids.append(a["id"])
        if tag == "a" and a.get("href", "").startswith("#ref-"):
            self.refs.append(a["href"][1:])
        if tag == "figure":
            self.figures += 1
        if tag == "svg":
            self.svgs += 1
        if tag == "h1":
            self.h1 += 1
    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False
if not guide_file.is_file():
    errors.append(f"Missing output: {guide_route}")
elif not retired(guide_route):
    guide = Guide()
    guide.feed(guide_file.read_text(encoding="utf-8"))
    locale = inspect_route(guide_route)
    if locale.html.get("lang") != "fa" or locale.html.get("dir") != "rtl" or locale.alternates:
        errors.append("Engineering guide locale/translation metadata is invalid")
    if guide.h1 != 1:
        errors.append(f"Engineering guide content missing: {guide.figures} figures, {guide.svgs} SVGs, {guide.h1} titles")
    if len(guide.ids) != len(set(guide.ids)) or set(guide.refs) - set(guide.ids):
        errors.append("Engineering guide contains duplicate IDs or unresolved references")
for name in ("fa/index.html", "fa/writing/index.html", "index.html", "writing/index.html"):
    parser = Links()
    parser.feed((root / name).read_text(encoding="utf-8"))
    expected = name == "fa/writing/index.html" and not retired(guide_route)
    count = parser.links.count(guide_route)
    if (expected and count != 1) or (retired(guide_route) and count) or (name == "fa/index.html" and count > 1) or (not name.startswith("fa/") and count) or obsolete_route in parser.links:
        errors.append(f"{name}: canonical article listing must be unique")
for name in ("feed.xml", "sitemap.xml"):
    html = (root / name).read_text(encoding="utf-8")
    if ((name != "feed.xml" or retired(guide_route)) and listed_xml(name, guide_route) == retired(guide_route)) or listed_xml(name, obsolete_route):
        errors.append(f"{name}: wrong canonical article listing")
redirect = root / obsolete_route.lstrip("/") / "index.html"
if not redirect.is_file() or f"url={guide_route}" not in redirect.read_text(encoding="utf-8"):
    errors.append("Missing redirect from accidentally published duplicate URL")
if inspect_route("/writing/document-parsing/").alternates:
    errors.append("Older English tutorial incorrectly advertised as translation of v6")
# Publication Studio must be self-hosted, excluded from indexing, and keep source files intact.
admin = root / "admin/index.html"
if not admin.is_file():
    errors.append("Studio entry point missing")
else:
    html = admin.read_text(encoding="utf-8")
    for marker in ('/admin/studio.js', 'noindex,nofollow,noarchive', 'sandbox=""', "connect-src https://api.github.com"):
        if marker not in html:
            errors.append(f"Studio security/boot marker missing: {marker}")
    if "unpkg.com" in html or (root / "admin/config.yml").exists():
        errors.append("Retired Sveltia CMS remains deployed")
    for name in ("studio.js", "studio-core.js", "studio.css", "preview-worker.js"):
        if not (root / "admin" / name).is_file():
            errors.append(f"Studio asset missing: {name}")
    if "/admin/" in (root / "sitemap.xml").read_text(encoding="utf-8"):
        errors.append("Admin must not appear in sitemap")
    if "Disallow: /admin/" not in (root / "robots.txt").read_text(encoding="utf-8"):
        errors.append("Admin crawler exclusion missing")
# Unique canonical post routes, including newly created Studio posts.
from collections import Counter
post_routes = []
for path in Path("_posts").iterdir():
    if path.suffix not in (".md", ".markdown", ".html"):
        continue
    text = path.read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.S)
    route = re.search(r"^permalink:\s*(.*?)\s*$", front.group(1), re.M) if front else None
    if not route:
        errors.append(f"Missing explicit permalink: {path}")
    else:
        post_routes.append(route.group(1).strip("'\\\""))
for route, count in Counter(post_routes).items():
    if count > 1:
        errors.append(f"Duplicate post permalink: {route}")
# All explicit removals must retain an unindexed notice, without a post or archive card.
for path in Path("retired-posts").glob("*.html"):
    source = path.read_text(encoding="utf-8")
    match = re.search(r"^permalink:\s*(/[^\s]+/)\s*$", source, re.M)
    if not match:
        errors.append(f"Invalid removal notice: {path}")
        continue
    route = match.group(1)
    if route in post_routes:
        errors.append(f"Published and removed simultaneously: {route}")
    if not retired(route):
        errors.append(f"Missing removal notice: {route}")
        continue
    html = (root / route.lstrip("/") / "index.html").read_text(encoding="utf-8")
    if 'content="noindex,nofollow"' not in html:
        errors.append(f"Removal notice is indexable: {route}")
    for name in ("index.html", "fa/index.html", "writing/index.html", "fa/writing/index.html", "feed.xml", "sitemap.xml"):
        output = root / name
        if not output.is_file():
            continue
        if name.endswith(".xml"):
            found = listed_xml(name, route)
        else:
            parser = Links()
            parser.feed(output.read_text(encoding="utf-8"))
            found = route in parser.links
        if found:
            errors.append(f"Removed route still listed in {name}: {route}")

# Research is removed; migrated articles are real posts and legacy pages are redirects.
for name in ("research/index.html", "fa/research/index.html"):
    if (root / name).exists():
        errors.append(f"Removed research section still built: {name}")
for name in ("index.html", "fa/index.html", "writing/index.html", "fa/writing/index.html"):
    parser = Links()
    parser.feed((root / name).read_text(encoding="utf-8"))
    if any(link in ("/research/", "/fa/research/") for link in parser.links):
        errors.append(f"Research navigation remains: {name}")
for old, route, archive in [["fa/market-analysis/index.html","/fa/writing/market-analysis/","fa/writing/index.html"],["market_analysis.html","/writing/market-analysis/","writing/index.html"],["fa/vector-databases/index.html","/fa/writing/vector-databases/","fa/writing/index.html"],["vector_dbs.html","/writing/vector-databases/","writing/index.html"]]:
    legacy = root / old
    if not legacy.is_file() or f"url={route}" not in legacy.read_text(encoding="utf-8"):
        errors.append(f"Legacy redirect missing: {old}")
    parser = Links()
    parser.feed((root / archive).read_text(encoding="utf-8"))
    if parser.links.count(route) != (0 if retired(route) else 1):
        errors.append(f"Wrong migrated article count: {route}")
    if listed_xml("sitemap.xml", route) == retired(route):
        errors.append(f"Wrong migrated sitemap state: {route}")
    if listed_xml("sitemap.xml", "/" + old.replace("index.html", "")):
        errors.append(f"Legacy redirect indexed: {old}")

# All articles must be regular posts under their language's writing namespace.
for path in Path("_posts").iterdir():
    if path.suffix not in (".md", ".markdown", ".html"):
        continue
    text = path.read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---(?:\n|$)", text, re.S)
    if not front:
        errors.append(f"Invalid front matter: {path}")
        continue
    fields = front.group(1)
    match = re.search(r"^lang:\s*(fa|en)\s*$", fields, re.M)
    route_match = re.search(r"^permalink:\s*(/[^\s]+/)\s*$", fields, re.M)
    if not match or not route_match:
        errors.append(f"Missing post language or permalink: {path}")
        continue
    lang, route = match.group(1), route_match.group(1)
    prefix = "/fa/writing/" if lang == "fa" else "/writing/"
    if not re.fullmatch(re.escape(prefix) + r"[a-z0-9]+(?:-[a-z0-9]+)*/", route):
        errors.append(f"Post outside canonical writing namespace: {path}: {route}")
    legacy = {"2026-11-26-document-parsing.md": "fa", "2026-11-26-document-parsing-en.html": "en"}
    filename_lang = re.match(r"^\d{4}-\d{2}-\d{2}-(fa|en)-([a-z0-9]+(?:-[a-z0-9]+)*)\.(md|markdown|html)$", path.name)
    slug = "document-parsing" if path.name in legacy else filename_lang.group(2) if filename_lang else ""
    if route != prefix + slug + "/":
        errors.append(f"Post filename slug does not match permalink: {path}")
    if legacy.get(path.name, filename_lang.group(1) if filename_lang else None) != lang:
        errors.append(f"Post filename language does not match metadata: {path}")
    if not (root / route.lstrip("/") / "index.html").is_file():
        errors.append(f"Post not built: {route}")
    for archive_lang in ("fa", "en"):
        archive = root / ("fa/writing/index.html" if archive_lang == "fa" else "writing/index.html")
        parser = Links()
        parser.feed(archive.read_text(encoding="utf-8"))
        if parser.links.count(route) != int(archive_lang == lang):
            errors.append(f"Post missing, duplicate or wrong language in archive: {route}")
    if not listed_xml("sitemap.xml", route):
        errors.append(f"Post missing in sitemap: {route}")

# Home, roadmap and about are the only standalone editorial pages.
allowed_pages = {"index.html", "fa/index.html", "roadmap.html", "fa/roadmap/index.html",
                 "about.html", "fa/about/index.html", "writing/index.html", "fa/writing/index.html"}
for page in root.rglob("*.html"):
    name = page.relative_to(root).as_posix()
    html = page.read_text(encoding="utf-8")
    if 'data-layout-version="publication-4"' not in html:
        continue
    if name in allowed_pages or name == "404.html":
        continue
    if re.fullmatch(r"(?:fa/)?writing/[a-z0-9]+(?:-[a-z0-9]+)*/index\.html", name):
        continue
    errors.append(f"Editorial page outside writing: {name}")
for path in Path("redirects").glob("*.html"):
    source = path.read_text(encoding="utf-8")
    old = re.search(r"^permalink:\s*(/\S+)\s*$", source, re.M)
    new = re.search(r"^redirect_to:\s*(\S+)\s*$", source, re.M)
    if not old or not new:
        errors.append(f"Invalid compatibility redirect: {path}")
        continue
    route, target = old.group(1), new.group(1)
    page = root / (route.lstrip("/") + ("index.html" if route.endswith("/") else ""))
    if not page.is_file() or f"url={target}" not in page.read_text(encoding="utf-8") or listed_xml("sitemap.xml", route):
        errors.append(f"Missing or indexed legacy redirect: {route}")
    if target.startswith("/") and not (root / (target.lstrip("/") + ("index.html" if target.endswith("/") else ""))).is_file():
        errors.append(f"Broken redirect target: {target}")
if errors:
    print("\n".join(sorted(set(errors))))
    sys.exit(1)
print(f"Verified {len(routes)} required outputs and navigation on {len(indexes)} indexes.")
