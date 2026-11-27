"""Check actual generated SEO and optional discovery configuration."""
import json
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import re

root = Path(sys.argv[1])
settings = json.loads(Path("_data/discovery.json").read_text(encoding="utf-8"))
host = "https://mshojaei77.github.io"
class Head(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_head = False
        self.in_title = False
        self.json = None
        self.schemas = []
        self.meta = {}
        self.canonicals = []
        self.title = ""
        self.scripts = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        if tag == "title":
            self.in_title = True
        if tag == "meta":
            key = a.get("name") or a.get("property")
            if key:
                self.meta.setdefault(key, []).append(a.get("content", ""))
        if tag == "link" and a.get("rel") == "canonical":
            self.canonicals.append(a.get("href"))
        if tag == "script":
            self.scripts.append(a)
            if a.get("type") == "application/ld+json":
                self.json = ""
    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.json is not None:
            self.json += data
    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "head":
            self.in_head = False
        if tag == "script" and self.json is not None:
            self.schemas.append(json.loads(self.json))
            self.json = None
def parse(path):
    p = Head()
    p.feed(path.read_text(encoding="utf-8"))
    return p
def file_for(route):
    return root / (route.lstrip("/") + ("index.html" if route.endswith("/") else ""))
def analytics(p):
    return [s for s in p.scripts if s.get("data-analytics-provider") == "goatcounter" or "gc.zgo.at" in s.get("src","")]
def values(p, key, expected):
    assert p.meta.get(key,[]) == ([expected] if expected else []), (key,p.meta.get(key))
urls = [e.text for e in ET.parse(root/"sitemap.xml").iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
routes = {urlsplit(u).path for u in urls}
articles = sorted(r for r in routes if re.fullmatch(r"/(?:fa/)?writing/[^/]+/",r))
public = {"/","/fa/","/writing/","/fa/writing/","/roadmap.html","/fa/roadmap/","/about.html","/fa/about/"} | set(articles)
for route in public:
    p = parse(file_for(route))
    assert p.title.strip(), route
    assert len(p.meta.get("description",[])) == 1 and p.meta["description"][0].strip(), route
    assert p.canonicals == [host+route], (route,p.canonicals)
    assert p.meta.get("og:title") and p.meta.get("og:description"), route
    assert p.meta.get("og:image"), route
    expected = settings["goatcounter_site"]
    hits = analytics(p)
    assert len(hits) == bool(expected), (route,hits)
    if expected:
        assert hits[0]["data-goatcounter-site"] == expected
        assert hits[0]["data-production-host"] == "mshojaei77.github.io"
    for key, name in (("google-site-verification","google_verification"),("msvalidate.01","bing_verification")):
        values(p,key,settings[name] if route in {"/","/fa/"} else "")
    if route in articles:
        blogs = [s for s in p.schemas if s.get("@type") == "BlogPosting"]
        assert len(blogs) == 1, (route,p.schemas)
        blog = blogs[0]
        assert blog.get("headline") and blog.get("author",{}).get("name") and blog.get("datePublished"), route
        assert blog.get("url") == host+route, (route,blog)
for route in ["/admin/","/projects/","/fa/market-analysis/","/writing/document-parsing/","/fa/writing/document-parsing/"]:
    if route not in public and file_for(route).is_file():
        assert not analytics(parse(file_for(route))), route
assert "/admin/" not in routes
print(f"Verified SEO, canonical URLs and analytics isolation for {len(articles)} live articles; configured={bool(settings['goatcounter_site'])}.")

# Preserve Google's exact root verification file through every build.
verification = "google483b45b4cf9637cf.html"
assert (root / verification).read_bytes() == b"google-site-verification: google483b45b4cf9637cf.html"
assert "/" + verification not in routes, "Verification file should not be in sitemap"
print("Verified exact Google ownership file.")

# Preserve the exact Bing ownership file in the published root.
assert (root / "BingSiteAuth.xml").read_bytes() == "<?xml version=\"1.0\"?>\n<users>\n\t<user>2AA6B158D3D26282FB332CC19F0505F7</user>\n</users>".encode("utf-8")
assert "/BingSiteAuth.xml" not in routes
print("Verified exact Bing ownership file.")
