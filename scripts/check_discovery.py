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
topics = json.loads(Path("_data/topics.json").read_text(encoding="utf-8"))
author_profiles = json.loads(Path("_data/authors.json").read_text(encoding="utf-8"))
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
        self.links = []
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
        if tag == "link":
            self.links.append(a)
            if a.get("rel") == "canonical":
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
class RelatedLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.links = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "section" and "related-articles" in values.get("class", "").split():
            self.inside = True
        elif self.inside and tag == "a":
            self.links.append((values.get("href", ""), values.get("hreflang", "")))
    def handle_endtag(self, tag):
        if tag == "section" and self.inside:
            self.inside = False
class AuthorLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "author" in values.get("rel", "").split():
            self.hrefs.append(values.get("href", ""))
class ArticleByline(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.active_time = None
        self.times = []
        self.text = ""
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "p" and "article-byline" in values.get("class", "").split():
            self.inside = True
        elif self.inside and tag == "time":
            self.active_time = {"datetime": values.get("datetime", ""), "text": ""}
            self.times.append(self.active_time)
    def handle_data(self, data):
        if self.inside:
            self.text += data
            if self.active_time is not None:
                self.active_time["text"] += data
    def handle_endtag(self, tag):
        if tag == "time":
            self.active_time = None
        elif tag == "p" and self.inside:
            self.inside = False
def parse(path):
    p = Head()
    p.feed(path.read_text(encoding="utf-8"))
    return p
def schema_nodes(p):
    nodes = []
    for schema in p.schemas:
        nodes.append(schema)
        if isinstance(schema.get("@graph"), list):
            nodes.extend(node for node in schema["@graph"] if isinstance(node, dict))
    return nodes
def file_for(route):
    return root / (route.lstrip("/") + ("index.html" if route.endswith("/") else ""))
def analytics(p):
    return [s for s in p.scripts if s.get("data-analytics-provider") == "goatcounter" or "gc.zgo.at" in s.get("src","")]
def values(p, key, expected):
    assert p.meta.get(key,[]) == ([expected] if expected else []), (key,p.meta.get(key))
sitemap_tree = ET.parse(root/"sitemap.xml")
urls = [e.text for e in sitemap_tree.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
sitemap_lastmod = {
    (entry.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text or ""): (entry.findtext("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod") or "")
    for entry in sitemap_tree.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}url")
    if entry.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") is not None
}
routes = {urlsplit(u).path for u in urls}
articles = sorted(r for r in routes if re.fullmatch(r"/(?:fa/)?writing/[^/]+/",r))
public = {"/","/fa/","/writing/","/fa/writing/","/roadmap.html","/fa/roadmap/","/about.html","/fa/about/","/privacy/","/fa/privacy/"} | set(articles)
seen_page_titles = {}
seen_page_descriptions = {}
for route in public:
    p = parse(file_for(route))
    assert p.title.strip(), route
    assert len(p.meta.get("description",[])) == 1 and p.meta["description"][0].strip(), route
    assert p.canonicals == [host+route], (route,p.canonicals)
    assert p.meta.get("og:title") and p.meta.get("og:description"), route
    assert p.meta.get("og:image"), route
    icons = [link for link in p.links if "icon" in (link.get("rel", "").split())]
    assert icons == [{"rel": "icon", "type": "image/png", "sizes": "512x512", "href": "/assets/img/favicon.png"}], (route, icons)
    title = p.meta["og:title"][0].strip()
    description = p.meta["description"][0].strip()
    assert description.casefold() != title.casefold(), f"Page description repeats its title: {route}"
    assert title not in seen_page_titles, f"Duplicate page title: {title} ({route}, {seen_page_titles.get(title)})"
    assert description not in seen_page_descriptions, f"Duplicate page description: {route}, {seen_page_descriptions.get(description)}"
    seen_page_titles[title] = route
    seen_page_descriptions[description] = route
    expected = settings["goatcounter_site"]
    hits = analytics(p)
    assert len(hits) == bool(expected), (route,hits)
    if expected:
        assert hits[0]["data-goatcounter-site"] == expected
        assert hits[0]["data-production-host"] == "mshojaei77.github.io"
    for key, name in (("google-site-verification","google_verification"),("msvalidate.01","bing_verification")):
        values(p,key,settings[name] if route in {"/","/fa/"} else "")
    nodes = schema_nodes(p)
    profile_pages = [node for node in nodes if node.get("@type") == "ProfilePage"]
    if route in {"/about.html", "/fa/about/"}:
        assert len(profile_pages) == 1, (route, nodes)
        profile = profile_pages[0]
        person = profile.get("mainEntity", {})
        language = "fa" if route.startswith("/fa/") else "en"
        identity = author_profiles["محمد شجاعی" if language == "fa" else "Mohammad Shojaei"]
        assert profile.get("url") == host + route and profile.get("inLanguage") == language, route
        assert person.get("@type") == "Person" and person.get("@id") == "https://mshojaei77.github.io/#person", route
        assert person.get("name") == identity["name"] and person.get("alternateName") == identity["alternate_name"], route
        assert person.get("url") == identity["url"] and person.get("sameAs") == identity["same_as"], route
    else:
        assert not profile_pages, f"Unexpected ProfilePage markup: {route}"
    if route in {"/", "/fa/"}:
        assert p.meta.get("og:site_name") == ["Mohammad Shojaei"], route
        people = [node for node in nodes if node.get("@type") == "Person" and node.get("@id") == "https://mshojaei77.github.io/#person"]
        language = "fa" if route == "/fa/" else "en"
        identity = author_profiles["محمد شجاعی" if language == "fa" else "Mohammad Shojaei"]
        assert len(people) == 1 and people[0].get("sameAs") == identity["same_as"], route
        assert people[0].get("name") == identity["name"] and people[0].get("url") == identity["url"], route
        websites = [node for node in nodes if node.get("@type") == "WebSite"]
        if route == "/":
            assert len(websites) == 1, (route, websites)
            assert websites[0].get("@id") == host + "/#website"
            assert websites[0].get("url") == host + "/"
            assert websites[0].get("name") == "Mohammad Shojaei"
            assert websites[0].get("alternateName") == "mshojaei77.github.io"
            assert websites[0].get("publisher") == {"@id": host + "/#person"}
        else:
            assert not websites, f"WebSite site-name markup should be declared on the domain root only: {route}"
    if route in articles:
        language = "fa" if route.startswith("/fa/") else "en"
        slug = route.rstrip("/").split("/")[-1]
        topic_ids = set(topics.get("posts", {}).get(slug, []))
        recommendations = topics.get("related_by_route", {}).get(route, [])
        expected_related = set()
        expected_related_order = []
        if recommendations:
            for item in recommendations[:3]:
                target_slug = item["slug"]
                target_routes = [
                    candidate_route for candidate_route in articles
                    if candidate_route.rstrip("/").split("/")[-1] == target_slug
                    and ("fa" if candidate_route.startswith("/fa/") else "en") == language
                ]
                assert len(target_routes) == 1, (route, target_slug, target_routes)
                expected_related_order.append(target_routes[0])
            expected_related = set(expected_related_order)
        elif topic_ids:
            for candidate_route in articles:
                candidate_language = "fa" if candidate_route.startswith("/fa/") else "en"
                candidate_slug = candidate_route.rstrip("/").split("/")[-1]
                candidate_topics = set(topics.get("posts", {}).get(candidate_slug, []))
                if candidate_route != route and candidate_language == language and topic_ids.intersection(candidate_topics):
                    expected_related.add(candidate_route)
        related = RelatedLinks()
        related.feed(file_for(route).read_text(encoding="utf-8"))
        if expected_related:
            assert 1 <= len(related.links) <= 3, f"Expected one to three related links for {route}"
            assert all(href in expected_related for href, _ in related.links), (route, related.links, expected_related)
            if recommendations:
                assert [href for href, _ in related.links] == expected_related_order, (route, related.links, expected_related_order)
            assert all(hreflang == language for _, hreflang in related.links), (route, related.links)
        else:
            assert not related.links, f"Unrelated article suggestions rendered for {route}"
        blogs = [s for s in nodes if s.get("@type") == "BlogPosting"]
        assert len(blogs) == 1, (route,p.schemas)
        blog = blogs[0]
        assert blog.get("headline") and blog.get("author",{}).get("name") and blog.get("datePublished"), route
        identity_key = "محمد شجاعی" if route.startswith("/fa/") else "Mohammad Shojaei"
        assert blog["author"].get("name") == author_profiles[identity_key]["name"], route
        assert blog["author"].get("url") == author_profiles[identity_key]["url"], route
        author_links = AuthorLinks()
        author_links.feed(file_for(route).read_text(encoding="utf-8"))
        assert author_links.hrefs == [author_profiles[identity_key]["url"]], (route, author_links.hrefs)
        assert blog.get("url") == host+route, (route,blog)
        byline = ArticleByline()
        byline.feed(file_for(route).read_text(encoding="utf-8"))
        published = blog["datePublished"]
        assert any(item["datetime"].startswith(published[:10]) for item in byline.times), (route, byline.times, published)
        assert ("تاریخ انتشار" in byline.text) if route.startswith("/fa/") else ("Published" in byline.text), route
        modified = blog.get("dateModified", "")
        if modified and modified[:10] != published[:10]:
            assert any(item["datetime"].startswith(modified[:10]) for item in byline.times), (route, byline.times, modified)
            assert ("آخرین به‌روزرسانی" in byline.text) if route.startswith("/fa/") else ("Last updated" in byline.text), route
        listed_lastmod = sitemap_lastmod.get(host + route, "")
        assert listed_lastmod and listed_lastmod[:10] == (modified or published)[:10], (route, listed_lastmod, modified or published)
for route in ["/admin/","/projects/","/fa/market-analysis/","/writing/document-parsing/","/fa/writing/document-parsing/"]:
    if route not in public and file_for(route).is_file():
        assert not analytics(parse(file_for(route))), route
assert "/admin/" not in routes
print(f"Verified SEO, canonical URLs and analytics isolation for {len(articles)} live articles; configured={bool(settings['goatcounter_site'])}.")

# The wildcard robots group currently permits all public search, AI-search,
# user-fetch, and training crawlers. Keep its Studio exclusion intact when
# adding any provider-specific groups.
robots = (root / "robots.txt").read_text(encoding="utf-8")
groups = {}
agents = []
rules = []
def save_robots_group():
    global agents, rules
    for agent in agents:
        groups.setdefault(agent, []).extend(rules)
    agents, rules = [], []
for raw in robots.splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line or ":" not in line:
        continue
    name, value = (part.strip() for part in line.split(":", 1))
    if name.lower() == "user-agent":
        if rules:
            save_robots_group()
        agents.append(value.lower())
    elif agents:
        rules.append((name.lower(), value))
save_robots_group()
for agent in ("googlebot", "google-extended", "bingbot", "duckduckbot", "duckassistbot", "oai-searchbot", "gptbot", "chatgpt-user", "perplexitybot", "perplexity-user", "claudebot"):
    effective = groups.get(agent, groups.get("*", []))
    assert ("disallow", "/admin/") in effective, (agent, effective)
    assert not any(name == "disallow" and value == "/" for name, value in effective), (agent, effective)
print("Verified crawler access policy for search, user-fetch and training agents; Studio remains excluded.")

# Preserve Google's exact root verification file through every build.
verification = "google483b45b4cf9637cf.html"
assert (root / verification).read_bytes() == b"google-site-verification: google483b45b4cf9637cf.html"
assert "/" + verification not in routes, "Verification file should not be in sitemap"
print("Verified exact Google ownership file.")

# Preserve the exact Bing ownership file in the published root.
assert (root / "BingSiteAuth.xml").read_bytes() == "<?xml version=\"1.0\"?>\n<users>\n\t<user>2AA6B158D3D26282FB332CC19F0505F7</user>\n</users>".encode("utf-8")
assert "/BingSiteAuth.xml" not in routes
print("Verified exact Bing ownership file.")

# IndexNow's key is intentionally public: the protocol requires a root file
# whose name and bytes match the key. It is not a credential for any dashboard.
indexnow_key = settings.get("indexnow_key", "")
assert re.fullmatch(r"[A-Za-z0-9-]{8,128}", indexnow_key), "Invalid IndexNow key"
assert (root / f"{indexnow_key}.txt").read_bytes() == indexnow_key.encode("ascii")
assert f"/{indexnow_key}.txt" not in routes, "IndexNow proof file should not be in the editorial sitemap"
print("Verified public IndexNow key file and sitemap exclusion.")
