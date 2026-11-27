"""Crawl and audit public GitHub Pages URLs; writes a timestamped JSON snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://mshojaei77.github.io"
HOST = "mshojaei77.github.io"
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARTICLE_URL = re.compile(r"^https://mshojaei77\.github\.io/(?:fa/)?writing/[^/]+/$")
FRONT_MATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.S)
PERMALINK = re.compile(r"^permalink:\s*['\"]?([^\s'\"]+)['\"]?\s*$", re.M)


class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.chain: list[dict[str, str | int]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.chain.append({"status": code, "from": req.full_url, "to": newurl})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.lang = ""
        self.direction = ""
        self.canonicals: list[str] = []
        self.robots: dict[str, list[str]] = defaultdict(list)
        self.social_metadata: dict[str, list[str]] = defaultdict(list)
        self.hreflang: dict[str, list[str]] = defaultdict(list)
        self.headings: list[dict[str, str]] = []
        self.links: list[str] = []
        self.link_details: list[dict[str, str]] = []
        self.link_text: list[str] = []
        self.schemas: list[dict] = []
        self.schema_errors: list[str] = []
        self.retired = False
        self.meta_refresh = ""
        self._in_title = False
        self._heading_level = 0
        self._heading_text = ""
        self._in_article = False
        self._in_related = False
        self._article_text: list[str] = []
        self._in_jsonld = False
        self._jsonld = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "html":
            self.lang, self.direction = a.get("lang", ""), a.get("dir", "")
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            name = (a.get("name") or "").lower()
            prop = (a.get("property") or "").lower()
            content = a.get("content", "") or ""
            if name == "description":
                self.description = content
            elif name in {"robots", "googlebot", "bingbot"}:
                self.robots[name].append(content)
            if prop.startswith("og:") or name.startswith("twitter:"):
                self.social_metadata[prop or name].append(content)
            elif (a.get("http-equiv") or "").lower() == "refresh":
                self.meta_refresh = content
        elif tag == "link":
            rel = (a.get("rel") or "").lower().split()
            if "canonical" in rel:
                self.canonicals.append(a.get("href", "") or "")
            if "alternate" in rel and a.get("hreflang"):
                self.hreflang[a["hreflang"].lower()].append(a.get("href", "") or "")
        elif tag == "section" and "related-articles" in (a.get("class") or "").split():
            self._in_related = True
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
            kind = "author" if "author" in (a.get("rel") or "").lower().split() else "related" if self._in_related else "content" if self._in_article else "navigation"
            self.link_details.append({"href": a["href"], "kind": kind})
            self.link_text.append("")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._heading_text = ""
        elif tag == "article" and "article-content" in (a.get("class") or "").split():
            self._in_article = True
        elif tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
            self._in_jsonld = True
            self._jsonld = ""
        if "data-retired-post" in a:
            self.retired = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_level:
            self.headings.append({"level": self._heading_level, "text": " ".join(self._heading_text.split())})
            self._heading_level, self._heading_text = 0, ""
        elif tag == "article":
            self._in_article = False
        elif tag == "section" and self._in_related:
            self._in_related = False
        elif tag == "script" and self._in_jsonld:
            try:
                value = json.loads(self._jsonld)
                if isinstance(value, dict):
                    self.schemas.append(value)
                elif isinstance(value, list):
                    self.schemas.extend(x for x in value if isinstance(x, dict))
            except (json.JSONDecodeError, TypeError) as exc:
                self.schema_errors.append(str(exc))
            self._in_jsonld, self._jsonld = False, ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._heading_level:
            self._heading_text += data
        if self._in_article:
            self._article_text.append(data)
        if self._in_jsonld:
            self._jsonld += data


def source_routes() -> dict[str, list[dict[str, str]]]:
    routes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for kind, directory in (("article", "_posts"), ("retired", "retired-posts"), ("redirect", "redirects")):
        for path in sorted(Path(directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".html"}:
                continue
            source = path.read_text(encoding="utf-8")
            match = FRONT_MATTER.match(source)
            if not match:
                continue
            front = match.group(1)
            permalink = PERMALINK.search(front)
            if not permalink:
                continue
            lang = re.search(r"^lang:\s*['\"]?([^\s'\"]+)", front, re.M)
            routes[permalink.group(1)].append({"kind": kind, "path": path.as_posix(), "lang": lang.group(1) if lang else ""})
    return routes


def fetch(url: str, attempts: int = 3) -> dict:
    error = ""
    for attempt in range(attempts):
        redirects = RedirectRecorder()
        opener = urllib.request.build_opener(redirects)
        req = urllib.request.Request(url, headers={"User-Agent": "Publication-SEO-Audit/1.0", "Cache-Control": "no-cache"})
        try:
            with opener.open(req, timeout=25) as response:
                body = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
                return {"status": response.status, "final_url": response.geturl(), "headers": headers, "body": body, "redirect_chain": redirects.chain, "error": ""}
        except urllib.error.HTTPError as exc:
            body, status, final_url = exc.read(), exc.code, exc.geturl()
            headers = {k.lower(): v for k, v in exc.headers.items()}
            if status not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                return {"status": status, "final_url": final_url, "headers": headers, "body": body, "redirect_chain": redirects.chain, "error": ""}
            error = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError) as exc:
            error = str(exc)
            if attempt + 1 == attempts:
                return {"status": 0, "final_url": "", "headers": {}, "body": b"", "redirect_chain": redirects.chain, "error": error}
        time.sleep(1 + attempt)
    return {"status": 0, "final_url": "", "headers": {}, "body": b"", "redirect_chain": [], "error": error}


def same_host_page(href: str, base_url: str) -> str | None:
    target = urllib.parse.urljoin(base_url, href.strip())
    parts = urllib.parse.urlsplit(target)
    if parts.scheme not in {"http", "https"} or (parts.hostname or "").lower() != HOST:
        return None
    # Query strings and fragments are navigation state, not distinct crawl pages.
    scheme = parts.scheme.lower()
    return urllib.parse.urlunsplit((scheme, HOST, parts.path or "/", "", ""))


def parse_robots_rules(text: str) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    agents: list[str] = []
    rules: list[tuple[str, str]] = []
    def save() -> None:
        nonlocal agents, rules
        for agent in agents:
            groups[agent].extend(rules)
        agents, rules = [], []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (piece.strip() for piece in line.split(":", 1))
        if key.lower() == "user-agent":
            if rules:
                save()
            agents.append(value.lower())
        elif agents and key.lower() in {"allow", "disallow"}:
            rules.append((key.lower(), value))
    save()
    return groups


def crawler_allowed(path: str, rules: dict[str, list[tuple[str, str]]], agent: str = "googlebot") -> bool:
    effective = rules.get(agent.lower(), rules.get("*", []))
    matches = [(len(value), name == "allow") for name, value in effective if value and path.startswith(value)]
    return max(matches)[1] if matches else True


def flatten_schema_types(schemas: list[dict]) -> list[str]:
    types = []
    for schema in schemas:
        value = schema.get("@type", [])
        types.extend(value if isinstance(value, list) else [value] if value else [])
        graph = schema.get("@graph", [])
        for node in graph if isinstance(graph, list) else []:
            if isinstance(node, dict):
                node_type = node.get("@type", [])
                types.extend(node_type if isinstance(node_type, list) else [node_type] if node_type else [])
        main_entity = schema.get("mainEntity")
        if isinstance(main_entity, dict):
            node_type = main_entity.get("@type", [])
            types.extend(node_type if isinstance(node_type, list) else [node_type] if node_type else [])
    return sorted(set(str(x) for x in types))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="UNKNOWN", help="Verified production revision SHA.")
    parser.add_argument("--output", default="scripts/reports/seo-url-audit.json", help="JSON output path; kept under scripts/ so Jekyll excludes it.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when critical production discovery checks fail.")
    args = parser.parse_args()

    sitemap_response = fetch(BASE + "/sitemap.xml")
    if sitemap_response["status"] != 200:
        raise SystemExit(f"Production sitemap HTTP {sitemap_response['status']}: {sitemap_response['error']}")
    sitemap_xml = ET.fromstring(sitemap_response["body"])
    sitemap_entries = [node.text for node in sitemap_xml.findall(f"{SITEMAP_NS}url/{SITEMAP_NS}loc") if node.text]
    sitemap_urls = set(sitemap_entries)
    rss_response = fetch(BASE + "/feed.xml")
    rss_urls = set()
    if rss_response["status"] == 200:
        feed = ET.fromstring(rss_response["body"])
        # Jekyll emits Atom links in href attributes, not as text nodes.
        rss_urls = {
            node.get("href", "")
            for node in feed.findall(f"{ATOM_NS}entry/{ATOM_NS}link[@rel='alternate']")
            if node.get("href")
        }
    source = source_routes()
    author_profiles = json.loads(Path("_data/authors.json").read_text(encoding="utf-8")) if Path("_data/authors.json").is_file() else {}
    collisions = {route: files for route, files in source.items() if len(files) > 1}
    retired_routes = {route for route, files in source.items() if any(x["kind"] == "retired" for x in files)}
    redirect_routes = {route for route, files in source.items() if any(x["kind"] == "redirect" for x in files)}
    post_routes = {route for route, files in source.items() if any(x["kind"] == "article" for x in files)}
    expected_pages = {url for url in sitemap_urls}
    explicit = [
        BASE + "/admin/",
        BASE + "/google483b45b4cf9637cf.html",
        BASE + "/BingSiteAuth.xml",
        BASE + "/robots.txt",
        BASE + "/feed.xml",
        BASE + "/404.html",
    ]
    discovery = json.loads(Path("_data/discovery.json").read_text(encoding="utf-8"))
    explicit.append(BASE + f"/{discovery['indexnow_key']}.txt")
    for route in sorted(retired_routes | redirect_routes):
        explicit.append(BASE + route)

    queued = deque(sorted(expected_pages | set(explicit)))
    seen: set[str] = set()
    inlinks: dict[str, set[str]] = defaultdict(set)
    outlinks: dict[str, set[str]] = defaultdict(set)
    link_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    records: dict[str, dict] = {}
    robots_response = fetch(BASE + "/robots.txt")
    robots_text = robots_response["body"].decode("utf-8", "replace") if robots_response["status"] == 200 else ""
    robots_rules = parse_robots_rules(robots_text)

    while queued and len(seen) < 300:
        url = queued.popleft()
        if url in seen:
            continue
        seen.add(url)
        response = fetch(url)
        path = urllib.parse.urlsplit(url).path or "/"
        expected = "indexable" if url in sitemap_urls else (
            "retired" if path in retired_routes else "compatibility_redirect" if path in redirect_routes else
            "admin" if path.startswith("/admin/") else "verification_or_feed" if path in {
                "/google483b45b4cf9637cf.html", "/BingSiteAuth.xml", "/feed.xml", "/robots.txt", "/404.html"
            } or path.endswith(".txt") else "discovered_not_in_sitemap"
        )
        parser_html = PageParser()
        content_type = response["headers"].get("content-type", "")
        if "html" in content_type.lower() and response["body"]:
            parser_html.feed(response["body"].decode("utf-8", "replace"))
        meta_refresh_target = ""
        if parser_html.meta_refresh:
            match = re.search(r"url\s*=\s*['\"]?([^'\"]+)\s*$", parser_html.meta_refresh, re.I)
            if match:
                meta_refresh_target = urllib.parse.urljoin(url, match.group(1).strip())
        full_url = response["final_url"] or url
        canonical = parser_html.canonicals[0] if parser_html.canonicals else ""
        headers_robots = response["headers"].get("x-robots-tag", "")
        robots_directives = {name: values for name, values in parser_html.robots.items()}
        noindex = any("noindex" in item.lower() for values in robots_directives.values() for item in values) or "noindex" in headers_robots.lower()
        record = {
            "url": url,
            "content_type": content_type,
            "language": parser_html.lang,
            "direction": parser_html.direction,
            "publication_status": expected,
            "expected_indexability": expected == "indexable",
            "http_status": response["status"],
            "final_url": full_url,
            "redirect_chain": response["redirect_chain"] + ([{"type": "meta-refresh", "to": meta_refresh_target}] if meta_refresh_target else []),
            "response_headers": {
                name: response["headers"][name]
                for name in ("content-type", "cache-control", "age", "etag", "last-modified", "x-cache", "via")
                if name in response["headers"]
            },
            "canonical_url": canonical,
            "robots_meta": dict(robots_directives),
            "x_robots_tag": headers_robots,
            "crawlability": "allowed" if crawler_allowed(path, robots_rules) else "blocked",
            "indexability": "noindex" if noindex else "indexable" if response["status"] == 200 and canonical == url else "unknown_or_noncanonical",
            "title": " ".join(parser_html.title.split()),
            "description": parser_html.description,
            "h1": [x["text"] for x in parser_html.headings if x["level"] == 1],
            "headings": parser_html.headings,
            "hreflang": {k: v for k, v in parser_html.hreflang.items()},
            "social_metadata": {k: v for k, v in parser_html.social_metadata.items()},
            "structured_data": flatten_schema_types(parser_html.schemas),
            "structured_data_summary": [
                {
                    key: schema.get(key)
                    for key in ("@type", "@id", "url", "headline", "name", "description", "datePublished", "dateModified", "inLanguage", "author", "mainEntity", "mainEntityOfPage")
                    if schema.get(key) is not None
                }
                for schema in parser_html.schemas
            ],
            "structured_data_valid": not parser_html.schema_errors,
            "structured_data_errors": parser_html.schema_errors,
            "sitemap_membership": url in sitemap_urls,
            "rss_membership": url in rss_urls,
            "internal_inlinks": [],
            "internal_outlinks": [],
            "visible_author_profile_links": sorted({
                urllib.parse.urljoin(url, link["href"])
                for link in parser_html.link_details
                if link["kind"] == "author"
            }),
            "content_hash_sha256": hashlib.sha256(response["body"]).hexdigest() if response["body"] else "",
            "rendered_content_available": bool(" ".join(parser_html._article_text).strip()) if expected == "indexable" and "writing/" in path else bool(" ".join(parser_html._article_text).strip()) or bool(parser_html.title),
            "retired_post_marker": parser_html.retired or path in retired_routes,
            "deployment_revision": args.revision,
            "source_files": source.get(path, []),
            "meta_refresh_target": meta_refresh_target,
            "response_error": response["error"],
            "body_bytes": len(response["body"]),
        }
        records[url] = record
        if response["status"] == 200 and parser_html.lang:
            for link in parser_html.link_details:
                href = link["href"]
                target = same_host_page(href, url)
                if not target or target == url:
                    continue
                outlinks[url].add(target)
                inlinks[target].add(url)
                link_kinds[(url, target)].add(link["kind"])
                if target not in seen:
                    queued.append(target)
        if meta_refresh_target:
            target = same_host_page(meta_refresh_target, url)
            if target:
                outlinks[url].add(target)
                inlinks[target].add(url)
                if target not in seen:
                    queued.append(target)
        if response["redirect_chain"]:
            target = same_host_page(response["final_url"], url)
            if target and target != url:
                outlinks[url].add(target)
                inlinks[target].add(url)
                if target not in seen:
                    queued.append(target)

    def click_depths(start: str) -> dict[str, int]:
        distances = {start: 0}
        frontier = deque([start])
        while frontier:
            current = frontier.popleft()
            for target in outlinks.get(current, set()):
                if target in records and target not in distances:
                    distances[target] = distances[current] + 1
                    frontier.append(target)
        return distances

    en_depths = click_depths(BASE + "/")
    fa_depths = click_depths(BASE + "/fa/")
    for url, record in records.items():
        record["internal_inlinks"] = sorted(inlinks.get(url, set()))
        record["internal_outlinks"] = sorted(outlinks.get(url, set()))
        target_lang = "fa" if urllib.parse.urlsplit(url).path.startswith("/fa/") else "en"
        article_sources = [source_url for source_url in records if ARTICLE_URL.fullmatch(source_url)]
        record["article_inlinks"] = sorted(
            source_url for source_url in article_sources
            if (source_url, url) in link_kinds
            and ("content" in link_kinds[(source_url, url)] or "related" in link_kinds[(source_url, url)])
            and ("fa" if urllib.parse.urlsplit(source_url).path.startswith("/fa/") else "en") == target_lang
        ) if ARTICLE_URL.fullmatch(url) else []
        record["contextual_article_inlinks"] = sorted(
            source_url for source_url in article_sources
            if "content" in link_kinds.get((source_url, url), set())
            and ("fa" if urllib.parse.urlsplit(source_url).path.startswith("/fa/") else "en") == target_lang
        ) if ARTICLE_URL.fullmatch(url) else []
        record["related_article_inlinks"] = sorted(
            source_url for source_url in article_sources
            if "related" in link_kinds.get((source_url, url), set())
            and ("fa" if urllib.parse.urlsplit(source_url).path.startswith("/fa/") else "en") == target_lang
        ) if ARTICLE_URL.fullmatch(url) else []
        record["article_outlinks"] = sorted(
            target_url for target_url in outlinks.get(url, set())
            if ARTICLE_URL.fullmatch(target_url)
            and ("fa" if urllib.parse.urlsplit(target_url).path.startswith("/fa/") else "en") == target_lang
            and link_kinds.get((url, target_url), set()).intersection({"content", "related"})
        ) if ARTICLE_URL.fullmatch(url) else []
        record["click_depth_from_en_home"] = en_depths.get(url)
        record["click_depth_from_fa_home"] = fa_depths.get(url)
        depths = [value for value in (en_depths.get(url), fa_depths.get(url)) if value is not None]
        record["click_depth_from_home"] = min(depths) if depths else None

    indexable_records = [records[url] for url in sitemap_urls if url in records]
    article_path = re.compile(r"^/(?:fa/)?writing/[^/]+/$")
    articles_en = [
        item for item in indexable_records
        if article_path.fullmatch(urllib.parse.urlsplit(item["url"]).path) and item["language"] == "en"
    ]
    articles_fa = [
        item for item in indexable_records
        if article_path.fullmatch(urllib.parse.urlsplit(item["url"]).path) and item["language"] == "fa"
    ]
    unexpected_noindex = [x["url"] for x in indexable_records if "noindex" in x["indexability"]]
    canonical_conflicts = [x["url"] for x in indexable_records if x["canonical_url"] != x["url"]]
    broken_links = [x["url"] for x in records.values() if x["http_status"] == 0 or x["http_status"] >= 400]
    orphan_articles = [
        x["url"] for x in indexable_records
        if re.fullmatch(r"https://mshojaei77\.github\.io/(?:fa/)?writing/[^/]+/", x["url"])
        and not x["internal_inlinks"]
    ]
    invalid_hreflang = []
    for record in indexable_records:
        for lang, alternates in record["hreflang"].items():
            for alternate in alternates:
                if alternate not in sitemap_urls or alternate not in records:
                    invalid_hreflang.append({"url": record["url"], "lang": lang, "alternate": alternate})
    invalid_structured_data = [x["url"] for x in indexable_records if not x["structured_data_valid"]]
    author_identity_issues = []
    for item in indexable_records:
        path = urllib.parse.urlsplit(item["url"]).path
        language = "fa" if path.startswith("/fa/") else "en"
        expected_author = author_profiles.get("محمد شجاعی" if language == "fa" else "Mohammad Shojaei", {})
        schemas = item["structured_data_summary"]
        if article_path.fullmatch(path):
            blog_posts = [schema for schema in schemas if schema.get("@type") == "BlogPosting"]
            if len(blog_posts) != 1:
                author_identity_issues.append({"url": item["url"], "issue": f"Expected one BlogPosting, found {len(blog_posts)}"})
            else:
                article_author = blog_posts[0].get("author", {})
                if (
                    article_author.get("name") != expected_author.get("name")
                    or article_author.get("url") != expected_author.get("url")
                    or item["visible_author_profile_links"] != [expected_author.get("url")]
                ):
                    author_identity_issues.append({"url": item["url"], "issue": "Visible author link or BlogPosting author identity does not match the localized public profile"})
        if path in {"/about.html", "/fa/about/"}:
            profile_pages = [schema for schema in schemas if schema.get("@type") == "ProfilePage"]
            if len(profile_pages) != 1:
                author_identity_issues.append({"url": item["url"], "issue": f"Expected one ProfilePage, found {len(profile_pages)}"})
            else:
                profile = profile_pages[0]
                person = profile.get("mainEntity", {})
                if profile.get("url") != item["url"] or person.get("@id") != "https://mshojaei77.github.io/#person" or person.get("name") != expected_author.get("name") or person.get("sameAs") != expected_author.get("same_as"):
                    author_identity_issues.append({"url": item["url"], "issue": "ProfilePage mainEntity does not match the localized public identity"})
    missing_metadata = [
        x["url"] for x in indexable_records
        if not x["title"].strip() or not x["description"].strip() or not x["language"]
        or not x["social_metadata"].get("og:title") or not x["social_metadata"].get("og:description")
        or not x["social_metadata"].get("og:url") or not x["social_metadata"].get("og:image")
    ]
    missing_article_content = [
        item["url"] for item in indexable_records
        if article_path.fullmatch(urllib.parse.urlsplit(item["url"]).path)
        and not item["rendered_content_available"]
    ]
    article_metadata_duplicates = [
        x["url"] for x in indexable_records
        if article_path.fullmatch(urllib.parse.urlsplit(x["url"]).path)
        and (x["social_metadata"].get("og:title") or [x["title"]])[0].strip().casefold() == x["description"].strip().casefold()
    ]
    language_metadata_mismatches = [
        item["url"] for item in indexable_records
        if item["language"] != ("fa" if urllib.parse.urlsplit(item["url"]).path.startswith("/fa/") else "en")
        or item["direction"] != ("rtl" if urllib.parse.urlsplit(item["url"]).path.startswith("/fa/") else "ltr")
    ]
    rss_missing_post_routes = sorted(route for route, files in source.items() if any(x["kind"] == "article" for x in files) and BASE + route not in rss_urls)
    robots_policy_issues = []
    if robots_response["status"] != 200:
        robots_policy_issues.append(f"robots.txt returned HTTP {robots_response['status']}")
    for agent in ("googlebot", "bingbot", "oai-searchbot", "perplexitybot"):
        if not crawler_allowed("/fa/writing/document-parsing/", robots_rules, agent):
            robots_policy_issues.append(f"{agent} is blocked from a public article")
        if crawler_allowed("/admin/", robots_rules, agent):
            robots_policy_issues.append(f"{agent} is not blocked from /admin/")
    active_tombstone_collisions = [route for route, files in source.items() if any(x["kind"] == "article" for x in files) and any(x["kind"] == "retired" for x in files)]
    missing_post_routes = sorted(route for route, files in source.items() if any(x["kind"] == "article" for x in files) and BASE + route not in sitemap_urls)

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_base": BASE,
        "deployment_revision": args.revision,
        "discovery_endpoints": {
            "/sitemap.xml": {
                "http_status": sitemap_response["status"],
                "final_url": sitemap_response["final_url"],
                "redirect_chain": sitemap_response["redirect_chain"],
                "content_type": sitemap_response["headers"].get("content-type", ""),
                "xml_valid": True,
                "root_element": sitemap_xml.tag,
                "url_count": len(sitemap_urls),
                "unique_url_count": len(sitemap_urls),
                "duplicate_url_entry_count": len(sitemap_entries) - len(sitemap_urls),
                "body_bytes": len(sitemap_response["body"]),
                "headers": {
                    name: sitemap_response["headers"][name]
                    for name in ("cache-control", "age", "etag", "last-modified", "x-cache", "via")
                    if name in sitemap_response["headers"]
                },
            },
            "/feed.xml": {
                "http_status": rss_response["status"],
                "final_url": rss_response["final_url"],
                "redirect_chain": rss_response["redirect_chain"],
                "content_type": rss_response["headers"].get("content-type", ""),
                "xml_valid": rss_response["status"] == 200,
                "entry_count": len(rss_urls),
                "article_entry_count": sum(1 for url in rss_urls if article_path.fullmatch(urllib.parse.urlsplit(url).path)),
                "body_bytes": len(rss_response["body"]),
                "headers": {
                    name: rss_response["headers"][name]
                    for name in ("cache-control", "age", "etag", "last-modified", "x-cache", "via")
                    if name in rss_response["headers"]
                },
            },
            "/robots.txt": {
                "http_status": robots_response["status"],
                "final_url": robots_response["final_url"],
                "redirect_chain": robots_response["redirect_chain"],
                "content_type": robots_response["headers"].get("content-type", ""),
                "googlebot_public_access": crawler_allowed("/fa/writing/document-parsing/", robots_rules, "googlebot"),
                "bingbot_public_access": crawler_allowed("/fa/writing/document-parsing/", robots_rules, "bingbot"),
                "openai_search_public_access": crawler_allowed("/fa/writing/document-parsing/", robots_rules, "oai-searchbot"),
                "perplexity_public_access": crawler_allowed("/fa/writing/document-parsing/", robots_rules, "perplexitybot"),
                "admin_disallowed_for_googlebot": not crawler_allowed("/admin/", robots_rules, "googlebot"),
                "admin_disallowed_for_bingbot": not crawler_allowed("/admin/", robots_rules, "bingbot"),
                "policy_text": robots_text,
            },
        },
        "inventory_counts": {
            "sitemap_urls": len(sitemap_urls),
            "production_urls_crawled": len(records),
            "articles_en": len(articles_en),
            "articles_fa": len(articles_fa),
            "post_source_routes": len(post_routes),
            "retired_routes": len(retired_routes),
            "compatibility_redirects": len(redirect_routes),
            "rss_urls": len(rss_urls),
            "rss_article_urls": sum(1 for url in rss_urls if article_path.fullmatch(urllib.parse.urlsplit(url).path)),
            "sitemap_indexable_pages": len(indexable_records),
        },
        "issue_counts": {
            "unexpected_noindex": len(unexpected_noindex),
            "canonical_conflicts": len(canonical_conflicts),
            "source_route_collisions": len(collisions),
            "active_tombstone_collisions": len(active_tombstone_collisions),
            "broken_internal_or_seed_urls": len(broken_links),
            "orphan_articles": len(orphan_articles),
            "invalid_hreflang": len(invalid_hreflang),
            "invalid_structured_data_json": len(invalid_structured_data),
            "article_description_repeats_title": len(article_metadata_duplicates),
            "missing_core_or_social_metadata": len(missing_metadata),
            "missing_article_content_without_js": len(missing_article_content),
            "author_identity_issues": len(author_identity_issues),
            "language_metadata_mismatches": len(language_metadata_mismatches),
            "duplicate_sitemap_entries": len(sitemap_entries) - len(sitemap_urls),
            "rss_missing_post_routes": len(rss_missing_post_routes),
            "robots_policy_issues": len(robots_policy_issues),
            "missing_post_routes_from_sitemap": len(missing_post_routes),
        },
        "issues": {
            "unexpected_noindex": unexpected_noindex,
            "canonical_conflicts": canonical_conflicts,
            "source_route_collisions": collisions,
            "active_tombstone_collisions": active_tombstone_collisions,
            "broken_internal_or_seed_urls": broken_links,
            "orphan_articles": orphan_articles,
            "invalid_hreflang": invalid_hreflang,
            "invalid_structured_data_json": invalid_structured_data,
            "article_description_repeats_title": article_metadata_duplicates,
            "missing_core_or_social_metadata": missing_metadata,
            "missing_article_content_without_js": missing_article_content,
            "author_identity_issues": author_identity_issues,
            "language_metadata_mismatches": language_metadata_mismatches,
            "duplicate_sitemap_entries": len(sitemap_entries) - len(sitemap_urls),
            "rss_missing_post_routes": rss_missing_post_routes,
            "robots_policy_issues": robots_policy_issues,
            "missing_post_routes_from_sitemap": missing_post_routes,
        },
        "urls": [records[url] for url in sorted(records)],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    strict_fields = (
        "unexpected_noindex", "canonical_conflicts", "source_route_collisions",
        "active_tombstone_collisions", "broken_internal_or_seed_urls", "orphan_articles",
        "invalid_hreflang", "invalid_structured_data_json", "language_metadata_mismatches",
        "missing_post_routes_from_sitemap", "missing_core_or_social_metadata",
        "missing_article_content_without_js",
        "author_identity_issues",
        "duplicate_sitemap_entries", "rss_missing_post_routes", "robots_policy_issues",
    )
    strict_failures = [f"{name}={report['issue_counts'][name]}" for name in strict_fields if report["issue_counts"][name]]
    report["strict_check"] = {"enabled": args.strict, "passed": not strict_failures, "failures": strict_failures}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "inventory_counts": report["inventory_counts"], "issue_counts": report["issue_counts"], "strict_check": report["strict_check"]}, ensure_ascii=False))
    if args.strict and strict_failures:
        raise SystemExit("Critical production SEO checks failed: " + ", ".join(strict_failures))


if __name__ == "__main__":
    main()
