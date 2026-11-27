"""Bounded post-deployment smoke check for public discovery endpoints."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

BASE = "https://mshojaei77.github.io"
REPO = "mshojaei77/mshojaei77.github.io"
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
ATOM = "{http://www.w3.org/2005/Atom}"


def get(path: str, attempts: int = 8) -> tuple[bytes, dict[str, str], int]:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(BASE + path, headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "Publication-Production-Smoke/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}, response.status
        except urllib.error.HTTPError as exc:
            error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(5 * (attempt + 1), 20))
    raise RuntimeError(f"GET {path} failed after bounded retries: {error}")


def assert_200(path: str) -> tuple[bytes, dict[str, str]]:
    body, headers, status = get(path)
    assert status == 200, f"{path}: HTTP {status}"
    print(f"PASS {path} HTTP {status} {headers.get('content-type', '')}")
    return body, headers


def robots_group(text: str, user_agent: str) -> list[str]:
    groups: dict[str, list[str]] = {}
    current: list[str] = []
    agents: list[str] = []
    def save() -> None:
        nonlocal agents, current
        for agent in agents:
            groups.setdefault(agent, []).extend(current)
        agents, current = [], []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        if name.lower() == "user-agent":
            if current:
                save()
            agents.append(value.lower())
        elif agents:
            current.append(f"{name.lower()}:{value}")
    save()
    return groups.get(user_agent.lower(), groups.get("*", []))


class IndexablePage(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lang = ""
        self.in_title = False
        self.title = ""
        self.canonicals: list[str] = []
        self.robots: list[str] = []
        self.icons: list[dict[str, str]] = []
        self.schemas: list[dict] = []
        self.in_jsonld = False
        self.jsonld = ""
        self.in_byline = False
        self.active_time: dict[str, str] | None = None
        self.byline_times: list[dict[str, str]] = []
        self.byline_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang", "") or ""
        elif tag == "title":
            self.in_title = True
        elif tag == "meta" and (values.get("name", "").lower() in {"robots", "googlebot", "bingbot"}):
            self.robots.append((values.get("content", "") or "").lower())
        elif tag == "link" and "canonical" in (values.get("rel", "").lower().split()):
            self.canonicals.append(values.get("href", "") or "")
        if tag == "link" and "icon" in (values.get("rel", "").lower().split()):
            self.icons.append({key: value or "" for key, value in values.items() if key in {"rel", "type", "sizes", "href"}})
        if tag == "p" and "article-byline" in (values.get("class", "").split()):
            self.in_byline = True
        elif self.in_byline and tag == "time":
            self.active_time = {"datetime": values.get("datetime", "") or "", "text": ""}
            self.byline_times.append(self.active_time)
        if tag == "script" and (values.get("type", "").lower() == "application/ld+json"):
            self.in_jsonld = True
            self.jsonld = ""

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.in_byline:
            self.byline_text += data
            if self.active_time is not None:
                self.active_time["text"] += data
        if self.in_jsonld:
            self.jsonld += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "time":
            self.active_time = None
        elif tag == "p" and self.in_byline:
            self.in_byline = False
        elif tag == "script" and self.in_jsonld:
            parsed = json.loads(self.jsonld)
            if isinstance(parsed, dict):
                self.schemas.append(parsed)
            elif isinstance(parsed, list):
                self.schemas.extend(item for item in parsed if isinstance(item, dict))
            self.in_jsonld = False


def schema_nodes(page: IndexablePage) -> list[dict]:
    nodes = []
    for schema in page.schemas:
        nodes.append(schema)
        if isinstance(schema.get("@graph"), list):
            nodes.extend(node for node in schema["@graph"] if isinstance(node, dict))
    return nodes


def main() -> None:
    sha = os.environ.get("DEPLOYMENT_SHA", "")
    created_at = os.environ.get("DEPLOYMENT_CREATED_AT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("DEPLOYMENT_SHA must be the deployed 40-character commit SHA.")

    # The public Pages API provides the actual deployed commit. A newer deploy
    # may overtake this workflow_run; skip stale smoke jobs rather than report
    # the wrong revision as failed or verified.
    try:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("GITHUB_TOKEN with Pages read permission is required to identify the deployed revision.")
        api_request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/pages/builds/latest", headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "User-Agent": "Publication-Production-Smoke/1.0"})
        with urllib.request.urlopen(api_request, timeout=20) as response:
            pages_build = json.loads(response.read())
    except Exception as error:
        raise RuntimeError(f"Could not verify the latest Pages deployment commit: {error}") from error
    if pages_build.get("commit") != sha:
        print(f"SKIP superseded deployment {sha[:12]}; latest Pages commit is {pages_build.get('commit', 'unknown')[:12]}.")
        return
    print(f"PASS deployed revision {sha}")

    required = ["/", "/fa/", "/writing/", "/fa/writing/", "/about.html", "/roadmap.html", "/admin/", "/fa/writing/document-parsing/"]
    pages = {}
    page_headers = {}
    for path in required:
        body, headers = assert_200(path)
        pages[path] = body
        page_headers[path] = headers

    sitemap, _ = assert_200("/sitemap.xml")
    sitemap_root = ET.fromstring(sitemap)
    sitemap_urls = {element.text for element in sitemap_root.findall("s:url/s:loc", NS) if element.text}
    sitemap_lastmod = {
        element.findtext("s:loc", namespaces=NS): element.findtext("s:lastmod", namespaces=NS) or ""
        for element in sitemap_root.findall("s:url", NS)
    }
    assert BASE + "/fa/writing/document-parsing/" in sitemap_urls
    assert BASE + "/fa/writing/document-parsing-fa-review-v7/" not in sitemap_urls
    assert not any("/admin/" in url for url in sitemap_urls)
    print(f"PASS sitemap XML: {len(sitemap_urls)} canonical URLs")

    for canonical in sorted(sitemap_urls):
        parts = urlsplit(canonical)
        assert parts.scheme == "https" and parts.netloc == "mshojaei77.github.io" and not parts.query and not parts.fragment, canonical
        path = parts.path or "/"
        body = pages.get(path)
        headers = page_headers.get(path, {})
        if body is None:
            body, headers, status = get(path, attempts=4)
            assert status == 200, f"Sitemap URL {path}: HTTP {status}"
            pages[path] = body
            page_headers[path] = headers
        page = IndexablePage()
        page.feed(body.decode("utf-8"))
        assert page.canonicals == [canonical], f"{path}: canonical mismatch {page.canonicals}"
        assert page.title.strip(), f"{path}: missing title"
        expected_lang = "fa" if path == "/fa/" or path.startswith("/fa/") else "en"
        assert page.lang == expected_lang, f"{path}: expected lang={expected_lang}, got {page.lang!r}"
        assert not any("noindex" in directive for directive in page.robots), f"{path}: sitemap includes a noindex page"
        assert "noindex" not in headers.get("x-robots-tag", "").lower(), f"{path}: sitemap includes an X-Robots-Tag noindex page"
        assert page.icons == [{"rel": "icon", "type": "image/png", "sizes": "512x512", "href": "/assets/img/favicon.png"}], f"{path}: invalid favicon declaration {page.icons}"
        websites = [node for node in schema_nodes(page) if node.get("@type") == "WebSite"]
        if path == "/":
            assert len(websites) == 1 and websites[0].get("name") == "Mohammad Shojaei" and websites[0].get("url") == BASE + "/", f"Root WebSite schema missing or inaccurate: {websites}"
        else:
            assert not websites, f"WebSite site-name markup should be declared at the domain root only: {path}"
        if re.fullmatch(r"/(?:fa/)?writing/[^/]+/", path):
            posts = [node for node in schema_nodes(page) if node.get("@type") == "BlogPosting"]
            assert len(posts) == 1, f"{path}: expected one BlogPosting schema"
            post = posts[0]
            published = post.get("datePublished", "")
            modified = post.get("dateModified", "")
            assert published and any(item["datetime"].startswith(published[:10]) for item in page.byline_times), f"{path}: visible publication date does not match BlogPosting"
            if path.startswith("/fa/"):
                assert "تاریخ انتشار" in page.byline_text, f"{path}: missing Persian publication-date label"
            else:
                assert "Published" in page.byline_text, f"{path}: missing publication-date label"
            if modified and modified[:10] != published[:10]:
                assert any(item["datetime"].startswith(modified[:10]) for item in page.byline_times), f"{path}: visible modification date does not match BlogPosting"
                if path.startswith("/fa/"):
                    assert "آخرین به‌روزرسانی" in page.byline_text, f"{path}: missing Persian modification-date label"
                else:
                    assert "Last updated" in page.byline_text, f"{path}: missing modification-date label"
            lastmod = sitemap_lastmod.get(canonical, "")
            assert lastmod and lastmod[:10] == (modified or published)[:10], f"{path}: sitemap lastmod does not match article dates"
    print(f"PASS production sitemap crawlability/indexability: {len(sitemap_urls)} URLs return canonical, titled, indexable HTML")

    feed, _ = assert_200("/feed.xml")
    ET.fromstring(feed)
    robots, _ = assert_200("/robots.txt")
    robots_text = robots.decode("utf-8")
    assert "Disallow: /admin/" in robots_text and BASE + "/sitemap.xml" in robots_text
    for agent in ("Googlebot", "Google-Extended", "bingbot", "DuckDuckBot", "DuckAssistBot", "OAI-SearchBot", "GPTBot", "ChatGPT-User", "PerplexityBot", "Perplexity-User", "ClaudeBot"):
        rules = robots_group(robots_text, agent)
        assert "disallow:/admin/" in rules, f"{agent} must inherit the Studio exclusion"
        assert "disallow:/" not in rules, f"{agent} must be allowed to crawl public content"
    print("PASS robots policy keeps the public site crawlable and Studio excluded")

    google, _ = assert_200("/google483b45b4cf9637cf.html")
    assert google == b"google-site-verification: google483b45b4cf9637cf.html"
    bing, _ = assert_200("/BingSiteAuth.xml")
    assert bing == b'<?xml version="1.0"?>\n<users>\n\t<user>2AA6B158D3D26282FB332CC19F0505F7</user>\n</users>'
    ET.fromstring(bing)
    print("PASS preserved Google and Bing verification bytes")

    discovery = json.loads(Path("_data/discovery.json").read_text(encoding="utf-8"))
    key = discovery["indexnow_key"]
    key_body, _ = assert_200(f"/{key}.txt")
    assert key_body == key.encode("ascii")
    assert BASE + f"/{key}.txt" not in sitemap_urls

    favicon, _ = assert_200("/assets/img/favicon.png")
    expected_favicon = Path("assets/img/favicon.png").read_bytes()
    assert favicon == expected_favicon
    assert favicon[:8] == bytes.fromhex("89504e470d0a1a0a") and int.from_bytes(favicon[16:20], "big") == 512 and int.from_bytes(favicon[20:24], "big") == 512
    print("PASS production PNG favicon bytes and dimensions")

    home = pages["/"].decode("utf-8")
    fa_home = pages["/fa/"].decode("utf-8")
    assert '<meta property="og:site_name" content="Mohammad Shojaei">' in home and '<meta property="og:site_name" content="Mohammad Shojaei">' in fa_home
    assert '/assets/js/analytics.js?v=1' in home
    assert 'data-goatcounter-site="mshojaei77"' in home
    analytics, _ = assert_200("/assets/js/analytics.js?v=1")
    assert b"count.v5.js" in analytics and b"script.integrity" in analytics and b"sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ" in analytics
    admin = pages["/admin/"].decode("utf-8")
    assert "/assets/js/analytics.js" not in admin and "noindex,nofollow,noarchive" in admin
    print("PASS analytics loader, official SRI pin and Studio tracking isolation")

    if created_at:
        expected = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        _, headers, _ = get("/")
        modified = parsedate_to_datetime(headers["last-modified"]).astimezone(timezone.utc)
        assert modified >= expected, f"Root Last-Modified {modified.isoformat()} predates deployment start {expected.isoformat()}"
        print(f"PASS production Last-Modified {modified.isoformat()} follows deployment start")


if __name__ == "__main__":
    main()
