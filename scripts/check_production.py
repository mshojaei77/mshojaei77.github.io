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

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False


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
    sitemap_urls = {element.text for element in ET.fromstring(sitemap).findall("s:url/s:loc", NS)}
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

    home = pages["/"].decode("utf-8")
    assert '/assets/js/analytics.js?v=1' in home
    assert 'data-goatcounter-site="mshojaei77"' in home
    analytics, _ = assert_200("/assets/js/analytics.js?v=1")
    assert b"count.v5.js" in analytics and b"script.integrity" in analytics and b"sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ" in analytics
    admin = pages["/admin/"].decode("utf-8")
    assert "/assets/js/analytics.js" not in admin and "noindex,nofollow,noarchive" in admin
    print("PASS analytics loader, official SRI pin and Studio tracking isolation")

    if created_at:
        expected = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        _, headers = get("/")
        modified = parsedate_to_datetime(headers["last-modified"]).astimezone(timezone.utc)
        assert modified >= expected, f"Root Last-Modified {modified.isoformat()} predates deployment start {expected.isoformat()}"
        print(f"PASS production Last-Modified {modified.isoformat()} follows deployment start")


if __name__ == "__main__":
    main()
