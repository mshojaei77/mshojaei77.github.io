"""Notify IndexNow of verified content and template URL changes after Pages deployment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
from pathlib import Path

HOST = "mshojaei77.github.io"
BASE = f"https://{HOST}"
KEY_FILE = Path("_data/discovery.json")
POST_PATH = re.compile(r"^_posts/[^/]+\.(?:md|markdown|html)$")
PERMALINK = re.compile(r"^permalink:\s*['\"]?([^\s'\"]+)['\"]?\s*$", re.M)
FRONT_MATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.S)
XML_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
PAGES_WORKFLOW_NAME = "pages-build-deployment"
STATE_BRANCH = "indexnow-state"
STATE_PATH = ".indexnow-checkpoint.json"
STATE_SCHEMA_VERSION = 1
ARTICLE_RENDER_DEPENDENCIES = {"_layouts/article.html", "_includes/related-articles.html"}
TOPIC_RENDER_DEPENDENCIES = {"_data/topics.json"}
AUTHOR_DATA_DEPENDENCIES = {"_data/authors.json"}
AUTHOR_PROFILE_DEPENDENCIES = {"_includes/home-seo.html", "_includes/profile-page-schema.html"}
HOME_RENDER_DEPENDENCIES = {"index.html", "fa/index.html", "_includes/home.html", "_data/home.json"}
STANDALONE_ROUTES = {
    "index.html": "/",
    "fa/index.html": "/fa/",
    "writing/index.html": "/writing/",
    "fa/writing/index.html": "/fa/writing/",
    "about.md": "/about.html",
    "fa/about/index.html": "/fa/about/",
    "roadmap.md": "/roadmap.html",
    "fa/roadmap/index.md": "/fa/roadmap/",
    "privacy.md": "/privacy/",
    "fa/privacy/index.md": "/fa/privacy/",
}
FULL_SITE_RENDER_DEPENDENCIES = {
    "_config.yml",
    "_data/languages.yml",
    "_includes/discovery-head.html",
    "_layouts/publication.html",
    "_includes/language-switch.html",
}


def git(*args: str, cwd: Path = Path(".")) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def source_at(revision: str, path: str, cwd: Path = Path(".")) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def permalink(source: str | None) -> str | None:
    if not source:
        return None
    match = FRONT_MATTER.match(source)
    if not match:
        return None
    found = PERMALINK.search(match.group(1))
    value = found.group(1) if found else None
    return value if value and value.startswith("/") else None


def paths_at(revision: str, directory: str, cwd: Path = Path(".")) -> list[str]:
    return [p for p in git("ls-tree", "-r", "--name-only", revision, "--", directory, cwd=cwd).splitlines() if p]


def route_map(revision: str, directory: str, cwd: Path = Path(".")) -> dict[str, str]:
    result = {}
    for path in paths_at(revision, directory, cwd):
        route = permalink(source_at(revision, path, cwd))
        if route:
            result[route] = path
    return result


def standalone_route(revision: str, path: str, cwd: Path = Path(".")) -> str | None:
    source = source_at(revision, path, cwd)
    return permalink(source) or STANDALONE_ROUTES.get(path)


def changed_files(before: str, after: str, cwd: Path = Path(".")) -> list[tuple[str, str]]:
    output = git("diff", "--name-status", "--no-renames", before, after, cwd=cwd)
    return [parts for row in output.splitlines() if len(parts := row.split("\t", 1)) == 2]


def changed_paths(before: str, after: str, cwd: Path = Path(".")) -> list[tuple[str, str]]:
    return [(status, path) for status, path in changed_files(before, after, cwd) if POST_PATH.match(path)]


def valid_indexnow_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.netloc == HOST and bool(parsed.path) and not parsed.query and not parsed.fragment


def checkpoint_payload(revision: str, status: str, known_urls: list[str], now: datetime | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Checkpoint revision must be a full commit SHA.")
    if status not in {"submission_accepted", "no_changes", "full_resync_accepted"}:
        raise ValueError(f"Invalid IndexNow checkpoint state: {status}")
    urls = sorted(set(known_urls))
    invalid = [url for url in urls if not valid_indexnow_url(url)]
    if invalid:
        raise ValueError(f"Checkpoint contains an ineligible URL: {invalid[0]}")
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "last_acknowledged_revision": revision,
        "last_acknowledged_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "notification_state": status,
        "known_canonical_urls": urls,
    }


def parse_checkpoint(raw: str) -> dict:
    value = json.loads(raw)
    if value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError("Unsupported IndexNow checkpoint schema.")
    if value.get("notification_state") not in {"submission_accepted", "no_changes", "full_resync_accepted"}:
        raise RuntimeError("IndexNow checkpoint has an invalid notification state.")
    revision = value.get("last_acknowledged_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("IndexNow checkpoint has an invalid acknowledged revision.")
    urls = value.get("known_canonical_urls")
    if not isinstance(urls, list) or any(not isinstance(url, str) or not valid_indexnow_url(url) for url in urls):
        raise RuntimeError("IndexNow checkpoint has an invalid canonical URL inventory.")
    if len(urls) != len(set(urls)):
        raise RuntimeError("IndexNow checkpoint contains duplicate URLs.")
    return value


def force_resync_urls(current_urls: list[str], previously_known_urls: list[str]) -> tuple[list[str], list[str]]:
    current, previous = set(current_urls), set(previously_known_urls)
    if any(not valid_indexnow_url(url) for url in current | previous):
        raise RuntimeError("Cannot reconcile ineligible URLs after a history rewrite.")
    return sorted(current), sorted(previous - current)


def valid_production_redirect(status: int, body: bytes, sitemap_urls: set[str]) -> bool:
    canonical = re.search(rb'<link\s+rel="canonical"\s+href="([^"]+)"', body, re.I)
    target = canonical.group(1).decode("utf-8") if canonical else ""
    lower = body.lower()
    return status == 200 and b"noindex" in lower and b'http-equiv="refresh"' in lower and target in sitemap_urls


def is_ancestor(ancestor: str, descendant: str, cwd: Path = Path(".")) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"Could not compare checkpoint revisions: {result.stderr.strip()}")


def has_commit(revision: str, cwd: Path = Path(".")) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def checkpoint_ref(cwd: Path = Path(".")) -> str | None:
    rows = git("ls-remote", "--heads", "origin", f"refs/heads/{STATE_BRANCH}", cwd=cwd).splitlines()
    if not rows:
        return None
    revision = rows[0].split()[0]
    git("fetch", "--no-tags", "origin", f"+refs/heads/{STATE_BRANCH}:refs/remotes/origin/{STATE_BRANCH}", cwd=cwd)
    return revision


def load_checkpoint(cwd: Path = Path(".")) -> dict | None:
    revision = checkpoint_ref(cwd)
    if not revision:
        return None
    raw = source_at(f"refs/remotes/origin/{STATE_BRANCH}", STATE_PATH, cwd)
    return parse_checkpoint(raw) if raw else None


def save_checkpoint(payload: dict, cwd: Path = Path(".")) -> None:
    revision = payload["last_acknowledged_revision"]
    ref = checkpoint_ref(cwd)
    if ref is None:
        # The state branch is created only after production verification and an
        # accepted/no-op reconciliation. It is separate from the Pages source.
        try:
            git("push", "origin", f"{revision}:refs/heads/{STATE_BRANCH}", cwd=cwd)
        except subprocess.CalledProcessError:
            if checkpoint_ref(cwd) is None:
                raise
        ref = checkpoint_ref(cwd)
    if not ref:
        raise RuntimeError("Could not create or fetch the IndexNow state branch.")

    existing = load_checkpoint(cwd)
    if existing:
        prior = existing["last_acknowledged_revision"]
        if prior == revision:
            return
        if is_ancestor(revision, prior, cwd):
            print(f"Checkpoint already acknowledges a newer revision {prior[:12]}; it was not rolled back.")
            return

    git("checkout", "--detach", f"refs/remotes/origin/{STATE_BRANCH}", cwd=cwd)
    (cwd / STATE_PATH).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    git("add", STATE_PATH, cwd=cwd)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd)
    if changed.returncode == 0:
        return
    if changed.returncode != 1:
        raise RuntimeError("Could not inspect the staged IndexNow checkpoint change.")
    git("config", "user.name", "github-actions[bot]", cwd=cwd)
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", cwd=cwd)
    git("commit", "-m", f"chore(discovery): acknowledge IndexNow revision {revision[:12]}", cwd=cwd)
    try:
        git("push", "origin", f"HEAD:refs/heads/{STATE_BRANCH}", cwd=cwd)
    except subprocess.CalledProcessError as exc:
        latest = load_checkpoint(cwd)
        if latest and (latest["last_acknowledged_revision"] == revision or is_ancestor(revision, latest["last_acknowledged_revision"], cwd)):
            print("A concurrent IndexNow worker already persisted this or a newer revision.")
            return
        raise RuntimeError("Could not persist the IndexNow checkpoint after successful submission.") from exc


def requires_full_site_notification(before: str, after: str, cwd: Path = Path(".")) -> bool:
    changed = {path for _, path in changed_files(before, after, cwd)}
    return bool(changed.intersection(FULL_SITE_RENDER_DEPENDENCIES))


def changed_urls(before: str, after: str, cwd: Path = Path(".")) -> tuple[list[str], list[str]]:
    new_posts = route_map(after, "_posts", cwd)
    retirements = route_map(after, "retired-posts", cwd)
    redirects = route_map(after, "redirects", cwd)
    changed = {path for _, path in changed_files(before, after, cwd)}
    updates: set[str] = set()
    deletions: set[str] = set()
    for status, path in changed_paths(before, after, cwd):
        old_route = permalink(source_at(before, path, cwd)) if status != "A" else None
        new_route = permalink(source_at(after, path, cwd)) if status != "D" else None
        if new_route and new_route in new_posts:
            updates.add(new_route)
        if old_route and old_route not in new_posts:
            if old_route in redirects:
                updates.add(old_route)
            elif old_route in retirements:
                deletions.add(old_route)
            else:
                deletions.add(old_route)
    changed_article_routes = updates | deletions
    for status, path in changed_files(before, after, cwd):
        if path not in STANDALONE_ROUTES:
            continue
        old_route = standalone_route(before, path, cwd) if status != "A" else None
        new_route = standalone_route(after, path, cwd) if status != "D" else None
        if new_route:
            updates.add(new_route)
        if old_route and old_route != new_route:
            if old_route in redirects:
                updates.add(old_route)
            else:
                deletions.add(old_route)
    if changed.intersection(HOME_RENDER_DEPENDENCIES):
        updates.update({"/", "/fa/"})
    # A published post also changes its language archive. Its homepage is
    # included only when the homepage actually links to that recent post.
    archives: set[str] = set()
    for route in changed_article_routes:
        archive = "/fa/writing/" if route.startswith("/fa/") else "/writing/"
        archives.add(archive)
    if changed.intersection(ARTICLE_RENDER_DEPENDENCIES | TOPIC_RENDER_DEPENDENCIES):
        updates.update(new_posts)
        archives.update("/fa/writing/" if route.startswith("/fa/") else "/writing/" for route in new_posts)
    if changed.intersection(TOPIC_RENDER_DEPENDENCIES):
        updates.update({"/", "/fa/"})
    if changed.intersection(AUTHOR_DATA_DEPENDENCIES):
        updates.update(set(new_posts) | {"/", "/fa/", "/about.html", "/fa/about/"})
    if changed.intersection(AUTHOR_PROFILE_DEPENDENCIES):
        updates.update({"/", "/fa/", "/about.html", "/fa/about/"})
    live_urls = {BASE + r for r in updates | archives}
    deleted_urls = {BASE + r for r in deletions}
    return sorted(live_urls), sorted(deleted_urls)


def api_json(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def pages_run_id_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise RuntimeError("Expected a GitHub Actions URL for the Pages deployment status.")
    match = re.fullmatch(r"/mshojaei77/mshojaei77\.github\.io/actions/runs/(\d+)(?:/job/\d+)?", parsed.path)
    if not match:
        raise RuntimeError("Could not identify the GitHub Pages workflow run from its deployment status URL.")
    return match.group(1)


def deployment_pair(repo: str, run_id: str, token: str, expected_sha: str = "") -> tuple[str, str] | None:
    if run_id:
        run = api_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}", token)
    else:
        pages = api_json(f"https://api.github.com/repos/{repo}/pages/builds/latest", token)
        pages_sha = pages.get("commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", pages_sha):
            raise RuntimeError("GitHub Pages did not report a valid deployed commit SHA.")
        workflows = api_json(f"https://api.github.com/repos/{repo}/actions/workflows?per_page=100", token).get("workflows", [])
        pages_workflow = next((item for item in workflows if item.get("name") == PAGES_WORKFLOW_NAME and item.get("state") == "active"), None)
        if not pages_workflow:
            raise RuntimeError("Could not identify the active GitHub Pages deployment workflow.")
        runs = api_json(
            f"https://api.github.com/repos/{repo}/actions/workflows/{pages_workflow['id']}/runs?branch=main&status=success&per_page=100",
            token,
        ).get("workflow_runs", [])
        matches = [item for item in runs if item.get("head_branch") == "main" and item.get("conclusion") == "success" and item.get("head_sha") == pages_sha]
        if not matches:
            raise RuntimeError(f"Could not match deployed Pages revision {pages_sha[:12]} to a successful Pages workflow run.")
        run = max(matches, key=lambda item: item.get("run_number", 0))
        run_id = str(run["id"])
    if run.get("status") != "completed" or run.get("conclusion") != "success" or run.get("head_branch") != "main":
        raise RuntimeError("IndexNow requires a successful Pages deployment from main.")
    pages = api_json(f"https://api.github.com/repos/{repo}/pages/builds/latest", token)
    if pages.get("commit") != run.get("head_sha"):
        print(f"A newer Pages deployment superseded {run.get('head_sha', '')[:12]}; this run is skipped.")
        return None
    if expected_sha and run.get("head_sha") != expected_sha:
        print(f"Pages revision {run.get('head_sha', '')[:12]} does not match this deployment event; this run is skipped.")
        return None
    workflow_id = run["workflow_id"]
    current_id = int(run["id"])
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/runs?branch=main&status=success&per_page=100"
    runs = api_json(url, token).get("workflow_runs", [])
    ordered = sorted((x for x in runs if x.get("conclusion") == "success" and x.get("head_branch") == "main"), key=lambda x: x.get("run_number", 0), reverse=True)
    current_index = next((index for index, item in enumerate(ordered) if int(item["id"]) == current_id), None)
    if current_index is None:
        raise RuntimeError("The selected Pages deployment is absent from the successful deployment history.")
    if current_index > 0:
        newer_sha = ordered[0].get("head_sha", "")
        if newer_sha != run.get("head_sha"):
            print("A newer successful Pages deployment exists; this run is skipped.")
            return None
    previous = next((item for item in ordered[current_index + 1:] if item.get("head_sha") != run.get("head_sha")), None)
    if not previous:
        print("No previous successful Pages deployment exists; seeding baseline without submissions.")
        return None
    return previous["head_sha"], run["head_sha"]


def fetch(url: str, attempts: int = 8) -> tuple[bytes, dict[str, str], int]:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "Publication-Production-Smoke/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}, response.status
        except urllib.error.HTTPError as exc:
            error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(5 * (attempt + 1), 20))
    raise RuntimeError(f"Production did not return a usable response for {url}: {error}")


def key_and_file() -> tuple[str, str]:
    config = json.loads(KEY_FILE.read_text(encoding="utf-8"))
    key = config["indexnow_key"]
    if not re.fullmatch(r"[A-Za-z0-9-]{8,128}", key):
        raise RuntimeError("Invalid IndexNow key format.")
    return key, f"/{key}.txt"


def validate_production(live_urls: list[str], deleted_urls: list[str], key: str, key_path: str) -> list[str]:
    nonce = str(int(time.time()))
    key_body, _, status = fetch(BASE + key_path + f"?smoke={nonce}")
    if status != 200 or key_body.decode("utf-8").strip() != key:
        raise RuntimeError("The IndexNow verification file is missing or mismatched in production.")
    sitemap_body, _, status = fetch(BASE + f"/sitemap.xml?smoke={nonce}")
    if status != 200:
        raise RuntimeError("Production sitemap is unavailable.")
    sitemap_root = ET.fromstring(sitemap_body)
    if sitemap_root.tag != "{http://www.sitemaps.org/schemas/sitemap/0.9}urlset":
        raise RuntimeError("Production sitemap does not have the expected urlset root.")
    sitemap_url_list = [node.text or "" for node in sitemap_root.findall("s:url/s:loc", XML_NS)]
    if not sitemap_url_list or len(sitemap_url_list) != len(set(sitemap_url_list)):
        raise RuntimeError("Production sitemap is empty or contains duplicate URLs.")
    for sitemap_url in sitemap_url_list:
        if not valid_indexnow_url(sitemap_url):
            raise RuntimeError(f"Production sitemap contains an ineligible URL: {sitemap_url}")
    sitemap_urls = set(sitemap_url_list)
    for url in live_urls:
        html, _, code = fetch(url + f"?smoke={nonce}")
        if url in sitemap_urls:
            if code != 200 or f'rel="canonical" href="{url}"'.encode() not in html:
                raise RuntimeError(f"Refusing to submit a URL that is not live and canonical in production: {url}")
        else:
            if not valid_production_redirect(code, html, sitemap_urls):
                raise RuntimeError(f"Refusing to submit an absent URL without a valid production redirect: {url}")
    for url in deleted_urls:
        # Studio retires removed posts as noindex pages. Submit the old canonical
        # URL only as a deletion notification so participating engines recrawl it.
        if url in sitemap_urls:
            raise RuntimeError(f"A retired URL remains in the production sitemap: {url}")
        body, _, code = fetch(url + f"?smoke={nonce}")
        if code not in {200, 404, 410}:
            raise RuntimeError(f"Unexpected production status for removed URL {url}: {code}")
        if code == 200 and b"noindex" not in body.lower():
            raise RuntimeError(f"Refusing a removal notice without noindex: {url}")
    return sorted(sitemap_urls)


def current_sitemap_urls() -> list[str]:
    body, _, status = fetch(BASE + f"/sitemap.xml?indexnow={int(time.time())}")
    if status != 200:
        raise RuntimeError(f"Production sitemap returned HTTP {status}.")
    root = ET.fromstring(body)
    urls = [node.text or "" for node in root.findall("s:url/s:loc", XML_NS)]
    if not urls or len(urls) != len(set(urls)):
        raise RuntimeError("Production sitemap is empty or contains duplicate URL entries.")
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != HOST or not parsed.path or parsed.query or parsed.fragment:
            raise RuntimeError(f"Refusing an ineligible sitemap URL: {url}")
    return sorted(urls)


def submit(key: str, urls: list[str]) -> int:
    body = json.dumps({"host": HOST, "key": key, "urlList": urls}).encode("utf-8")
    request = urllib.request.Request("https://api.indexnow.org/indexnow", data=body, headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                require_accepted_status(response.status)
                print(f"IndexNow accepted {len(urls)} changed URLs; HTTP {response.status}.")
                return response.status
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                delay = int(exc.headers.get("Retry-After", "60"))
                if attempt == 2 or delay > 600:
                    raise RuntimeError(f"IndexNow rate limited the request (HTTP 429; Retry-After {delay}s). Rerun after the stated delay.") from exc
                time.sleep(max(1, delay))
                continue
            if exc.code >= 500 and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"IndexNow rejected the changed URL batch (HTTP {exc.code}).") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"IndexNow request failed after bounded retries: {exc}") from exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("IndexNow submission did not complete.")


def require_accepted_status(status: int) -> None:
    if status == 202:
        raise RuntimeError("IndexNow key validation is still pending (HTTP 202); the checkpoint will not advance.")
    if status != 200:
        raise RuntimeError(f"Unexpected IndexNow response HTTP {status}; the checkpoint will not advance.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List changed article/archive URLs without accessing production or submitting.")
    parser.add_argument("--all-current", action="store_true", help="Submit every current canonical sitemap URL once after verifying the latest successful deployment.")
    parser.add_argument("--before", help="Previous deployment SHA for an explicit diff.")
    parser.add_argument("--after", help="Current deployment SHA for an explicit diff.")
    args = parser.parse_args()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("INDEXNOW_PAGES_RUN_ID", "")
    run_url = os.environ.get("INDEXNOW_PAGES_RUN_URL", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    expected_deployment_sha = os.environ.get("INDEXNOW_DEPLOYMENT_SHA", "")
    if not run_id and run_url:
        run_id = pages_run_id_from_url(run_url)
    if args.all_current:
        if args.dry_run or args.before or args.after:
            parser.error("--all-current is a one-time live submission mode and cannot be combined with --dry-run or explicit diffs.")
        if not (repo and token):
            parser.error("--all-current requires GITHUB_REPOSITORY and a GitHub token.")
        pair = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if pair is None:
            return 0
        before, after = pair
        updates, deletions = current_sitemap_urls(), []
        key, key_path = key_and_file()
        print(f"Initial rollout: {before[:12]} -> {after[:12]}; canonical sitemap URLs: {len(updates)}.")
        for url in updates:
            print(url)
        known_urls = validate_production(updates, deletions, key, key_path)
        latest = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if latest is None or latest[1] != after:
            print("A newer successful Pages deployment overtook this run; no URLs were submitted.")
            return 0
        submit(key, updates)
        latest = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if latest is None or latest[1] != after:
            print("A newer deployment arrived during submission; the checkpoint remains unchanged for recovery.")
            return 0
        save_checkpoint(checkpoint_payload(after, "submission_accepted", known_urls))
        return 0
    if args.before or args.after:
        if not (args.before and args.after):
            parser.error("--before and --after must be supplied together.")
        if not (re.fullmatch(r"[0-9a-f]{40}", args.before) and re.fullmatch(r"[0-9a-f]{40}", args.after)):
            parser.error("--before and --after must be full commit SHAs.")
        before, after = args.before, args.after
        if not args.dry_run:
            if not (repo and token):
                parser.error("Submitting an explicit deployment diff also requires GITHUB_REPOSITORY and GITHUB_TOKEN.")
            current = deployment_pair(repo, run_id, token, expected_deployment_sha)
            if current is None or current[1] != after:
                raise RuntimeError("Refusing to submit explicit URLs unless --after is the latest successful Pages deployment.")
        updates, deletions = changed_urls(before, after)
        print(f"Deployments: {before[:12]} -> {after[:12]}; changed URLs: {len(updates)} live, {len(deletions)} retired.")
        for url in updates + deletions:
            print(url)
        if args.dry_run:
            return 0
        key, key_path = key_and_file()
        known_urls = validate_production(updates, deletions, key, key_path)
        latest = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if latest is None or latest[1] != after:
            print("A newer successful Pages deployment overtook this run; no URLs were submitted.")
            return 0
        if updates or deletions:
            submit(key, updates + deletions)
        return 0
    else:
        if not (repo and token):
            parser.error("Set GITHUB_REPOSITORY and GITHUB_TOKEN in deployment mode.")
        pair = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if pair is None:
            return 0
        previous_deployment, after = pair
        checkpoint = load_checkpoint()
        if checkpoint:
            before = checkpoint["last_acknowledged_revision"]
            if before == after:
                print(f"Checkpoint already acknowledges deployed revision {after[:12]}; duplicate submission skipped.")
                return 0
            if has_commit(before) and has_commit(after) and is_ancestor(before, after):
                updates, deletions = changed_urls(before, after)
                full_site_change = requires_full_site_notification(before, after)
                if full_site_change:
                    print("A site-wide publication dependency changed; reconciling every current sitemap URL.")
                    updates = current_sitemap_urls()
            else:
                print("The acknowledged revision is missing or outside deployed history; performing a one-time full sitemap reconciliation.")
                updates, deletions = force_resync_urls(current_sitemap_urls(), checkpoint["known_canonical_urls"])
                full_site_change = True
        else:
            before = previous_deployment
            updates, deletions = changed_urls(before, after)
            full_site_change = requires_full_site_notification(before, after)
            if full_site_change:
                print("A site-wide publication dependency changed; reconciling every current sitemap URL.")
                updates = current_sitemap_urls()

    if args.dry_run:
        print(f"Deployments: {before[:12]} -> {after[:12]}; changed URLs: {len(updates)} live, {len(deletions)} retired.")
        for url in updates + deletions:
            print(url)
        return 0

    key, key_path = key_and_file()
    print(f"Deployments: {before[:12]} -> {after[:12]}; changed URLs: {len(updates)} live, {len(deletions)} retired.")
    for url in updates + deletions:
        print(url)
    known_urls = validate_production(updates, deletions, key, key_path)
    latest = deployment_pair(repo, run_id, token, expected_deployment_sha)
    if latest is None or latest[1] != after:
        print("A newer successful Pages deployment overtook this run; no URLs were submitted and the checkpoint remains unchanged.")
        return 0
    if updates or deletions:
        submit(key, updates + deletions)
        status = "full_resync_accepted" if checkpoint and full_site_change else "submission_accepted"
        latest = deployment_pair(repo, run_id, token, expected_deployment_sha)
        if latest is None or latest[1] != after:
            print("A newer deployment arrived during submission; the checkpoint remains unchanged for recovery.")
            return 0
    else:
        status = "no_changes"
        print("No eligible canonical URLs changed; production key and sitemap checks passed.")
    save_checkpoint(checkpoint_payload(after, status, known_urls))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"IndexNow failed: {error}", file=sys.stderr)
        raise SystemExit(1)
