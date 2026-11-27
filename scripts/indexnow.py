"""Notify IndexNow of article and language archive changes after Pages deployment."""
from __future__ import annotations

import argparse
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


def git(*args: str, cwd: Path = Path(".")) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def source_at(revision: str, path: str, cwd: Path = Path(".")) -> str | None:
    try:
        return git("show", f"{revision}:{path}", cwd=cwd)
    except subprocess.CalledProcessError:
        return None


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


def changed_paths(before: str, after: str, cwd: Path = Path(".")) -> list[tuple[str, str]]:
    output = git("diff", "--name-status", "--no-renames", before, after, "--", "_posts", cwd=cwd)
    changes = []
    for row in output.splitlines():
        parts = row.split("\t", 1)
        if len(parts) == 2 and POST_PATH.match(parts[1]):
            changes.append((parts[0], parts[1]))
    return changes


def changed_urls(before: str, after: str, cwd: Path = Path(".")) -> tuple[list[str], list[str]]:
    new_posts = route_map(after, "_posts", cwd)
    retirements = route_map(after, "retired-posts", cwd)
    updates: set[str] = set()
    deletions: set[str] = set()
    for status, path in changed_paths(before, after, cwd):
        old_route = permalink(source_at(before, path, cwd)) if status != "A" else None
        new_route = permalink(source_at(after, path, cwd)) if status != "D" else None
        if new_route and new_route in new_posts:
            updates.add(new_route)
        if old_route and old_route not in new_posts and old_route in retirements:
            deletions.add(old_route)
    # A published post also changes its language archive. Its homepage is
    # included only when the homepage actually links to that recent post.
    archives: set[str] = set()
    for route in updates | deletions:
        archive = "/fa/writing/" if route.startswith("/fa/") else "/writing/"
        archives.add(archive)
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


def deployment_pair(repo: str, run_id: str, token: str) -> tuple[str, str] | None:
    run = api_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}", token)
    if run.get("status") != "completed" or run.get("conclusion") != "success" or run.get("head_branch") != "main":
        raise RuntimeError("IndexNow requires a successful Pages deployment from main.")
    workflow_id = run["workflow_id"]
    current_id = int(run["id"])
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_id}/runs?branch=main&status=success&per_page=100"
    runs = api_json(url, token).get("workflow_runs", [])
    ordered = sorted((x for x in runs if x.get("conclusion") == "success" and x.get("head_branch") == "main"), key=lambda x: x.get("run_number", 0), reverse=True)
    if not ordered or int(ordered[0]["id"]) != current_id:
        print("A newer successful Pages deployment exists; this run is skipped.")
        return None
    if len(ordered) < 2:
        print("No previous successful Pages deployment exists; seeding baseline without submissions.")
        return None
    return ordered[1]["head_sha"], run["head_sha"]


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


def validate_production(live_urls: list[str], deleted_urls: list[str], key: str, key_path: str) -> None:
    nonce = str(int(time.time()))
    key_body, _, status = fetch(BASE + key_path + f"?smoke={nonce}")
    if status != 200 or key_body.decode("utf-8").strip() != key:
        raise RuntimeError("The IndexNow verification file is missing or mismatched in production.")
    sitemap_body, _, status = fetch(BASE + f"/sitemap.xml?smoke={nonce}")
    if status != 200:
        raise RuntimeError("Production sitemap is unavailable.")
    sitemap_urls = {node.text for node in ET.fromstring(sitemap_body).findall("s:url/s:loc", XML_NS)}
    for url in live_urls:
        if url not in sitemap_urls:
            raise RuntimeError(f"Refusing to submit a URL absent from the production sitemap: {url}")
        html, _, code = fetch(url + f"?smoke={nonce}")
        if code != 200 or f'rel="canonical" href="{url}"'.encode() not in html:
            raise RuntimeError(f"Refusing to submit a URL that is not live and canonical in production: {url}")
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
        if not (repo and run_id and token):
            parser.error("--all-current requires a successful Pages run identity and GitHub token.")
        pair = deployment_pair(repo, run_id, token)
        if pair is None:
            return 0
        before, after = pair
        if expected_deployment_sha and after != expected_deployment_sha:
            raise RuntimeError("Pages workflow SHA does not match the successful deployment status event.")
        updates, deletions = current_sitemap_urls(), []
        key, key_path = key_and_file()
        print(f"Initial rollout: {before[:12]} -> {after[:12]}; canonical sitemap URLs: {len(updates)}.")
        for url in updates:
            print(url)
        validate_production(updates, deletions, key, key_path)
        latest = deployment_pair(repo, run_id, token)
        if latest is None or latest[1] != after:
            print("A newer successful Pages deployment overtook this run; no URLs were submitted.")
            return 0
        submit(key, updates)
        Path(".indexnow-run-marker").write_text(after, encoding="ascii")
        return 0
    if args.before or args.after:
        if not (args.before and args.after):
            parser.error("--before and --after must be supplied together.")
        if not (re.fullmatch(r"[0-9a-f]{40}", args.before) and re.fullmatch(r"[0-9a-f]{40}", args.after)):
            parser.error("--before and --after must be full commit SHAs.")
        before, after = args.before, args.after
        if not args.dry_run:
            if not (repo and run_id and token):
                parser.error("Submitting an explicit deployment diff also requires GITHUB_REPOSITORY, INDEXNOW_PAGES_RUN_ID and GITHUB_TOKEN.")
            current = deployment_pair(repo, run_id, token)
            if current is None or current[1] != after:
                raise RuntimeError("Refusing to submit explicit URLs unless --after is the latest successful Pages deployment.")
    else:
        if not (repo and run_id and token):
            parser.error("Set GITHUB_REPOSITORY, INDEXNOW_PAGES_RUN_ID and GITHUB_TOKEN in deployment mode.")
        pair = deployment_pair(repo, run_id, token)
        if pair is None:
            return 0
        before, after = pair
        if expected_deployment_sha and after != expected_deployment_sha:
            raise RuntimeError("Pages workflow SHA does not match the successful deployment status event.")
    updates, deletions = changed_urls(before, after)
    key, key_path = key_and_file()
    print(f"Deployments: {before[:12]} -> {after[:12]}; changed URLs: {len(updates)} live, {len(deletions)} retired.")
    for url in updates + deletions:
        print(url)
    if args.dry_run:
        return 0
    if not updates and not deletions:
        print("No published article URLs changed; no IndexNow request sent.")
        Path(".indexnow-run-marker").write_text(after, encoding="ascii")
        return 0
    validate_production(updates, deletions, key, key_path)
    # Recheck the latest Pages run immediately before submitting, so a deploy
    # that overtook this job cannot send notifications for a stale revision.
    latest = deployment_pair(repo, run_id, token)
    if latest is None or latest[1] != after:
        print("A newer successful Pages deployment overtook this job; no URLs were submitted.")
        return 0
    submit(key, updates + deletions)
    Path(".indexnow-run-marker").write_text(after, encoding="ascii")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"IndexNow failed: {error}", file=sys.stderr)
        raise SystemExit(1)
