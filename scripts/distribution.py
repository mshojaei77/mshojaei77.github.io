#!/usr/bin/env python3
"""Private, deterministic article distribution helpers.

Generated packages and the optional JSON registry live outside the public repo.
The script never publishes, contacts, or authenticates with an external platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SITE = "https://mshojaei77.github.io"
REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_VERSION = 1
PLATFORMS = {"medium", "devto", "virgool", "telegram", "telegraph", "linkedin", "hashnode", "github", "huggingface", "x", "reddit"}
# Campaign sources include distribution platforms and independent discovery
# channels. This list controls UTM labels only; it does not enable posting.
DISCOVERY_SOURCES = {"hackernews", "lobsters", "stackoverflow", "kagi", "marginalia", "oohdirectory"}
SOURCES = PLATFORMS | DISCOVERY_SOURCES
MEDIUMS = {"social", "referral", "syndication", "newsletter"}
RELATIONSHIPS = {
    "exact_copy", "substantially_equivalent", "older_edition", "newer_edition",
    "short_summary", "distinct_adaptation", "related_resource",
    "original_source_not_yet_migrated", "unknown_relationship",
}
PUBLICATION_STATES = {
    "not_planned", "planned", "draft_prepared", "external_draft_created",
    "published_publicly_observed", "published_owner_confirmed", "archived", "unknown",
}
CANONICAL_STATES = {
    "unverified", "correct", "points_to_external_copy", "conflict",
    "missing", "not_supported", "not_applicable", "not_applicable_distinct_content",
    "access_blocked", "unknown",
}
ALLOWED_HOSTS = {
    "medium": {"medium.com"},
    "devto": {"dev.to"},
    "virgool": {"virgool.ir"},
    "telegram": {"t.me"},
    "telegraph": {"telegra.ph"},
    "linkedin": {"linkedin.com", "www.linkedin.com"},
    "hashnode": {"hashnode.com", "hashnode.dev"},
    "github": {"github.com"},
    "huggingface": {"huggingface.co"},
    "x": {"x.com", "twitter.com"},
    "reddit": {"reddit.com", "www.reddit.com"},
}
FRONT_MATTER = re.compile(r"\A(?:\ufeff)?---\s*\r?\n(?P<header>[\s\S]*?)\r?\n---(?:\r?\n|$)")
FIELD = re.compile(r"^([A-Za-z_][\w-]*):[ \t]*(.*)$", re.M)
ROUTE = re.compile(r"^/(?:fa/)?writing/[a-z0-9]+(?:-[a-z0-9]+)*/$")
SLUG = re.compile(r"/writing/([a-z0-9]+(?:-[a-z0-9]+)*)/$")
LIQUID = re.compile(r"{%[-]?\s*(?:include|assign|capture|if|unless|for|case|comment|raw|render|highlight|link|post_url|seo|feed_meta|increment|decrement)\b|{{[-]?\s*(?:site|page|content|post|paginator)\.", re.I)
HTML_UNSAFE = re.compile(r"<\s*script\b|\son[a-z]+\s*=|javascript\s*:", re.I)


class DistributionError(ValueError):
    """Invalid URL, record, or article export input."""


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def valid_blog_destination(value: str, *, canonical_only: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc.lower() == "mshojaei77.github.io"
        and not parsed.username and not parsed.password
        and (not canonical_only or (not parsed.query and not parsed.fragment))
        and bool(ROUTE.fullmatch(parsed.path))
    )


def valid_article_url(value: str) -> bool:
    return valid_blog_destination(value, canonical_only=True)


def validate_external_url(value: str, platform: str) -> str:
    if platform not in PLATFORMS:
        raise DistributionError(f"Unsupported platform: {platform}")
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
    except (AttributeError, ValueError):
        raise DistributionError("External URL must be an absolute HTTPS URL.") from None
    if parsed.scheme != "https" or not host or host not in ALLOWED_HOSTS[platform]:
        raise DistributionError(f"URL host does not match platform {platform}.")
    if parsed.username or parsed.password:
        raise DistributionError("URLs containing credentials are not accepted.")
    if any(key.lower().startswith("utm_") for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise DistributionError("Store the clean external URL without tracking parameters.")
    return value


def tracking_url(article_url: str, source: str, medium: str, campaign: str, content: str = "") -> str:
    if not valid_blog_destination(article_url):
        raise DistributionError("Tracking links must target a canonical blog article URL.")
    source = source.strip().lower()
    medium = medium.strip().lower()
    campaign = campaign.strip().lower()
    content = content.strip()
    if source not in SOURCES:
        raise DistributionError("Choose a supported distribution source.")
    if medium not in MEDIUMS:
        raise DistributionError("Choose a supported campaign medium.")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?", campaign):
        raise DistributionError("Campaign must be a short lowercase ASCII slug.")
    if content and (len(content) > 100 or any(ord(c) < 32 for c in content)):
        raise DistributionError("Campaign content must be at most 100 printable characters.")
    parsed = urlsplit(article_url)
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    query.extend((("utm_source", source), ("utm_medium", medium), ("utm_campaign", campaign)))
    if content:
        query.append(("utm_content", content))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def record_key(record: dict) -> tuple[str, str, str]:
    return record.get("article_url") or "", record["platform"], record["external_url"]


def validate_registry(registry: object) -> dict:
    if not isinstance(registry, dict) or registry.get("schema_version") != REGISTRY_VERSION:
        raise DistributionError(f"Registry schema_version must be {REGISTRY_VERSION}.")
    records = registry.get("records")
    if not isinstance(records, list):
        raise DistributionError("Registry records must be an array.")
    seen: set[tuple[str, str, str]] = set()
    clean = []
    required = {"article_id", "platform", "external_url", "relationship", "publication_status", "canonical_status"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise DistributionError(f"Record {index} is missing required fields.")
        article_url = record.get("article_url")
        if article_url is not None and not valid_article_url(article_url):
            raise DistributionError(f"Record {index} has a non-canonical blog article URL.")
        platform = record["platform"]
        validate_external_url(record["external_url"], platform)
        if record["relationship"] not in RELATIONSHIPS:
            raise DistributionError(f"Record {index} has an unknown relationship.")
        if record["publication_status"] not in PUBLICATION_STATES:
            raise DistributionError(f"Record {index} has an unknown publication status.")
        if record["canonical_status"] not in CANONICAL_STATES:
            raise DistributionError(f"Record {index} has an unknown canonical status.")
        if not isinstance(record["article_id"], str) or not re.fullmatch(r"(?:(?:en|fa)-|external-(?:medium|devto|virgool|telegram|telegraph)-)[a-z0-9]+(?:-[a-z0-9]+)*", record["article_id"]):
            raise DistributionError(f"Record {index} has an invalid article_id.")
        if article_url is None and record["relationship"] not in {"original_source_not_yet_migrated", "unknown_relationship", "related_resource"}:
            raise DistributionError(f"Record {index} needs a blog URL or an unmapped relationship.")
        key = record_key(record)
        if key in seen:
            raise DistributionError(f"Duplicate registry record: {record['article_id']} / {platform}.")
        seen.add(key)
        optional = {
            "article_url": article_url,
            "article_status": record.get("article_status", "active" if article_url else "not_migrated"),
            "last_verified_at": record.get("last_verified_at"),
            "last_observed_at": record.get("last_observed_at"),
            "evidence": record.get("evidence", ""),
            "content_sha256": record.get("content_sha256", ""),
            "updated_at": record.get("updated_at", ""),
        }
        for name, value in optional.items():
            if value is not None and not isinstance(value, str):
                raise DistributionError(f"Record {index} field {name} must be a string or null.")
        if optional["article_status"] not in {"active", "retired", "not_migrated", "unknown"}:
            raise DistributionError(f"Record {index} has an unknown article_status.")
        if optional["content_sha256"] and not re.fullmatch(r"[a-f0-9]{64}", optional["content_sha256"]):
            raise DistributionError(f"Record {index} content_sha256 must be a SHA-256 hex digest.")
        clean.append({**record, **optional})
    return {"schema_version": REGISTRY_VERSION, "records": clean}


_PUBLICATION_RANK = {"unknown": 0, "not_planned": 1, "planned": 2, "draft_prepared": 3, "external_draft_created": 4, "published_owner_confirmed": 5, "archived": 6}
_PUBLICATION_RANK["published_publicly_observed"] = 4
_CANONICAL_RANK = {"unknown": 0, "unverified": 1, "access_blocked": 1, "missing": 2, "points_to_external_copy": 2, "conflict": 2, "not_supported": 3, "not_applicable": 3, "not_applicable_distinct_content": 3, "correct": 4}


def merge_registries(current: object, incoming: object) -> dict:
    left, right = validate_registry(current), validate_registry(incoming)
    merged = {record_key(record): dict(record) for record in left["records"]}
    for record in right["records"]:
        key = record_key(record)
        if key not in merged:
            merged[key] = dict(record)
            continue
        old = merged[key]
        new = dict(old)
        for field in ("relationship", "article_id", "content_sha256"):
            if record.get(field):
                new[field] = record[field]
        if _PUBLICATION_RANK[record["publication_status"]] >= _PUBLICATION_RANK[old["publication_status"]]:
            new["publication_status"] = record["publication_status"]
        if _CANONICAL_RANK[record["canonical_status"]] >= _CANONICAL_RANK[old["canonical_status"]]:
            new["canonical_status"] = record["canonical_status"]
            new["last_verified_at"] = record.get("last_verified_at") or old.get("last_verified_at")
        else:
            new["last_verified_at"] = old.get("last_verified_at")
        new["evidence"] = record.get("evidence") or old.get("evidence", "")
        new["updated_at"] = now_utc()
        merged[key] = new
    return validate_registry({"schema_version": REGISTRY_VERSION, "records": list(merged.values())})


def default_registry_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "MohammadShojaei" / "PublicationOperations" / "distribution-registry.json"


def read_front_matter(source: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(source)
    if not match:
        raise DistributionError("Article source must have Jekyll front matter.")
    fields: dict[str, str] = {}
    for item in FIELD.finditer(match.group("header")):
        key, value = item.group(1), item.group(2).strip()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise DistributionError(f"Unsupported JSON-quoted front matter field {key}: {error}") from error
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("''", "'")
        fields[key] = value
    return fields, source[match.end():]


def validate_post_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    posts_root = (REPO_ROOT / "_posts").resolve()
    if not is_inside(candidate, posts_root) or candidate.suffix.lower() not in {".md", ".markdown", ".html"}:
        raise DistributionError("Only article sources inside _posts/ can be exported.")
    if not candidate.is_file():
        raise DistributionError("Article source does not exist.")
    return candidate


def article_details(path: str | Path) -> dict:
    article_path = validate_post_path(path)
    source = article_path.read_text(encoding="utf-8")
    fields, body = read_front_matter(source)
    lang = fields.get("lang", "")
    route = fields.get("permalink", "")
    title = fields.get("title", "")
    if lang not in {"en", "fa"} or not title or not valid_article_url(SITE + route):
        raise DistributionError("Article needs a valid language, title, and stable blog permalink.")
    article_id = lang + "-" + SLUG.search(route).group(1)
    return {
        "path": article_path,
        "title": title,
        "description": fields.get("description", ""),
        "lang": lang,
        "route": route,
        "url": SITE + route,
        "article_id": article_id,
        "date": fields.get("date", ""),
        "image": fields.get("image", ""),
        "extension": article_path.suffix.lower(),
        "body": body,
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }


def _without_code(source: str) -> str:
    source = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", " ", source)
    return re.sub(r"`+[^`]*`+", " ", source)


def clean_body(article: dict) -> str:
    body = article["body"]
    if LIQUID.search(_without_code(body)):
        raise DistributionError("This article contains Jekyll Liquid outside code examples. Use the live article or render the component manually before exporting.")
    if article["extension"] == ".html" and HTML_UNSAFE.search(body):
        raise DistributionError("HTML export contains script, event-handler, or javascript: markup; review it manually before sharing.")
    body = re.sub(r"(?P<prefix>[\[(])/(?P<path>(?!/)[^\s)\]]*)", lambda match: match.group("prefix") + SITE + "/" + match.group("path"), body)
    body = re.sub(r"(?P<prefix>\b(?:href|src)=[\"'])/(?P<path>(?!/)[^\"']*)", lambda match: match.group("prefix") + SITE + "/" + match.group("path"), body, flags=re.I)
    return body.strip() + "\n"


def _markdown_title(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def export_files(article: dict, platform: str, live_confirmed: bool) -> dict[str, str]:
    if platform not in {"medium", "devto", "telegram", "virgool", "linkedin"}:
        raise DistributionError("Supported packages: medium, devto, telegram, virgool, linkedin.")
    full_copy = platform in {"medium", "devto"}
    if full_copy and not live_confirmed:
        raise DistributionError("Confirm that the production article returned HTTP 200 with its expected self-canonical and sitemap entry before preparing a full-copy package.")
    slug = SLUG.search(article["route"]).group(1)
    common = (
        f"# {article['title']}\n\n"
        f"Maintained blog edition: {article['url']}\n\n"
        f"Production readiness was owner-confirmed for this package on {now_utc()}; this flag does not independently verify HTTP, canonical, indexability, sitemap membership, or search indexing.\n"
    )
    if platform == "medium":
        body = clean_body(article)
        return {
            "article-manifest.json": json.dumps({
                "schema_version": 1, "article_id": article["article_id"], "language": article["lang"],
                "blog_url": article["url"], "platform": platform, "source_sha256": article["sha256"],
                "readiness_owner_confirmed": True, "generated_at": now_utc(),
            }, ensure_ascii=False, indent=2) + "\n",
            "medium.md": f"# {article['title']}\n\n{body}",
            "instructions.md": common + "Use Medium's official Import a story flow with the blog URL above. Verify the published story's rendered canonical before treating it as a full copy. The Markdown file is a fallback; it has no Medium API credentials or publishing automation.\n",
        }
    if platform == "devto":
        body = clean_body(article)
        header = ["---", f"title: {_markdown_title(article['title'])}", "published: false", f"canonical_url: {article['url']}"]
        if article["description"]:
            header.append(f"description: {_markdown_title(article['description'])}")
        if article["image"].startswith("/"):
            header.append(f"cover_image: {SITE + article['image']}")
        header.append("---")
        return {
            "article-manifest.json": json.dumps({
                "schema_version": 1, "article_id": article["article_id"], "language": article["lang"],
                "blog_url": article["url"], "platform": platform, "source_sha256": article["sha256"],
                "readiness_owner_confirmed": True, "generated_at": now_utc(),
            }, ensure_ascii=False, indent=2) + "\n",
            "devto.md": "\n".join(header) + "\n\n" + body,
            "instructions.md": common + "This is a DEV draft export. Keep `published: false`; review the content and canonical field in DEV before publishing.\n",
        }
    if platform == "telegram":
        return {"telegram-template.md": common + "## خلاصهٔ کوتاه\n\n[خلاصه را با لحن خودتان بنویسید.]\n\n## نکته‌های اصلی\n\n- [نکتهٔ اول]\n- [نکتهٔ دوم]\n\nاین فایل متن مقاله را به‌صورت خودکار خلاصه نمی‌کند.\n"}
    if platform == "virgool":
        title = article["title"] if article["lang"] == "fa" else "عنوان فارسی پیشنهادی"
        return {"virgool-template.md": f"# {title}\n\nمقالهٔ کامل وبلاگ: {article['url']}\n\n## زاویهٔ مستقل برای ویرگول\n\n[مسئلهٔ مشخصی را انتخاب کنید و یک نسخهٔ فارسی با هدفی متفاوت بنویسید. متن مقاله را خودکار بازنویسی نکنید.]\n\n## نکته‌های فنی قابل استفاده\n\n- [نکتهٔ مستند اول]\n- [نکتهٔ مستند دوم]\n"}
    return {"linkedin-template.md": common + f"## مسئلهٔ مهندسی\n\n[یک مسئلهٔ مشخص از مقاله را توضیح دهید.]\n\n## نکتهٔ فنی\n\n[یک نکتهٔ مستند را با محدودیت‌هایش بیان کنید.]\n\n## پیوند\n\n[مقالهٔ کامل]({article['url']})\n\nاین فایل تجربهٔ شخصی یا benchmark تولید نمی‌کند.\n"}


def safe_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if is_inside(output, REPO_ROOT):
        raise DistributionError("Distribution packages must be written outside the public repository.")
    return output


def cmd_tracking(args: argparse.Namespace) -> int:
    print(tracking_url(args.url, args.source, args.medium, args.campaign, args.content))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    article = article_details(args.article)
    files = export_files(article, args.platform, args.live_confirmed)
    slug = SLUG.search(article["route"]).group(1)
    root = safe_output_dir(args.output_dir or (default_registry_path().parent / "exports"))
    dest = root / slug / args.platform
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (dest / name).write_text(content, encoding="utf-8", newline="\n")
    print(f"Prepared {len(files)} file(s) at {dest}")
    return 0


def cmd_registry_validate(args: argparse.Namespace) -> int:
    registry = json.loads(Path(args.file).read_text(encoding="utf-8"))
    clean = validate_registry(registry)
    print(f"Valid schema {REGISTRY_VERSION} registry with {len(clean['records'])} record(s).")
    return 0


def cmd_registry_merge(args: argparse.Namespace) -> int:
    current_path = safe_output_dir(Path(args.file).expanduser())
    incoming_path = Path(args.import_file).expanduser()
    current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else {"schema_version": REGISTRY_VERSION, "records": []}
    incoming = json.loads(incoming_path.read_text(encoding="utf-8"))
    result = merge_registries(current, incoming)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Merged registry to {current_path} ({len(result['records'])} record(s)).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tracking = sub.add_parser("tracking", help="build a campaign URL for a canonical blog article")
    tracking.add_argument("--url", required=True)
    tracking.add_argument("--source", required=True, choices=sorted(SOURCES))
    tracking.add_argument("--medium", required=True, choices=sorted(MEDIUMS))
    tracking.add_argument("--campaign", required=True)
    tracking.add_argument("--content", default="")
    tracking.set_defaults(run=cmd_tracking)
    export = sub.add_parser("export", help="prepare a private, non-publishing platform package")
    export.add_argument("--article", required=True)
    export.add_argument("--platform", required=True, choices=("medium", "devto", "telegram", "virgool", "linkedin"))
    export.add_argument("--live-confirmed", action="store_true", help="owner has checked the current production URL, self-canonical, indexability, and sitemap")
    export.add_argument("--output-dir", help="private output directory (must be outside the public repository)")
    export.set_defaults(run=cmd_export)
    registry = sub.add_parser("registry", help="validate or safely merge a private registry JSON file")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)
    validate = registry_sub.add_parser("validate")
    validate.add_argument("--file", required=True)
    validate.set_defaults(run=cmd_registry_validate)
    merge = registry_sub.add_parser("merge")
    merge.add_argument("--file", default=str(default_registry_path()))
    merge.add_argument("--import-file", required=True)
    merge.set_defaults(run=cmd_registry_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (DistributionError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
