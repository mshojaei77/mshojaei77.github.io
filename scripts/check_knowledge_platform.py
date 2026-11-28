"""Validate knowledge-platform references against the published Jekyll sources."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors: list[str] = []


def load_json(name: str):
    try:
        return json.loads((ROOT / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{name}: {exc}")
        return {}


def front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"---\s*\r?\n(.*?)\r?\n---", text, re.S)
    if not match:
        return {}
    return {key: value.strip().strip('"\'') for key, value in re.findall(r"^([A-Za-z0-9_]+):\s*(.*)$", match.group(1), re.M)}


posts: dict[tuple[str, str], tuple[str, dict[str, str]]] = {}
kind_counts: dict[str, int] = {}
for post_file in sorted((ROOT / "_posts").glob("*")):
    meta = front_matter(post_file)
    route = meta.get("permalink", "").strip("/")
    lang = meta.get("lang", "")
    kind = meta.get("article_kind", "")
    if not route or lang not in {"en", "fa"}:
        continue
    slug = route.split("/")[-1]
    posts[(lang, slug)] = ("/" + route + "/", meta)
    kind_counts[kind] = kind_counts.get(kind, 0) + 1

if len(posts) != 57:
    errors.append(f"Expected 57 active post routes; found {len(posts)}")
allowed_kinds = {"guide", "essay", "tutorial", "research-summary"}
if set(kind_counts) - allowed_kinds or "" in kind_counts:
    errors.append(f"Unexpected or missing article_kind values: {kind_counts}")
kind_source = (ROOT / "_data/article_kinds.yml").read_text(encoding="utf-8")
kind_ids = re.findall(r"^- id:\s*([a-z-]+)\s*$", kind_source, re.M)
if set(kind_ids) != allowed_kinds or len(kind_ids) != len(set(kind_ids)):
    errors.append(f"Article type labels do not match supported article kinds: {kind_ids}")

topics = load_json("_data/topics.json")
clusters = topics.get("clusters", {})
if len(clusters) != 8:
    errors.append(f"Expected the existing eight topic clusters; found {len(clusters)}")
for cluster_id, cluster in clusters.items():
    for lang in ("en", "fa"):
        if not cluster.get("labels", {}).get(lang):
            errors.append(f"{cluster_id}: missing {lang} label")
        if not cluster.get("descriptions", {}).get(lang):
            errors.append(f"{cluster_id}: missing {lang} domain description")
        if not cluster.get("concepts", {}).get(lang) or not cluster.get("failure_modes", {}).get(lang):
            errors.append(f"{cluster_id}: missing {lang} concepts or failure modes")
    if len(cluster.get("articles", [])) < 3:
        errors.append(f"{cluster_id}: domain needs at least three related article references")
    for slug in cluster.get("articles", []):
        if not any(post_slug == slug for _, post_slug in posts):
            errors.append(f"{cluster_id}: unknown article slug {slug}")

paths_doc = load_json("_data/learning_paths.json")
paths = paths_doc.get("paths", [])
path_ids = [path.get("id") for path in paths]
if len(path_ids) != 5 or len(path_ids) != len(set(path_ids)):
    errors.append("Learning paths must contain five unique IDs")
for path in paths:
    path_id = path.get("id", "")
    for lang in ("en", "fa"):
        source = ROOT / ("learn" if lang == "en" else "fa/learn") / f"{path_id}.md"
        meta = front_matter(source) if source.is_file() else {}
        if meta.get("path_id") != path_id or meta.get("lang") != lang:
            errors.append(f"{path_id}: missing or mismatched {lang} route source")
        if not path.get("title", {}).get(lang) or not path.get("description", {}).get(lang):
            errors.append(f"{path_id}: missing {lang} title or description")
        if not path.get("objectives", {}).get(lang) or not path.get("completion", {}).get(lang):
            errors.append(f"{path_id}: missing {lang} objectives or completion criteria")
    module_ids = [module.get("id") for module in path.get("modules", [])]
    if not module_ids or len(module_ids) != len(set(module_ids)):
        errors.append(f"{path_id}: modules must be non-empty with unique IDs")
    for module in path.get("modules", []):
        if not module.get("articles"):
            errors.append(f"{path_id}/{module.get('id')}: module has no reading")
        if not module.get("activity", {}).get("en") or not module.get("activity", {}).get("fa"):
            errors.append(f"{path_id}/{module.get('id')}: missing bilingual practice activity")
        for article in module.get("articles", []):
            if (article.get("lang"), article.get("slug")) not in posts:
                errors.append(f"{path_id}/{module.get('id')}: unknown article {article}")

tools = load_json("_data/playground_tools.json")
tool_ids = [tool.get("id") for tool in tools] if isinstance(tools, list) else []
if len(tool_ids) != 4 or len(tool_ids) != len(set(tool_ids)):
    errors.append("The first playground release must have four unique tools")
for tool in tools if isinstance(tools, list) else []:
    script = ROOT / tool.get("script", "").lstrip("/")
    if not script.is_file():
        errors.append(f"{tool.get('id')}: missing tool script {tool.get('script')}")
    for lang in ("en", "fa"):
        if not tool.get("title", {}).get(lang) or not tool.get("description", {}).get(lang):
            errors.append(f"{tool.get('id')}: missing {lang} title or description")
        route = ROOT / ("playground" if lang == "en" else "fa/playground") / f"{tool.get('id')}.md"
        meta = front_matter(route) if route.is_file() else {}
        if meta.get("tool_id") != tool.get("id") or meta.get("lang") != lang:
            errors.append(f"{tool.get('id')}: missing or mismatched {lang} route source")

artifacts = load_json("_data/research_artifacts.json")
if not isinstance(artifacts, list) or len(artifacts) < 3:
    errors.append("Research Lab must describe at least three existing public artifacts")
else:
    artifact_ids = [artifact.get("id") for artifact in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("Research artifact IDs must be unique")
    for artifact in artifacts:
        if not artifact.get("url", "").startswith("https://") or not artifact.get("verified_on"):
            errors.append(f"{artifact.get('id')}: missing secure source URL or verification date")
        for lang in ("en", "fa"):
            if not artifact.get("description", {}).get(lang) or not artifact.get("limitations", {}).get(lang):
                errors.append(f"{artifact.get('id')}: missing {lang} artifact description or limitations")
        for reference in artifact.get("articles", []):
            if (reference.get("lang"), reference.get("slug")) not in posts:
                errors.append(f"{artifact.get('id')}: unknown related article {reference}")

projects = load_json("_data/home.json").get("selected_work", [])
case_slugs = []
for project in projects:
    case = project.get("case_study")
    if not case:
        errors.append(f"{project.get('repo')}: missing evidence-backed case-study metadata")
        continue
    slug = case.get("slug", "")
    case_slugs.append(slug)
    for lang in ("en", "fa"):
        for key in ("implementation", "architecture", "tradeoffs", "limitations"):
            if not case.get(key, {}).get(lang):
                errors.append(f"{project.get('repo')}/{key}: missing {lang} case-study text")
        route = ROOT / ("work" if lang == "en" else "fa/work") / f"{slug}.md"
        meta = front_matter(route) if route.is_file() else {}
        if meta.get("case_study_id") != slug or meta.get("lang") != lang:
            errors.append(f"{slug}: missing or mismatched {lang} case-study route")
    for domain_id in case.get("domains", []):
        if domain_id not in clusters:
            errors.append(f"{slug}: unknown knowledge domain {domain_id}")
if len(case_slugs) != 4 or len(case_slugs) != len(set(case_slugs)):
    errors.append("Expected one unique case study for each of the four existing selected projects")

series_posts = [meta for (_, _), (_, meta) in posts.items() if meta.get("series_id")]
for meta in series_posts:
    if not meta.get("series_id") or not meta.get("series_number") or not meta.get("series_source_url", "").startswith("https://"):
        errors.append("Reading-series article is missing ID, installment number, or source URL")
for lang, route in (("en", "/reading/"), ("fa", "/fa/reading/")):
    source = ROOT / ("reading/index.md" if lang == "en" else "fa/reading/index.md")
    page_meta = front_matter(source) if source.is_file() else {}
    if page_meta.get("lang") != lang or page_meta.get("permalink") != route:
        errors.append(f"Reading index has invalid {lang} route metadata")

if errors:
    print("Knowledge platform validation failed:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print(f"Validated {len(posts)} classified article routes, {len(clusters)} knowledge domains, {len(paths)} bilingual learning paths, and {len(tool_ids)} bilingual playground tools.")
