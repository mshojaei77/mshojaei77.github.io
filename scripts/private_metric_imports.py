"""Normalize explicitly described, private webmaster and GoatCounter exports.

This module performs no network requests and never persists provider credentials.
Inputs and outputs must stay outside the public repository. Imported rows remain
provider-scoped; overlapping date ranges are reported but never summed together.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"google_search_console", "bing_webmaster", "bing_ai_performance", "goatcounter"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KNOWN_GSC_DIMENSIONS = {"query", "page", "date", "country", "device", "searchAppearance"}
AI_REFERRER_HOSTS = {
    "chatgpt.com": "ChatGPT",
    "chat.openai.com": "ChatGPT",
    "perplexity.ai": "Perplexity",
    "copilot.microsoft.com": "Microsoft Copilot",
    "gemini.google.com": "Gemini",
    "claude.ai": "Claude",
}


class ImportErrorDetail(ValueError):
    """An input export is ambiguous, malformed, or not a supported schema."""


def _private_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents:
        raise ImportErrorDetail(f"{label} must be outside the public repository.")
    return resolved


def _iso_datetime(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImportErrorDetail(f"{field} must be an ISO 8601 timestamp with a timezone.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ImportErrorDetail(f"{field} is not a valid ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ImportErrorDetail(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _period(item: dict[str, Any]) -> tuple[str, str]:
    start, end = item.get("period_start"), item.get("period_end")
    if not isinstance(start, str) or not DATE_RE.fullmatch(start):
        raise ImportErrorDetail("period_start must be YYYY-MM-DD.")
    if not isinstance(end, str) or not DATE_RE.fullmatch(end):
        raise ImportErrorDetail("period_end must be YYYY-MM-DD.")
    try:
        if date.fromisoformat(start) > date.fromisoformat(end):
            raise ImportErrorDetail("period_start must not be after period_end.")
    except ValueError as exc:
        raise ImportErrorDetail("The measurement period contains an invalid date.") from exc
    return start, end


def _number(value: object, field: str, *, percent: bool = False) -> float | int | None:
    if value is None or value == "" or value == "-" or value == "~":
        return None
    if isinstance(value, bool):
        raise ImportErrorDetail(f"{field} cannot be a boolean.")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip().replace("\u00a0", "").replace(" ", "")
        has_percent = raw.endswith("%")
        raw = raw[:-1] if has_percent else raw
        if "," in raw and "." in raw:
            raw = raw.replace(",", "")
        elif "," in raw:
            # Locale ambiguity is rejected; callers should use API JSON or a
            # normalized export rather than silently reinterpret decimal commas.
            raise ImportErrorDetail(f"{field} uses an ambiguous comma decimal/thousands format: {value!r}.")
        try:
            number = float(raw)
        except ValueError as exc:
            raise ImportErrorDetail(f"{field} is not numeric: {value!r}.") from exc
        percent = percent or has_percent
    else:
        raise ImportErrorDetail(f"{field} is not numeric: {value!r}.")
    if number < 0:
        raise ImportErrorDetail(f"{field} cannot be negative.")
    if percent:
        number /= 100
    if field in {"clicks", "impressions", "count", "hits_total", "hits_nonbot", "hits_bot"} and number.is_integer():
        return int(number)
    return number


def _column_map(item: dict[str, Any], headers: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    mapping = item.get("columns")
    if not isinstance(mapping, dict) or not isinstance(mapping.get("dimensions"), dict):
        raise ImportErrorDetail("Table imports require an explicit columns.dimensions mapping.")
    dimensions = mapping["dimensions"]
    metrics = mapping.get("metrics", {})
    if not isinstance(dimensions, dict) or not dimensions or not isinstance(metrics, dict):
        raise ImportErrorDetail("columns.dimensions must be nonempty and columns.metrics must be an object.")
    all_fields = {str(value) for value in [*dimensions.values(), *metrics.values()] if value}
    missing = sorted(all_fields - set(headers))
    if missing:
        raise ImportErrorDetail(f"Mapped export columns are missing: {', '.join(missing)}.")
    allowed_metrics = {"clicks", "impressions", "ctr", "position", "citations", "count"}
    unknown = set(metrics) - allowed_metrics
    if unknown:
        raise ImportErrorDetail(f"Unsupported metric field(s): {', '.join(sorted(unknown))}.")
    return {str(k): str(v) for k, v in dimensions.items()}, {str(k): str(v) for k, v in metrics.items()}


def _table_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ImportErrorDetail("The table export has no header row.")
        headers = [name.strip() if name else "" for name in reader.fieldnames]
        if len(headers) != len(set(headers)) or any(not h for h in headers):
            raise ImportErrorDetail("The table export has blank or duplicate column names.")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ImportErrorDetail(f"Row {line_number} has more fields than the header.")
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
        return headers, rows


def _mapped_table(path: Path, delimiter: str, item: dict[str, Any]) -> list[dict[str, Any]]:
    headers, rows = _table_rows(path, delimiter)
    dimension_columns, metric_columns = _column_map(item, headers)
    output = []
    for index, row in enumerate(rows, start=2):
        dimensions = {name: row[column] for name, column in dimension_columns.items()}
        if not all(value.strip() for value in dimensions.values()):
            raise ImportErrorDetail(f"Row {index} has a blank dimension value.")
        metrics = {}
        for name, column in metric_columns.items():
            value = _number(row[column], name, percent=(name == "ctr" and item.get("ctr_format") == "percent"))
            metrics[name] = value
        output.append({"dimensions": dimensions, "metrics": metrics})
    return output


def _gsc_api(path: Path, item: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = item.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions or any(d not in KNOWN_GSC_DIMENSIONS for d in dimensions):
        raise ImportErrorDetail("GSC API imports require explicit supported dimensions.")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows", []), list):
        raise ImportErrorDetail("GSC API JSON must be a Search Analytics response object with a rows array.")
    output = []
    for index, row in enumerate(payload.get("rows", []), start=1):
        keys = row.get("keys")
        if not isinstance(keys, list) or len(keys) != len(dimensions):
            raise ImportErrorDetail(f"GSC API row {index} key count does not match dimensions.")
        dim_values = dict(zip(dimensions, keys))
        # Search Analytics API returns CTR as a ratio (for example, 0.125),
        # unlike some UI exports that render the value with a percent sign.
        metrics = {name: _number(row.get(name), name) for name in ("clicks", "impressions", "ctr", "position") if name in row}
        output.append({"dimensions": dim_values, "metrics": metrics})
    return output


def _safe_referrer(value: str, scheme: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if scheme == "h":
        parsed = urlsplit(value if "://" in value else "https://" + value)
        return (parsed.hostname or "").lower()
    # GoatCounter generated and campaign labels are provider-defined names, not
    # full request URLs. Keep the label but never store query/fragment material.
    return value.split("?", 1)[0].split("#", 1)[0][:120]


def _clean_path(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = parsed.path
    if not path.startswith("/"):
        raise ImportErrorDetail("GoatCounter page path is not an absolute site path.")
    return path


def _goat_rows_from_records(records: list[dict[str, Any]], paths: dict[int, str], refs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple, dict[str, Any]] = {}
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict) or "path_id" not in row or "count" not in row:
            raise ImportErrorDetail(f"GoatCounter hit row {index} is missing path_id/count.")
        path_id = int(row["path_id"])
        path = paths.get(path_id)
        if not path:
            raise ImportErrorDetail(f"GoatCounter path_id {path_id} is not present in paths metadata.")
        if not path.startswith("/"):
            continue  # Exclude event paths; they are not pageviews.
        ref = refs.get(int(row.get("ref_id", 0)), {})
        ref_scheme = str(ref.get("ref_scheme", "o"))
        source = _safe_referrer(str(ref.get("ref", "")), ref_scheme)
        timestamp = str(row.get("hour", row.get("day", "")))
        if not timestamp:
            raise ImportErrorDetail(f"GoatCounter hit row {index} has no hour/day.")
        day = timestamp[:10]
        clean_path = _clean_path(path)
        country = str(row.get("location", "")).split("-", 1)[0].upper()
        key = (day, clean_path, ref_scheme, source, country)
        bucket = aggregated.setdefault(key, {"dimensions": {}, "metrics": {"hits_total": 0}})
        bucket["dimensions"] = {
            "date": day,
            "path": clean_path,
            "language": "fa" if clean_path.startswith("/fa/") else "en",
            "referrer_scheme": ref_scheme,
            "referrer_source": source,
            "country": country,
        }
        bucket["metrics"]["hits_total"] += int(_number(row["count"], "count") or 0)
    return list(aggregated.values())


def _goatcounter_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        first = next(reader, None)
        if not first or len(first) < 2 or first[0] not in {"1", "2"}:
            raise ImportErrorDetail("GoatCounter CSV must start with its documented format version (1 or 2).")
        version = first[0]
        headers = [h.strip() for h in first[1:]]
        required = {"Path", "Event", "Bot", "Referrer", "Referrer scheme", "Location", "Date"}
        if not required.issubset(headers):
            raise ImportErrorDetail(f"GoatCounter CSV v{version} is missing required columns: {', '.join(sorted(required - set(headers)))}.")
        aggregated: dict[tuple, dict[str, Any]] = {}
        for line, values in enumerate(reader, start=2):
            if not values or all(not item.strip() for item in values):
                continue
            if len(values) != len(headers):
                raise ImportErrorDetail(f"GoatCounter CSV row {line} has the wrong number of columns.")
            row = dict(zip(headers, values))
            if row["Event"].strip().lower() == "true":
                continue
            path_value = _clean_path(row["Path"])
            date_value = row["Date"].strip()[:10]
            if not DATE_RE.fullmatch(date_value):
                raise ImportErrorDetail(f"GoatCounter CSV row {line} has an invalid Date.")
            scheme = row["Referrer scheme"].strip().lower() or "o"
            source = _safe_referrer(row["Referrer"], scheme)
            country = row["Location"].strip().split("-", 1)[0].upper()
            bot_code = row["Bot"].strip()
            is_bot = bot_code not in {"", "0"}
            key = (date_value, path_value, scheme, source, country)
            bucket = aggregated.setdefault(key, {"dimensions": {}, "metrics": {"hits_total": 0, "hits_nonbot": 0, "hits_bot": 0}})
            bucket["dimensions"] = {
                "date": date_value,
                "path": path_value,
                "language": "fa" if path_value.startswith("/fa/") else "en",
                "referrer_scheme": scheme,
                "referrer_source": source,
                "country": country,
            }
            bucket["metrics"]["hits_total"] += 1
            bucket["metrics"]["hits_bot" if is_bot else "hits_nonbot"] += 1
    return list(aggregated.values())


def _goatcounter_zip(path: Path) -> list[dict[str, Any]]:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ImportErrorDetail("GoatCounter JSON export must be a ZIP archive.") from exc
    with archive:
        names = set(archive.namelist())
        def read_json(name: str) -> Any:
            match = next((n for n in names if n.endswith("/" + name) or n == name), None)
            if not match:
                raise ImportErrorDetail(f"GoatCounter ZIP is missing {name}.")
            return json.loads(archive.read(match).decode("utf-8-sig"))
        def read_jsonl(name: str) -> list[dict[str, Any]]:
            match = next((n for n in names if n.endswith("/" + name) or n == name), None)
            if not match:
                raise ImportErrorDetail(f"GoatCounter ZIP is missing {name}.")
            return [json.loads(line) for line in archive.read(match).decode("utf-8-sig").splitlines() if line.strip()]
        info = read_json("info.json")
        version = str(info.get("export_version", ""))
        if not version.startswith("1."):
            raise ImportErrorDetail(f"Unsupported GoatCounter JSON export major version: {version!r}.")
        paths_rows = read_jsonl("paths.jsonl")
        refs_rows = read_jsonl("refs.jsonl")
        hits = read_jsonl("hit_stats.jsonl")
        paths = {int(row["id"]): (str(row["path"]), bool(row.get("event"))) for row in paths_rows}
        refs = {int(row["id"]): row for row in refs_rows}
        page_hits = []
        for row in hits:
            path_record = paths.get(int(row.get("path_id", -1)))
            if path_record is None:
                raise ImportErrorDetail(f"GoatCounter path_id {row.get('path_id')} is not present in paths metadata.")
            path, is_event = path_record
            if not is_event:
                page_hits.append(row)
        return _goat_rows_from_records(page_hits, {key: value[0] for key, value in paths.items() if not value[1]}, refs)


def _parse_item(item: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    source = item.get("source")
    report_type = item.get("report_type")
    fmt = item.get("format")
    if source not in SOURCES or not isinstance(report_type, str) or not report_type.strip():
        raise ImportErrorDetail("Each import needs a supported source and explicit report_type.")
    exported_at = _iso_datetime(item.get("exported_at"), "exported_at")
    start, end = _period(item)
    relative_file = item.get("file")
    if not isinstance(relative_file, str) or not relative_file.strip():
        raise ImportErrorDetail("Each import needs a relative file path.")
    if not isinstance(item.get("filters", {}), dict) or not isinstance(item.get("scope", {}), dict):
        raise ImportErrorDetail("filters and scope must be JSON objects so their meaning is explicit.")
    if not isinstance(item.get("limitations", []), list) or any(not isinstance(value, str) for value in item.get("limitations", [])):
        raise ImportErrorDetail("limitations must be an array of explanatory strings.")
    path = _private_path(manifest_path.parent / relative_file, "Provider export")
    if not path.is_file():
        raise ImportErrorDetail(f"Provider export file not found: {path.name}.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if fmt == "gsc_api_json":
        if source != "google_search_console" or report_type != "search_performance":
            raise ImportErrorDetail("gsc_api_json is only supported for Google Search Analytics responses.")
        rows = _gsc_api(path, item)
        limitations = ["Search Analytics API is subject to row limits and may omit anonymized query data.", "Preserve responseAggregationType; do not average positions across rows."]
    elif fmt in {"table_csv", "table_tsv", "mapped_json"}:
        if fmt == "mapped_json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            records = payload.get(item.get("records_key", "rows"), payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
                raise ImportErrorDetail("mapped_json must contain an array of objects at records_key or the top level.")
            fields = set().union(*(set(row) for row in records)) if records else set()
            dimension_columns, metric_columns = _column_map(item, list(fields))
            rows = [{"dimensions": {k: row[v] for k, v in dimension_columns.items()},
                     "metrics": {k: _number(row[v], k, percent=(k == "ctr" and item.get("ctr_format") == "percent")) for k, v in metric_columns.items()}}
                    for row in records]
        else:
            rows = _mapped_table(path, "\t" if fmt == "table_tsv" else ",", item)
        limitations = list(item.get("limitations", []))
        if source == "google_search_console":
            limitations.append("Search Console UI exports include only the visible/paged table and can convert unavailable values to zero.")
        if source == "bing_ai_performance":
            limitations.append("Bing AI Performance is sampled/aggregated and may be delayed; totals across views may not be additive.")
    elif fmt == "goatcounter_csv":
        if source != "goatcounter":
            raise ImportErrorDetail("goatcounter_csv is only supported for GoatCounter.")
        rows = _goatcounter_csv(path)
        limitations = ["Raw session identifiers and User-Agent fields are discarded; no unique-human metric is computed.", "GoatCounter individual-pageview CSV is available only when that account option is enabled."]
    elif fmt == "goatcounter_json_zip":
        if source != "goatcounter":
            raise ImportErrorDetail("goatcounter_json_zip is only supported for GoatCounter.")
        rows = _goatcounter_zip(path)
        limitations = ["GoatCounter JSON export hit counts are retained as provider aggregates; no unique-human metric is inferred.", "Overlapping exports should replace same-day buckets, not be summed."]
    else:
        raise ImportErrorDetail(f"Unsupported format {fmt!r}; supported formats are gsc_api_json, table_csv, table_tsv, mapped_json, goatcounter_csv, goatcounter_json_zip.")

    summary = {
        "source": source,
        "report_type": report_type,
        "format": fmt,
        "exported_at": exported_at,
        "period_start": start,
        "period_end": end,
        "filters": item.get("filters", {}),
        "scope": item.get("scope", {}),
        "file_name": path.name,
        "sha256": digest,
        "row_count": len(rows),
        "limitations": limitations,
    }
    if fmt == "gsc_api_json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        summary["response_aggregation_type"] = payload.get("responseAggregationType", "unknown")
    return {"summary": summary, "rows": rows}


def import_manifest(manifest_path: Path, output_dir: Path, known_fingerprints: set[str] | None = None) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    manifest_path = _private_path(manifest_path, "Import manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("imports"), list):
        raise ImportErrorDetail("Import manifest must have schema_version=1 and an imports array.")
    known = set(known_fingerprints or set())
    results = []
    warnings = []
    active = []
    for index, item in enumerate(manifest["imports"], start=1):
        if not isinstance(item, dict):
            raise ImportErrorDetail(f"Manifest imports[{index - 1}] must be an object.")
        parsed = _parse_item(item, manifest_path)
        summary = parsed["summary"]
        fingerprint = hashlib.sha256("|".join((summary["source"], summary["report_type"], summary["period_start"], summary["period_end"], str(summary["filters"]), summary["sha256"])).encode("utf-8")).hexdigest()
        summary["fingerprint"] = fingerprint
        if fingerprint in known:
            summary["import_status"] = "DUPLICATE_SKIPPED"
            parsed["rows"] = []
        else:
            summary["import_status"] = "IMPORTED"
            known.add(fingerprint)
            active.append(parsed)
        results.append(parsed)
    for i, first in enumerate(active):
        a = first["summary"]
        for second in active[i + 1:]:
            b = second["summary"]
            if a["source"] == b["source"] and a["report_type"] == b["report_type"] and max(a["period_start"], b["period_start"]) <= min(a["period_end"], b["period_end"]):
                warnings.append(f"Overlapping exports were retained separately: {a['file_name']} and {b['file_name']}; do not sum their totals.")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "provider-import-ledger.json"
    ledger = {"schema_version": 1, "fingerprints": sorted(known)}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return results, warnings, known


def summarize_imports(imports: list[dict[str, Any]]) -> dict[str, Any]:
    """Create private, provider-scoped aggregates without session-level data."""
    result: dict[str, Any] = {"definitions": {}, "sources": {}}
    for imported in imports:
        summary = imported["summary"]
        source = summary["source"]
        report_type = summary["report_type"]
        rows = imported.get("rows", [])
        key = f"{source}:{report_type}"
        if summary.get("import_status") == "DUPLICATE_SKIPPED":
            result["sources"][key] = {
                "status": "DUPLICATE_EXPORT_NO_NEW_DATA",
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "note": "This export fingerprint was imported previously; no metrics are repeated or represented as zero.",
            }
            continue
        if source == "goatcounter" and report_type == "pageviews":
            totals = defaultdict(int)
            by_language = defaultdict(int)
            by_path = defaultdict(int)
            by_referrer = defaultdict(int)
            by_campaign = defaultdict(int)
            by_ai_engine = defaultdict(int)
            bot_total = 0
            nonbot_total = 0
            for row in rows:
                dimensions, metrics = row.get("dimensions", {}), row.get("metrics", {})
                count = int(metrics.get("hits_total") or 0)
                path = str(dimensions.get("path", ""))
                language = str(dimensions.get("language", "en"))
                source_label = str(dimensions.get("referrer_source", ""))
                ref_scheme = str(dimensions.get("referrer_scheme", "o"))
                totals["reported_hits"] += count
                by_language[language] += count
                if path:
                    by_path[path] += count
                if source_label:
                    by_referrer[source_label] += count
                    if ref_scheme == "c":
                        by_campaign[source_label] += count
                    engine = AI_REFERRER_HOSTS.get(source_label.lower())
                    if engine:
                        by_ai_engine[engine] += count
                if "hits_bot" in metrics:
                    bot_total += int(metrics.get("hits_bot") or 0)
                    nonbot_total += int(metrics.get("hits_nonbot") or 0)
            result["sources"][key] = {
                "status": summary.get("import_status", "IMPORTED"),
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "exported_at": summary["exported_at"],
                "definitions": {
                    "reported_hits": "Sum of provider aggregate/pageview count rows; not unique people or proof of reading.",
                    "hits_nonbot": "Rows marked nonbot in individual-pageview CSV only; not a unique-person count.",
                    "ai_referrals": "Referrer labels that exactly match a maintained AI product host; not AI citations or crawls.",
                },
                "reported_hits": totals["reported_hits"],
                "hits_nonbot": nonbot_total if any("hits_nonbot" in row.get("metrics", {}) for row in rows) else None,
                "hits_bot": bot_total if any("hits_bot" in row.get("metrics", {}) for row in rows) else None,
                "by_language": dict(by_language),
                "top_paths": [
                    {"path": path, "reported_hits": count}
                    for path, count in sorted(by_path.items(), key=lambda entry: (-entry[1], entry[0]))[:20]
                ],
                "referrers": dict(by_referrer),
                "campaigns": dict(by_campaign),
                "ai_referrals": dict(by_ai_engine),
                "limitations": summary.get("limitations", []),
            }
        else:
            sums: dict[str, int | float] = defaultdict(float)
            for row in rows:
                for metric in ("clicks", "impressions", "citations", "count"):
                    value = row.get("metrics", {}).get(metric)
                    if isinstance(value, (int, float)):
                        sums[metric] += value
            result["sources"][key] = {
                "status": summary.get("import_status", "IMPORTED"),
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "exported_at": summary["exported_at"],
                "export_row_count": len(rows),
                "exported_row_sums_only": dict(sums),
                "limitations": summary.get("limitations", []),
                "aggregation_note": "Sums cover only rows present in this export; do not treat them as sitewide totals without provider documentation and complete-row evidence.",
            }
    result["definitions"] = {
        "ai_referrals": "Client-side referrer attribution to listed product hosts only; not AI citations, impressions, grounding queries, or crawler requests.",
        "unique_visitors": "Not inferred by this import workflow.",
    }
    return result
