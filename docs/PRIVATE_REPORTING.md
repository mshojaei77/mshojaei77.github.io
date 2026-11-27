# Private search and analytics reporting

`scripts/private_search_report.py` combines the strict production audit, article inventory, distribution registry, IndexNow checkpoint, and optional official provider exports. It performs no network requests and does not read credentials. Keep export files, the manifest, import ledger, and generated snapshots outside this repository.

## Supported provider inputs

| Provider | Accepted export | Notes |
| --- | --- | --- |
| Google Search Console | Search Analytics API JSON; CSV or TSV report table | CSV/TSV requires an explicit column map and report dimension. UI exports may be truncated to visible/paged rows; unavailable cells may be exported as zero. API response aggregation type and truncation limitations are retained. |
| Bing Webmaster | CSV/TSV tables; JSON row arrays | Supply an explicit column map because report headers vary by report. Search Performance and AI Performance exports are separate report types and must not be added together. |
| GoatCounter | Versioned CSV v1/v2; documented JSON export ZIP v1.x | Raw Session and User-Agent fields are discarded. Paths are reduced to pathnames and HTTP referrers to hostnames. No unique-human count is inferred. |

XLSX is not currently supported. Use the CSV option for the same Search Console report; this avoids a spreadsheet dependency. If an export is available only as XLSX, add support against a real sanitized sample rather than guessing workbook structure.

## Manifest format

Each export gets an explicit report type, export time, reporting period, filters, scope, and format. Table exports require exact header mappings. This prevents ambiguous column names from being guessed.

```json
{
  "schema_version": 1,
  "imports": [
    {
      "source": "google_search_console",
      "report_type": "search_performance",
      "format": "gsc_api_json",
      "file": "gsc-search-queries.json",
      "exported_at": "2026-11-28T12:00:00Z",
      "period_start": "2026-11-01",
      "period_end": "2026-11-27",
      "dimensions": ["query"],
      "filters": {"search_type": "web"},
      "scope": {"property": "https://mshojaei77.github.io/"}
    },
    {
      "source": "bing_ai_performance",
      "report_type": "ai_citations",
      "format": "table_csv",
      "file": "bing-ai-citations.csv",
      "exported_at": "2026-11-28T12:00:00Z",
      "period_start": "2026-11-01",
      "period_end": "2026-11-27",
      "columns": {
        "dimensions": {"grounding_query": "Grounding query", "page": "Cited page"},
        "metrics": {"citations": "Total citations"}
      },
      "filters": {},
      "scope": {"site": "https://mshojaei77.github.io/"}
    }
  ]
}
```

Paths in `file` are relative to the manifest. Both the manifest and all source exports must be outside the public repository. Use supported `format` values: `gsc_api_json`, `table_csv`, `table_tsv`, `mapped_json`, `goatcounter_csv`, and `goatcounter_json_zip`.

## Run locally

```powershell
python scripts/private_search_report.py `
  --audit "C:\private\seo-audit\seo-url-audit.json" `
  --imports-manifest "C:\private\provider-exports\manifest.json" `
  --output-dir "C:\private\search-operations"
```

The output directory contains a source-stamped report, a normalized private provider snapshot, and `provider-import-ledger.json`. The ledger fingerprints exports to identify exact re-imports. Overlapping reports remain separate and emit a warning; the tool never adds overlapping periods together. Re-importing updated exports creates a new snapshot while preserving the previous one.

The snapshot includes a provider-scoped aggregate summary. GoatCounter hit counts are labeled as reported hits, not unique people or proof of reading. AI referrals are counted only when an observed referrer host exactly matches a maintained product-host list; they are not AI citations or crawler requests. An exact duplicate import is marked as duplicate with no new metric values, not as zero traffic.

## Metric interpretation

- GSC clicks, impressions, CTR and position preserve the provider's row grouping. Do not average positions across rows. The Search Console API may return top rows rather than every row, and query data may be anonymized.
- Bing AI Performance is an AI-citation report, not visits or ordinary search rankings. Keep its measures separate from Search Performance.
- GoatCounter pageview or hit counts are not unique readers. Session identifiers, user agents and full HTTP referrer URLs are not retained in normalized snapshots.
- A missing export is `UNKNOWN`; it is not zero. Provider values are labeled as imported exports, not live API access.

## Security boundary

The private export, manifest, normalized rows, import ledger, and reports must remain outside Git and the Jekyll output. The repository excludes a local `/analytics/` drop folder from both Git and Jekyll; private operating copies should still live in the user's local operations directory. Studio GitHub tokens are now held only in tab memory. Reloading or closing the tab requires signing in again; drafts remain locally stored.

## Tests

Run synthetic parser fixtures without provider credentials or real analytics:

```powershell
python scripts/test_private_metric_imports.py
```
