# Personal AI engineering publication

Hosted at https://mshojaei77.github.io/. Existing URLs are public contracts: preserve them when editing or moving content.

- The homepage and editorial indexes use `_layouts/publication.html` and `assets/css/publication.css`.
- Public articles and long-form Markdown use the shared article layout. The canceled book is excluded from publication; original source remains recoverable from Git history.
- Add articles to _posts; the writing archive is generated automatically. Link external articles to their original URLs.
- For research, disclose sampling, versions, methods, results and limitations. Do not invent benchmarks or issue dates.

## Local validation

Install Ruby and Bundler, run `bundle install`, then `bundle exec jekyll build` and `python3 scripts/check_site.py _site`. Preview using `bundle exec jekyll serve`.

Do not configure analytics with placeholder measurement IDs. GitHub Pages deployment remains unchanged.

## Languages

English articles use `/writing/<slug>/`; Persian articles use `/fa/writing/<slug>/`. Legacy article URLs redirect. The route mapping is in `_data/languages.yml`. Persian pages have explicit `lang: fa` and use the shared RTL-aware publication layout. Only publish actual translations or clearly labeled summaries; do not silently substitute English material. Long-form pages use the same publication shell. Add the paired route and reciprocal language links when a full translation becomes available.

## Bilingual layout and verification

Both homepages render the same `_includes/home.html` template from paired records in `_data/home.json`. Update both languages together. Localized overview pages link explicitly to the original English resources, without claiming complete translation. Persian uses the unmodified Vazirmatn variable WOFF2 v33.003, self-hosted with upstream OFL and authors notices. The home social card is a dedicated 1200×630 PNG.

## Editorial voice / لحن نوشته‌های فارسی

Write as an AI engineer talking to engineering peers: conversational, native, technically precise and accessible to juniors. Prefer a concrete scenario over vague claims. Keep established technical terms (LLM, RAG, Retriever, Fine-tuning, Eval, Production) and explain unfamiliar concepts simply on first use when space allows. Use natural Persian, consistent respectful address, نیم‌فاصله, and Persian punctuation. Avoid bureaucratic prose, mechanical translation, marketing claims and unnecessary English substitutions. In reports, distinguish measurements, estimates, hypotheses and opinions; discuss trade-offs, failures and limitations. Hypothetical examples must remain explicitly hypothetical. Do not invent personal experience or results. Preserve original flow in translations, and keep overviews labeled as overviews, not full translations. Keep existing facts, destinations and publication status intact. Update Persian homepage card copy and matching index copy together; do not change English content unless explicitly requested.

## Published posts

Add new articles to `_posts/` with explicit `lang`, title, description, date and stable permalink. The homepage and writing indexes automatically list published posts; Each archive lists posts in its own language. The document-parsing weekend reading uses the shared article layout, retaining original content and interactive features. Its source upload Git blob was verified as `263ade133398d50ee20f5ffe57b6ae96e603fc58`. The article is available at `/fa/writing/document-parsing/` and included in the feed and sitemap. Do not mark an English translation as published until it exists.

Light is the first-visit default; the shared theme toggle saves the preference. Public code is linked directly to GitHub. Telegram channel and community links are shared across both languages.

All internal tutorials and articles have paired English and Persian routes. Resource links retain their original publication titles; Persian OCR examples retain source characters. The theme starts in light mode and remembers explicit user selection.

## Direct publication policy

Publish authored articles directly in the bilingual blog; do not link to Medium or advertise a separate newsletter. Keep existing technical reference titles and replace missing links to the author's work with verified local article routes only after the migrated articles have been published. The retired newsletter pages are removed from the generated site.

## Document parsing tutorial

The Persian article at `/fa/writing/document-parsing/` is the v6 engineering guide. The English route retains the earlier practical tutorial and downloadable code in `assets/examples/document-parsing/`. The routes remain cross-linked but are not advertised as equivalent translations. The accidentally published `/fa/writing/document-parsing-engineering/` route redirects to the canonical Persian article and is excluded from publication indexes. The Python fixture test exercises native text, a simple digital table, scan routing and English OCR. Optional external tools are documented extensions, not part of that smoke test. Keep source links, stable language routes, the English tutorial’s legacy section anchors, and the complete Persian v6 manuscript. Do not reintroduce project-specific benchmarks or claim production quality from the synthetic fixture.
## Publication Studio

Open https://mshojaei77.github.io/admin/. Studio is a self-hosted, dependency-free source editor; Jekyll and all public content are unchanged.

- Sign in with a fine-grained GitHub token restricted to `mshojaei77.github.io`, Contents and Pull requests read/write, automatic Metadata read. Studio verifies the GitHub numeric owner ID and username and repository write access. GitHub enforces actual write permissions; the owner identity check in the public static application is not a server-side allowlist. Keep repository write access restricted to the owner. Never paste tokens in chat or commit them. Revoke the old Sveltia token if it was used; the old CMS may have left credentials in browser storage. By default Studio holds its token only in tab memory. Explicitly checking Remember me stores a fine-grained token in this browser's localStorage for up to seven days (non-sliding expiry); auto sign-in rechecks owner identity and repository permissions. Sign out clears the stored token; expired or rejected credentials are discarded. This is a convenience option for trusted personal devices, not a secure server-issued session: same-origin JavaScript can read the stored token, so leave it unchecked on shared devices or when publishing untrusted HTML/JS. Use a short-lived repository-scoped token. Classic tokens are never persisted.
- Drafts and staged images are saved in IndexedDB, locally on this browser profile. Autosave is debounced by 250 ms, and the footer indicates when persistence completed. Download source for backup before clearing site data; local drafts are not encrypted, synced, or a replacement for backup. Use a single trusted browser profile; Studio uses Web Locks where supported to prevent simultaneous editing tabs.
- Use the source editor with formatting shortcuts, RTL/LTR toggle and editable title/description/RSS excerpt and publication date. Language, layout, filename and canonical permalink are preserved for existing posts. Changing the date sets midnight UTC of the chosen Gregorian calendar day; unchanged dates retain their original time and offset. Future dates are rejected (scheduled publishing is not supported). New posts can select the date in the creation/import dialog. The sidebar uses the cached front-matter date and sorts newest first; date changes update immediately and persist in local drafts. For content with its own H1, edit it in the source explicitly. The source remains UTF-8; Studio does not round-trip through a rich-text serializer. Existing source up to 950 KB is supported. Image uploads are staged locally, accept PNG/JPEG/WebP/GIF up to 2 MB, and are committed together with the article. The image preview is a placeholder for Markdown images; HTML and SVG are previewed in a no-script, no-network sandbox. The preview is an approximate local Markdown rendering, not the Jekyll site, and runs on demand in a Web Worker.
- New posts use a unique ASCII permalink slug, title, description and language. Save locally never calls GitHub. Publish validates the base blob and protected metadata, checks the filename-derived route against the directory listing, then commits the article and staged media atomically using GitHub GraphQL createCommitOnBranch with expectedHeadOid. This is a direct write to main, not a PR. CI and GitHub Pages run after the commit. If the response is lost, retry verifies content hashes before attempting another commit. Existing unresolved PRs remain recoverable in Studio. Git history retains revisions.
- The repository and remote PRs are public. Local drafts are private to the browser profile until submitted; after publication, the contents and uploaded images are public even if not yet deployed. A lost network connection never intentionally clears local work. Download the source before manually reconciling a conflict. For PRs that are closed without merging, use Check PR status to recover the draft.
- Studio intentionally has no OAuth server, CDN scripts, service worker, content database, or runtime framework. If a truly private admin route or server-side owner allowlist is needed, deploy an external authentication service; GitHub Pages cannot enforce that route restriction. No GitHub branch protection was changed. Direct commits are validated after the write, not gated by pre-merge CI.

CI runs the site, Python tutorial, localization and Studio browser checks against the actual Persian manuscript and a mocked GitHub API. The mocked tests do not prove live authenticated writes. Direct publication requires Contents write permission and an unprotected or appropriately authorized main branch. Review changes in GitHub and inspect the deployed article after publishing.

### File imports and post deletion

Use Import to create a local article from a UTF-8 `.md`, `.markdown` or `.html` file (950 KB including generated metadata). Recognized front matter suggests title, description, language and slug; review them in the dialog. HTML fragments are preserved; full HTML documents import only their body, not head styles or scripts outside it. Use Import file within an existing article to replace its body with the same format while retaining its original path and front matter. No automatic cross-format conversion. Only import trusted authored content; review HTML, links and Liquid before publishing.

Delete post asks for confirmation, then removes the original source and creates a localized noindex removal page at the same permalink in one atomic commit. Retry checks for an already committed deletion; a known failed deletion may be cancelled. The post disappears from the archive and feed after merge, its old URL still works, and Git history and media remain. Deleted permalinks are reserved. Local-only drafts can be removed with Discard local draft. Legacy pending PRs must be resolved first. Page deletion is not supported.

## Writing consolidation

The standalone Research section is removed in both languages and from Studio. Market analysis and vector database comparison are regular bilingual posts under `/writing/` and `/fa/writing/`. Their previous article URLs redirect to their canonical writing URLs. The migrated editions retain their original article bodies and use the November 26 edition commit dates for archive ordering; no new research or benchmark update is implied.

## Unified content structure

The only standalone editorial pages are Home, Roadmap and About in each language. Writing is a generated archive, not a separate content category. All published articles, guides and tutorials live in `_posts/` under `/writing/<slug>/` or `/fa/writing/<slug>/`. Use filenames `YYYY-MM-DD-en-slug.md` or `YYYY-MM-DD-fa-slug.md` (HTML is also supported); the two original Document Parsing filenames remain valid. The migrated content uses the November 26 consolidation edition date, not a newly researched or verified publication date. `redirects/` contains noindex compatibility redirects. The retired book source, scratch image note, unused styles and unused logo were removed from the current branch; Git history preserves them. Downloadable lab files, licenses, and live assets are retained.

Studio has an EN/FA switch in its header. Each mode shows only posts and editable About/Roadmap pages in that language, including local drafts. The choice is stored locally; switching flushes unsaved edits before closing the current editor. New drafts default to the selected language, and importing or creating a draft in the other language switches to its matching view. The Persian interface uses RTL and localized main controls; raw API diagnostics may remain in English. Filename locale is checked against front matter by CI and Studio, including the two legacy Document Parsing filenames.

Published sidebar titles come from actual GitHub front matter, cached by path and blob SHA. Missing or changed post titles are loaded in a bounded background queue (three requests concurrently), so login is not blocked by downloading bodies. Editing the title updates the active sidebar row immediately; a successful publish refreshes the cached title and SHA. Uncached or unavailable titles show a slug rather than a stale hardcoded label.
