# Personal AI engineering publication

Hosted at https://mshojaei77.github.io/. Existing URLs are public contracts: preserve them when editing or moving content.

- The homepage and editorial indexes use `_layouts/publication.html` and `assets/css/publication.css`.
- Public articles and long-form Markdown use the shared article layout. The canceled book is excluded from publication; original source remains in the repository and Git history.
- Add an article to the writing index only after it exists; link external articles to their original URLs.
- For research, disclose sampling, versions, methods, results and limitations. Do not invent benchmarks or issue dates.

## Local validation

Install Ruby and Bundler, run `bundle install`, then `bundle exec jekyll build` and `python3 scripts/check_site.py _site`. Preview using `bundle exec jekyll serve`.

Do not configure analytics with placeholder measurement IDs. GitHub Pages deployment remains unchanged.

## Languages

English remains at its original URLs; Persian uses `/fa/`. The route mapping is in `_data/languages.yml`. Persian pages have explicit `lang: fa` and use the shared RTL-aware publication layout. Only publish actual translations or clearly labeled summaries; do not silently substitute English material. Long-form pages use the same publication shell. Add the paired route and reciprocal language links when a full translation becomes available.

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
## Publication Studio (Sveltia CMS)

Open https://mshojaei77.github.io/admin/ after deployment. The static login page is public, but Sveltia's GitHub backend requires repository write access. Open authoring and OAuth are disabled; no credential is embedded in the site. Keep repository write access limited to your owner account. Verify this in repository Settings → Collaborators. This is GitHub-backed authorization, not a separate server-side username allowlist.

### First sign-in (owner only)

1. While signed in as `mshojaei77`, open https://github.com/settings/personal-access-tokens/new and create a **fine-grained** token, e.g. Publication Studio.
2. Select resource owner `mshojaei77`, **Only select repositories**, and `mshojaei77.github.io` only. Set Contents and Pull requests to Read and write (Metadata read is automatic). Use a short expiration; do not grant Administration, Actions, Workflows, or access to other repositories.
3. Open /admin/ and use Sign In with Token. Paste the token only into the CMS login dialog. Never commit it, send it in chat, or enter it in a document. The CMS stores the token in browser local storage; use a trusted browser profile, sign out on shared devices, and revoke it from GitHub if compromised.

Sveltia's official documentation: https://sveltiacms.app/docs/backends/github and https://sveltiacms.app/docs/workflows/editorial. The exact upstream CMS version (0.273.1) is pinned to avoid silent upgrades. The static application is served from UNPKG; review and test new versions before changing the pin.

### Editing and publishing

- Existing articles are separate file entries to protect their original filenames and permalinks. The Persian v6 guide and English HTML tutorial are deliberately raw-mode-only to preserve SVGs, HTML, anchors and code blocks. Raw mode still provides Markdown editing tools. Update both the article title and the existing H1 when appropriate. New Persian and English posts support visual editing, raw Markdown, sanitized live preview, metadata and media uploads. Switch the CMS UI language to Persian in user settings for RTL editing.
- Create a new post in the appropriate language collection. Choose a unique ASCII permalink once, for example `/fa/writing/my-new-post/` or `/writing/my-new-post/`. The Jekyll post filename uses the publication date. Do not edit an existing permalink or create another post to replace one. The original article paths remain fixed.
- Use Save to create a draft branch/PR, move the entry to Ready, then review the PR diff on GitHub and merge only after the Validate publication checks pass. **Direct publication from the CMS is deliberately disabled** until required branch checks are enforced at GitHub. No production branch protection or auto-merge was changed by this installation. The existing Pages build deploys the merge. Git history is the revision history.
- The repository is public; saved CMS drafts and their uploaded assets may be publicly readable in branches or PRs even before deployment. Keep confidential drafts and images outside the repository until ready for publication.
- About, Roadmap and Research pages can be edited in the Pages collection. Their route and layout metadata stay fixed. Avoid changing Liquid templates from the raw editor unless intentional. For changes to the number of v6 figures/references or tutorial code examples, update the matching regression tests in a separate reviewed PR rather than bypassing CI.
- Uploaded media goes under `assets/uploads/`, separate from existing theme assets. Preview is a content preview, not a complete Jekyll build; inspect the public site after deployment. Deletion of articles is intentionally disabled in the CMS to protect stable URLs.

