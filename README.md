# Personal AI engineering publication

Hosted at https://mshojaei77.github.io/. Existing URLs are public contracts: preserve them when editing or moving content.

- The homepage and editorial indexes use `_layouts/publication.html` and `assets/css/publication.css`.
- Book chapters and existing long-form Markdown use Just the Docs.
- Add an article to the writing index only after it exists; link external articles to their original URLs.
- Mark unfinished chapters as planned; retain their outlines.
- For research, disclose sampling, versions, methods, results and limitations. Do not invent benchmarks or issue dates.

## Local validation

Install Ruby and Bundler, run `bundle install`, then `bundle exec jekyll build` and `python3 scripts/check_site.py _site`. Preview using `bundle exec jekyll serve`.

Do not configure analytics with placeholder measurement IDs. GitHub Pages deployment remains unchanged.

## Languages

English remains at its original URLs; Persian uses `/fa/`. The route mapping is in `_data/languages.yml`. Persian pages have explicit `lang: fa` and use the shared RTL-aware publication layout. Only publish actual translations or clearly labeled summaries; do not silently substitute English material. Existing long-form English pages retain the docs theme. Add the paired route and reciprocal language links when a full translation becomes available.

## Bilingual layout and verification

Both homepages render the same `_includes/home.html` template from paired records in `_data/home.json`. Update both languages together. Localized overview pages link explicitly to the original English resources, without claiming complete translation. Persian uses the unmodified Vazirmatn variable WOFF2 v33.003, self-hosted with upstream OFL and authors notices. The home social card is a dedicated 1200×630 PNG.

## Editorial voice / لحن نوشته‌های فارسی

Write as an AI engineer talking to engineering peers: conversational, native, technically precise and accessible to juniors. Prefer a concrete scenario over vague claims. Keep established technical terms (LLM, RAG, Retriever, Fine-tuning, Eval, Production) and explain unfamiliar concepts simply on first use when space allows. Use natural Persian, consistent respectful address, نیم‌فاصله, and Persian punctuation. Avoid bureaucratic prose, mechanical translation, marketing claims and unnecessary English substitutions. In reports, distinguish measurements, estimates, hypotheses and opinions; discuss trade-offs, failures and limitations. Hypothetical examples must remain explicitly hypothetical. Do not invent personal experience or results. Preserve original flow in translations, and keep overviews labeled as overviews, not full translations. Keep existing facts, destinations and publication status intact. Update Persian homepage card copy and matching index copy together; do not change English content unless explicitly requested.

## Published posts

Add new articles to `_posts/` with explicit `lang`, title, description, date and stable permalink. The homepage and writing indexes automatically list published posts; Persian-only work is labeled as such. The document-parsing weekend reading preserves its original interactive article design in a dedicated layout and local assets. Its source upload Git blob was verified as `263ade133398d50ee20f5ffe57b6ae96e603fc58`. The article is available at `/fa/writing/document-parsing/` and included in the feed and sitemap. Do not mark an English translation as published until it exists.
