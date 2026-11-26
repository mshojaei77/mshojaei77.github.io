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
