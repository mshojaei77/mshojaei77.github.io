"""Simulate two independent post removals in the disposable CI checkout."""
from pathlib import Path
import re
for path, lang, slug in [("_posts/2026-11-26-document-parsing-en.html","en","document-parsing"),("_posts/2026-11-26-fa-vector-databases.html","fa","vector-databases")]:
    p=Path(path)
    if not p.exists():
        continue
    route=re.search(r"^permalink:\s*(/\S+)\s*$",p.read_text(encoding="utf-8"),re.M).group(1)
    target=Path("retired-posts")/f"{lang}-{slug}.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(f"---\nlayout: publication\nlang: {lang}\ntitle: Article removed\ndescription: Removed\npermalink: {route}\nsitemap: false\nnav_exclude: true\nretired_post: true\n---\n<section data-retired-post><h1>Article removed</h1></section>\n",encoding="utf-8")
    p.unlink()

# Test date override independently of the filename in the same disposable build.
for slug, date in (("ci-date-order-first", "2024-04-03"), ("ci-date-order-last", "2025-05-04")):
    p = Path("_posts") / f"2023-01-01-en-{slug}.md"
    p.write_text(f'---\nlayout: article\nlang: en\ntitle: "{slug}"\ndescription: "CI-only date fixture"\npermalink: /writing/{slug}/\ndate: "{date} 00:00:00 +0000"\nshow_article_header: true\n---\nTest fixture.\n', encoding="utf-8")
