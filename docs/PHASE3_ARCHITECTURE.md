# Phase 3 — Architecture and Research Decisions

**Baseline:** `eb3fba6cdf994c56a451410b10e4786fa7f8d6c9`, deployed to GitHub Pages on 2026-11-27. A fresh production check on 2026-11-28 returned HTTP 200 for both language homes, sitemap, mixed Atom feed and robots.txt. The sitemap contained 67 URLs. `scripts/check_keyword_map.py` reports 57/57 article routes mapped; the existing eight-cluster taxonomy, 57-route relationship graph and 25 prerequisite records are the starting point. No new taxonomy or article registry is needed.

## Decisions

1. Keep Jekyll, Liquid, static pages and GitHub Pages. Jekyll's `_data` files are the single source for domain clusters, learning paths and tool metadata; layouts render localized views. Do not migrate to Astro, Eleventy, MkDocs or a React app for these requirements.
2. Add one compact **Learn** navigation destination. It provides paths to the knowledge index, structured curricula, the playground, research artifacts, project case studies, reading notes and curated resources. Do not add every destination as a top-level navigation item.
3. Render public knowledge and learning pages from existing topic/relationship data. Cross-language article links carry visible language labels. A Persian article is not represented as an English translation.
4. Implement the first playground tools with browser-native JavaScript, loaded only on their own routes. Input stays in the current browser; no analytics event includes user input. No dependencies or backend are added.
5. Learning progress is optional, versioned local browser storage with a visible reset. It is not synchronized and is never sent to the server.
6. Publish no experiment as completed without a source artifact and an evidence status. Tasmim-Bench's current card documents 120 synthetic scenarios, 0 human-verified labels and 0 independently evaluation-eligible records; it must be described as a synthetic pilot rather than operational model evidence.
7. Preserve article URLs, RSS, sitemap, language metadata, retirement behavior, Studio and the existing editorial classifications. New routes receive deterministic validation.

## Reference review

References were checked on 2026-11-28. Repository state and license fields should be checked again before adopting or bundling any upstream code. Phase 3's initial UI and formulas use no third-party code.

| Reference | Evidence used | Decision |
|---|---|---|
| [Diátaxis](https://diataxis.fr/) | Separates tutorials, how-to guides, reference and explanation by reader need | Keep existing `article_kind`; map views to these distinctions without mass front-matter rewrites |
| [Jekyll documentation](https://jekyllrb.com/docs/) | Static generation and Liquid layouts match current build | Reuse current generator |
| [Jekyll data files](https://jekyllrb.com/docs/datafiles/) | YAML/JSON under `_data` are available to Liquid | Use structured data for curated paths and tool labels |
| [Jekyll pages](https://jekyllrb.com/docs/pages/) | Pages suit non-chronological knowledge views | Add static route sources with front matter |
| [GitHub Pages](https://docs.github.com/en/pages) | Existing host supports the current static/Jekyll product | Preserve hosting |
| [Simon Willison](https://simonwillison.net/) | Compact technical posts, personal voice and usable tools | Keep personal authorship central; avoid dashboard clutter |
| [Simon Willison's tools](https://tools.simonwillison.net/) | Small tools solve concrete tasks and can be independently useful | Build focused tools, not a framework |
| [Simon Willison's TIL](https://til.simonwillison.net/) | Short practical entries complement long-form articles | Reading notes can stay article-first |
| [Jay Alammar](https://jalammar.github.io/) | Explanatory visuals can clarify model mechanics | Use diagrams only when they explain a real relationship |
| [Lilian Weng](https://lilianweng.github.io/) | Deep essays organize concepts and references around an engineering question | Keep article as the canonical detailed explanation |
| [Huyen Chip](https://huyenchip.com/) | Production engineering context and decision trade-offs | Make project and learning paths connect to implementation choices |
| [Distill](https://distill.pub/) | Interactive explanations can pair narration with manipulation | Interactivity should teach a defined concept and remain readable without it |
| [Eugene Yan](https://eugeneyan.com/) | Engineering case studies and explicit synthesis | Research pages must identify evidence and limitations |
| [Hamel Husain](https://hamel.dev/) | Practical LLM experimentation and evaluation | Preserve methodology and failure cases |
| [Hugging Face Learn](https://huggingface.co/learn) | Courses expose sequence, prerequisites and runnable artifacts | Organize routes into short learning modules |
| [roadmap.sh](https://roadmap.sh/) | Broad roadmaps provide navigation but can become expansive catalogs | Keep paths focused on existing evidence, with known gaps visible |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | Course modules connect systems concepts and projects | Add applied activities alongside reading |
| [fast.ai](https://course.fast.ai/) | Progressive lessons combine explanation and practice | Give each path an explicit reader outcome |
| [MIT Missing Semester](https://missing.csail.mit.edu/) | A narrow scope can make a learning resource actionable | Do not mimic its general breadth; focus on AI engineering |
| [Karpathy nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | From-scratch progression with runnable code | Link to existing tutorials; do not copy notebooks |
| [Karpathy LLM101n](https://github.com/karpathy/LLM101n) | Project-centered learning direction | Use as optional further reading, not source text |
| [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) | Reproducible code accompanies a book structure | Prefer linking to the upstream source |
| [fastbook](https://github.com/fastai/fastbook) | Notebook-backed teaching | Do not add a notebook runtime globally |
| [Developer Roadmap source](https://github.com/nilbuild/developer-roadmap) | A data-driven graph can power multiple roadmap views | Reuse this site's already curated graph, not its scale |
| [Hugging Face course source](https://github.com/huggingface/course) | Source-backed course publishing | Link to maintained upstream tutorials |
| [nbdev](https://github.com/AnswerDotAI/nbdev) | Notebook-first literate programming has a larger toolchain | Not justified for current Markdown/Jekyll workflow |
| [JupyterLite](https://github.com/jupyterlite/jupyterlite) | Browser notebook execution has substantial runtime and asset cost | Defer; no current tool requires arbitrary Python execution |
| [Pyodide](https://pyodide.org/en/stable/) | Python-in-browser is possible, including worker use | Defer large runtime; native JS suffices for first tools |
| [Mermaid](https://github.com/mermaid-js/mermaid) | Text source can describe diagrams | Do not add a site-wide renderer; use accessible static SVG for selected diagrams |
| [vis-network](https://github.com/visjs/vis-network) | Interactive network graphs need a parallel readable representation | Defer graphical graph until usability evidence supports it |
| [markmap](https://github.com/markmap/markmap) | Markdown can generate a map-like view | Defer visual-only topic graph |
| [Observable Plot](https://github.com/observablehq/plot) | Declarative charts support data exploration | Avoid dependency for calculations without datasets |
| [Vega-Lite](https://github.com/vega/vega-lite) | Declarative visualization grammar is capable but adds runtime | Not needed for initial calculators |
| [Unicode UAX #15](https://www.unicode.org/reports/tr15/) | NFC/NFKC are Unicode normalization forms with distinct semantics | Label Unicode normalization separately from Persian-specific mappings |
| [Unicode UAX #9](https://www.unicode.org/reports/tr9/) | Bidirectional controls are part of the bidi algorithm | Detect controls but never remove them automatically |
| [Python Unicode database](https://docs.python.org/3/library/unicodedata.html) | Character name, category and direction properties have a defined database | Browser tool labels its limited character-name coverage and uses code-point fallback |
| [Hazm](https://github.com/roshan-research/hazm) and [Parsivar](https://github.com/ICTRC/Parsivar) | Persian NLP libraries demonstrate broader normalization/tokenization choices | Do not port library behavior wholesale into a small inspector |
| [W3C WCAG 2.2](https://www.w3.org/WAI/WCAG22/Understanding/) | Semantic, keyboard, focus and reflow requirements | Native labeled controls, keyboard operation and responsive layouts |
| [Hugging Face KV cache explanation](https://huggingface.co/docs/transformers/main/en/cache_explanation) | KV cache grows with batch, sequence, layers and key/value tensors | Memory estimator states the simplified formula and architecture assumptions |
| [vLLM documentation](https://docs.vllm.ai/en/stable/) | Serving behavior depends on runtime and implementation | Estimator output is a planning estimate, not a promise a model will run |
| [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) | Dataset limitations and provenance belong with datasets | Lab cards link source metadata and identify evaluation status |
| [Tasmim-Bench card](https://huggingface.co/datasets/mshojaei77/tasmim-bench) | Current release explicitly labels synthetic pilot limitations | Do not publish unsupported quality or model superiority claims |
| [Persian OCR benchmark card](https://huggingface.co/datasets/mshojaei77/persian-ocr-bench) | Dataset documents paired images/transcriptions and review field | Describe it as a small resource; do not generalize to all Persian OCR |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [Promptfoo](https://github.com/promptfoo/promptfoo), [Ragas](https://github.com/vibrantlabsai/ragas), [HELM](https://github.com/stanford-crfm/helm), [Hugging Face Evaluate](https://github.com/huggingface/evaluate) | Evaluation frameworks expose methodology, metric definitions and reproducibility | Link to external frameworks; no framework bundled into the public site |
| [Hugging Face Datasets](https://github.com/huggingface/datasets) | Versioned dataset artifacts are a reusable publication target | Link to upstream artifacts instead of copying data |
| [Jupyter Book](https://github.com/jupyter-book/jupyter-book) and [Material for MkDocs](https://github.com/squidfunk/mkdocs-material) | Book/document platforms offer mature navigation but are separate generators | Do not migrate the publication |
| [Astro](https://github.com/withastro/astro), [Eleventy](https://github.com/11ty/buildawesome) | Alternative static architectures can model content differently | Rejected absent a demonstrated Jekyll limitation |

## Deferred decisions

- No learning accounts, LMS, certificates, comments backend, cloud notebook or paid service.
- No mass-rewrite of existing article metadata. Existing `article_kind` values remain authoritative until an article has been reviewed.
- No fabricated benchmark leaderboard. Research Lab pages use artifact-authored statuses and limitations.
- No visual topic graph in the initial release; the semantic topic and learning lists are the accessible primary navigation.
