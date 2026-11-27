---
title: "Data Preparation for LLMs: From Sources to Auditable Corpora"
layout: article
lang: en
permalink: /writing/data-preparation/
nav_exclude: true
description: "A practical guide to building LLM training corpora: source provenance, text quality, deduplication, benchmark contamination, multilingual coverage, and reproducible processing."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Data Preparation for LLMs: From Sources to Auditable Corpora

Training data is not just a pile of text to clean. It is a collection of sources gathered under different conditions, with different quality, rights, languages, and risks. Data preparation turns those sources into a documented corpus that a training run can consume and a team can later inspect.

There is no single cleaning recipe that fits every model or goal. A web corpus for general pretraining, a code dataset, and a small instruction-tuning set need different filters. Each transformation should have an explicit purpose and a way to check what it removed.

## 1. Record where each source came from

Start with sources the project is authorized to use. “Publicly accessible” does not by itself establish permission to redistribute or use every item for every purpose. Record the source, collection date, applicable terms or license, permitted use, and any opt-out or deletion path. Keep stable source identifiers and hashes in access-controlled pipeline metadata so later stages can be traced. A hash is not anonymization: hashes of guessable or low-entropy content can still identify records, so do not publish per-record identifiers or hashes.

Separate the data by source and preserve that grouping through the pipeline. If a domain, source, or language later turns out to be low quality or out of scope, provenance lets you remove or reprocess those records rather than rebuilding everything from scratch.

## 2. Parse first, then normalize deliberately

Parsing converts a source format into records the pipeline can reason about: document text, metadata, and structure such as headings, paragraphs, code blocks, or tables. Keep useful structure when the task depends on it. A table flattened into a stream of numbers can lose which row and header a value belonged to.

Normalize only when it serves the training objective. Removing crawl boilerplate or decoding HTML entities may be appropriate. Lowercasing, stripping punctuation, or collapsing whitespace across every document can destroy information in names, code, equations, or languages that rely on those marks. Store the original or a recoverable intermediate, and make each transformation versioned and testable.

## 3. Filter for quality without erasing useful variation

Useful checks can catch empty documents, broken encodings, repeated navigation text, extreme lengths, malformed markup, or documents in an unexpected language. Language identification and quality classifiers can help route data, but their errors may vary across languages and domains. Review examples from both accepted and rejected records, including low-resource languages and mixed-language text.

Use targeted filters with logged reasons instead of one opaque “quality score.” A short document is not automatically bad; a duplicated legal clause may be important; a rare language may have unusual punctuation. Measure how each rule changes counts and inspect samples before applying it at scale.

## 4. Deduplicate at more than one level

Exact deduplication catches byte-for-byte or normalized-text repeats. Near-duplicate detection can find copied pages, mirrored articles, or small edits. One common family of methods represents documents as sets of token or word shingles, uses MinHash signatures to estimate Jaccard similarity, and uses locality-sensitive hashing (LSH) to find likely matches efficiently. The similarity threshold is a policy choice: set it too low and distinct documents may be removed; set it too high and copies remain.

Deduplication can reduce repeated exposure to the same text and reduce evaluation leakage when benchmark content appears in the training corpus. It can also change the distribution: boilerplate, popular references, and repeated safety language may disappear at different rates. Keep the duplicate groups and the representative-selection rule so the effect can be audited. Research on language-model training has found benefits from deduplication, but its results are tied to the paper's datasets and setup; they are not a guarantee that every deduplication rule improves every training run.

This small example removes only exact duplicate strings. It does not normalize text or detect near duplicates:

```python
from hashlib import sha256

def remove_exact_duplicates(documents):
    seen = set()
    kept = []
    duplicate_ids = []

    for document_id, text in documents:
        digest = sha256(text.encode("utf-8")).digest()
        if digest in seen:
            duplicate_ids.append(document_id)
            continue
        seen.add(digest)
        kept.append((document_id, text))

    return kept, duplicate_ids

documents = [("a", "A useful sentence."), ("b", "A useful sentence."), ("c", "A changed sentence.")]
kept, duplicates = remove_exact_duplicates(documents)
print([doc_id for doc_id, _ in kept], duplicates)  # ['a', 'c'] ['b']
```

For a real corpus, record each removed document's source ID and the ID of the retained representative. That keeps deduplication traceable and reversible.

## 5. Protect evaluation sets from contamination

If benchmark questions or answers enter the training data, evaluation may measure familiarity with those examples rather than generalization. Remove known evaluation content before training when possible, and record which benchmarks or protected collections were checked. Exact matching will miss paraphrases and related material, so a clean scan does not prove that contamination is absent. Keep evaluation data isolated and report the limits of any contamination check.

## 6. Treat privacy, safety, and rights as pipeline requirements

PII detection, sensitive-data filtering, source terms, retention, and takedown handling should be defined before bulk collection. Automated detectors can miss private information and can also wrongly remove benign text. Use risk-based review and a documented escalation path; do not assume a single regex makes a corpus safe.

Dataset documentation should answer practical questions: why the collection exists, how it was gathered, what it contains, what processing was applied, what uses are allowed, and what limitations remain. A dataset card or datasheet helps make those choices visible. It does not replace source-specific rights review or access controls.

## 7. Preserve multilingual coverage

Track language and script distributions before and after filtering. Include representative samples for each language you intend to support. A filter calibrated on English web text can reject valid Persian, Arabic, code-switched, or transliterated text. For Persian, inspect Arabic/Persian character variants, half-spaces, punctuation, digits, and mixed Latin terms; do not silently rewrite originals during an evaluation comparison.

Measure token counts with the tokenizer that will be used for training or inference. Uneven tokenization can affect context use and the number of training tokens allocated to each language, but token count alone is not a measure of data quality. Evaluate the resulting model on language-specific tasks and document the corpus mixture.

## 8. Make the pipeline reproducible

For each release, preserve the source snapshot or its stable references, code revision, configuration, tool versions, filtering thresholds, deduplication method, random seeds where relevant, and counts before and after each step. Log reasons for exclusions. Save small review samples from each stage so an engineer can inspect what changed without rerunning the full pipeline.

The goal is not the cleanest-looking text. It is a corpus with documented origins, transformations, exclusions, and known limits that matches the model's intended use.

## Further reading

- Gao et al., [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027), a documented mixture of text sources.
- Soldaini et al., [Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://aclanthology.org/2024.acl-long.840/), on corpus construction and its curation toolkit.
- Penedo et al., [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557), an example of a large web-data processing pipeline.
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577/).
- Gebru et al., [Datasheets for Datasets](https://arxiv.org/abs/1803.09010), on documenting dataset motivation, composition, collection, and limitations.
- Common Crawl, [Get Started](https://www.commoncrawl.org/get-started), on accessing its web-crawl archives and extracted formats.
- Cheng, Chang, and Wu, [A Survey on Data Contamination for Large Language Models](https://arxiv.org/abs/2502.14425).
