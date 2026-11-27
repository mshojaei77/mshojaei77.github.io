---
layout: article
lang: en
title: "Build a Small Search Engine from Scratch with Python"
description: "Build a reproducible toy search engine with an inverted index, BM25 ranking, and a tiny relevance test set, then see what production search still requires."
date: 2025-07-26 08:15:03 +0000
permalink: /writing/build-search-engine-from-scratch/
nav_exclude: true
article_kind: tutorial
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Build a Small Search Engine from Scratch with Python

A search engine does not begin with a large cluster or a crawler that downloads the whole web. The core retrieval loop is smaller: normalize documents into terms, build an index from terms to documents, score the documents that match a query, and evaluate whether useful results appear near the top.

This tutorial builds that loop for a tiny fixed corpus using only Python's standard library. It demonstrates an inverted index and BM25 ranking. It is an educational example; it does not crawl websites, persist an index, or represent web-scale performance.

## The retrieval path

For a small collection, the system can be described in five steps:

1. Store a set of documents with stable identifiers.
2. Normalize and tokenize each document.
3. Build an inverted index from each term to the documents containing it.
4. Tokenize a query and score matching documents.
5. Compare the ranked output with relevance judgments.

An inverted index avoids scanning every document for each query. It maps a term to a postings list that identifies the documents where the term appears. The Stanford [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html) walks through this structure and the stages that produce it.

## A tiny corpus and tokenizer

The example corpus has five documents about retrieval. The tokenizer normalizes Unicode with NFKC, case-folds text, and extracts word-like tokens. This is enough for the example; it is not a general language analyzer. Production systems need language-aware tokenization, spelling and normalization rules, and tests for their actual content.

```python
from collections import Counter, defaultdict
import math
import re
import unicodedata

DOCUMENTS = {
    "bm25-guide": (
        "BM25 ranks documents using term frequency and inverse document frequency. "
        "It limits repeated-term impact and normalizes document length."
    ),
    "index-guide": (
        "An inverted index maps each term to postings of document identifiers "
        "and term frequencies."
    ),
    "vector-guide": (
        "Dense retrieval encodes a query and documents as vectors, "
        "then ranks them by vector similarity."
    ),
    "hybrid-guide": (
        "Hybrid search combines BM25 lexical ranking with dense vector retrieval."
    ),
    "evaluation-guide": (
        "Search evaluation uses relevance judgments to compute precision at k, "
        "recall, and ranking metrics."
    ),
}

WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return WORD.findall(normalized)


def build_index(documents):
    postings = defaultdict(dict)
    lengths = {}

    for doc_id, text in documents.items():
        term_counts = Counter(tokenize(text))
        lengths[doc_id] = sum(term_counts.values())
        for term, frequency in term_counts.items():
            postings[term][doc_id] = frequency

    return dict(postings), lengths


INDEX, DOC_LENGTHS = build_index(DOCUMENTS)
```

The stored posting records a document identifier and term frequency. A larger implementation may also store positions, field identifiers, document norms, or compressed postings. The requirements follow from how the engine will answer queries: phrase search needs term positions, while title boosting needs field-aware postings or separate field statistics.

## Rank matching documents with BM25

BM25 is a lexical ranking function built from term frequency, inverse document frequency, and document-length normalization. A common form of the score is:

```text
score(d, q) = sum over query terms t of:
  IDF(t) * (tf(t, d) * (k1 + 1)) /
  (tf(t, d) + k1 * (1 - b + b * doc_length(d) / average_doc_length))

IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

Here, `tf` is a term's frequency in a document, `df` is the number of documents containing it, and `N` is the corpus size. The parameter `k1` controls how quickly repeated occurrences saturate; `b` controls document-length normalization. Elasticsearch documents `k1 = 1.2` and `b = 0.75` as its BM25 defaults. They are starting values for that implementation, not universal best settings.

The [probabilistic relevance framework](https://doi.org/10.1561/1500000019) explains BM25's relationship to probabilistic retrieval models. The code below uses the common Lucene-style scoring form.

```python
K1 = 1.2
B = 0.75
N = len(DOCUMENTS)
AVG_DOC_LENGTH = sum(DOC_LENGTHS.values()) / N


def search(query, limit=5):
    scores = defaultdict(float)

    for term in set(tokenize(query)):
        term_postings = INDEX.get(term)
        if not term_postings:
            continue

        document_frequency = len(term_postings)
        inverse_document_frequency = math.log(
            1 + (N - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        for doc_id, term_frequency in term_postings.items():
            length = DOC_LENGTHS[doc_id]
            normalization = term_frequency + K1 * (
                1 - B + B * length / AVG_DOC_LENGTH
            )
            scores[doc_id] += (
                inverse_document_frequency
                * term_frequency
                * (K1 + 1)
                / normalization
            )

    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [(doc_id, scores[doc_id]) for doc_id in ranked[:limit]]


for doc_id, score in search("hybrid search dense retrieval"):
    print(f"{doc_id}: {score:.3f}")
```

The engine only visits postings for query terms. Terms missing from the index contribute no score, and documents with no matching terms are not returned. For a production search experience, you will also need an explicit strategy for empty results, spelling errors, synonyms, phrase queries, field boosts, and stable snippets.

## Check the ranking with relevance judgments

A ranked list is not evidence that retrieval is good. Create a small set of queries and label which documents answer each one. Then compute a metric that matches the user experience. This example uses precision at one and mean reciprocal rank (MRR): precision at one asks whether the first result is relevant; reciprocal rank rewards the position of the first relevant result.

```python
QRELS = {
    "bm25 term frequency": {"bm25-guide"},
    "inverted index postings": {"index-guide"},
    "hybrid search dense retrieval": {"hybrid-guide", "vector-guide"},
}


def precision_at_k(ranked_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(doc_id in relevant_ids for doc_id in ranked_ids[:k]) / k


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank
    return 0.0


per_query = []
for query, relevant_ids in QRELS.items():
    ranked_ids = [doc_id for doc_id, _ in search(query)]
    per_query.append((
        precision_at_k(ranked_ids, relevant_ids, 1),
        reciprocal_rank(ranked_ids, relevant_ids),
    ))

mean_precision_at_1 = sum(p for p, _ in per_query) / len(per_query)
mrr = sum(rr for _, rr in per_query) / len(per_query)
print(f"P@1={mean_precision_at_1:.2f}, MRR={mrr:.2f}")
```

With this fixture the code prints `P@1=1.00, MRR=1.00`. The relevance labels are manually supplied only to exercise the evaluation code against five documents. They are not user judgments, and those scores are not search quality or a benchmark. The Stanford book's [ranked retrieval evaluation chapter](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) discusses measures such as average precision, precision at `k`, and NDCG. For a real system, use a diverse query set, documented judgments, and repeated comparisons.

## What a real search engine still needs

The example keeps every document and posting in process memory. It is useful for understanding scoring, but it omits the engineering that makes a search service durable and fast:

- **Collection:** a controlled way to acquire pages or records, obey rate limits, and retain provenance.
- **Parsing and normalization:** extraction of text, title, language, metadata, and canonical document identity.
- **Persistence:** an index format that can be loaded, updated, merged, backed up, and recovered.
- **Query processing:** phrase matching, spelling correction, analyzers, synonyms, filters, and query operators.
- **Ranking:** field boosts, recency or quality signals, and calibration against relevance judgments.
- **Serving:** concurrent requests, pagination, timeouts, caching, observability, and access control.
- **Evaluation:** regression queries, failure analysis, and checks that new content does not break useful rankings.

Even a focused index should have limits for document size and query complexity. If the corpus changes, an incremental update strategy should not leave duplicate or stale postings. If the collection has different languages or fields, use an analyzer and scoring policy suited to them rather than treating every token as equivalent.

## Where lexical search fits with embeddings

BM25 is a strong baseline when users search for exact terms, identifiers, product names, or words that also appear in documents. Dense retrieval can help when a query and a relevant passage use different wording. A hybrid system can combine the two candidate lists, then apply a reranker. The right design depends on the query set; lexical retrieval should remain in the evaluation even when embeddings are added.

For the next step, compare the [embedding models and retrieval representations](/writing/embeddings/) or see the Persian [vector database guide](/fa/writing/vector-databases/). A database can store and retrieve an index, but it cannot compensate for a poor corpus, tokenizer, query set, or relevance evaluation.

## References

- [Manning, Raghavan, and Schütze, *Introduction to Information Retrieval*: Inverted indexes](https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html)
- [Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*](https://doi.org/10.1561/1500000019)
- [Elasticsearch: Similarity settings and BM25 defaults](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
- [Manning, Raghavan, and Schütze: Evaluation of ranked retrieval results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
