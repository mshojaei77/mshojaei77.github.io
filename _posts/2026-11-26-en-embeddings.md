---
title: "Embeddings Explained: From Word Vectors to Retrieval Models"
layout: article
lang: en
permalink: /writing/embeddings/
nav_exclude: true
description: "Compare static, contextual, sentence and retrieval embeddings; understand similarity scores and how to evaluate embedding search for a RAG system."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Embeddings: From Word Vectors to Retrieval Models

An embedding is a learned mapping from an item to a vector of numbers. The item might be a word, a token, a sentence, an image, or a user. The vector makes it possible to compare items with mathematical operations, but a vector is not a self-contained definition of meaning. It is useful only in relation to the model, training objective, and task that produced it.

The word “embedding” is used for several different things in language systems. Keeping those meanings separate makes it easier to understand both Transformer internals and semantic search.

## Three representations often called embeddings

### Static word embeddings

Word2Vec and GloVe assign one vector to each vocabulary entry. Word2Vec learns vectors through context prediction objectives such as Skip-gram and CBOW; GloVe uses global word co-occurrence statistics. FastText adds character n-gram information, which can help represent rare forms related to seen subwords.

Because the vector belongs to the word type, a static model gives the same vector to both meanings of *bank* in “river bank” and “bank account.” These methods remain useful for understanding distributional learning and for some compact or specialized systems, but they do not create a context-specific representation for each occurrence.

### Contextual token representations

In a Transformer, token IDs first select vectors from an embedding table. The model then repeatedly transforms those vectors using attention and feed-forward layers. The hidden representation at a later layer depends on surrounding tokens, so the same token can have different representations in different sentences.

This is why “the embedding layer” and “the contextual embedding” are not interchangeable terms. The first is a learned lookup table near the model input; the second usually refers to a layer's context-dependent hidden states.

### Text embeddings for retrieval

A retrieval embedding model maps a query or document into a vector intended to place relevant items near one another. Many modern systems encode queries and passages separately, then rank candidates by cosine similarity, dot product, or another score. Some models require different prefixes or instructions for query and document inputs. Follow the model's own usage guidance; a generic sentence encoder may not be a good retrieval encoder.

Retrieval embeddings are trained for a task and a distribution. A useful vector space for English web search may perform poorly on Persian, specialized terminology, long passages, or a domain unlike its training data. The embedding dimension by itself says little about retrieval quality.

## Similarity is a ranking signal, not a probability

For vectors `x` and `y`, cosine similarity is:

```text
cosine(x, y) = dot(x, y) / (norm(x) * norm(y))
```

For example, vectors `[1, 0]` and `[0.8, 0.6]` have cosine similarity 0.8. That number describes their angle under this metric. It does not mean there is an 80% chance the texts are relevant. Similarity scores are generally not calibrated probabilities, and their useful range depends on the model, normalization, and corpus.

Some libraries normalize vectors and then use dot product as cosine similarity. Others do not. If vectors are not normalized, dot product also depends on their magnitudes. Match the index metric to the model's documented output and measure retrieval quality rather than assuming that “cosine” is always the right choice.

## A small, reproducible cosine example

This Python example uses only the standard library. The vectors are made-up values to demonstrate the calculation; they are not produced by an embedding model.

```python
from math import sqrt

def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have equal dimensions")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine is undefined for a zero vector")
    return dot / (norm_a * norm_b)

print(round(cosine([1.0, 0.0], [0.8, 0.6]), 2))  # 0.8
```

Real retrieval systems add important details that this example leaves out: model-specific encoding, batching, vector storage, filtering, approximate nearest-neighbor search, and relevance evaluation.

## Evaluate the retrieval behavior you need

Choosing an embedding model is an evaluation task. Build a small test set from realistic queries and relevant documents, including hard negatives: passages that share vocabulary with the query but do not answer it. Keep the test cases separate from documents used to tune the system where possible.

Useful retrieval metrics include Recall@k (whether a relevant result appears in the first *k* results), reciprocal rank (how early the first relevant result appears), and nDCG (how well graded relevance is ordered). For RAG, retrieval metrics are necessary but not sufficient: assess answer correctness and citation support as a separate stage. A reranker may improve ordering among retrieved candidates, but it cannot recover a document the first-stage retriever never returned.

For multilingual use, include queries and passages in each language and in realistic mixed-language forms. Test spelling variants, script variants, transliteration, domain vocabulary, and chunk sizes. Compare models on the same cases, with the same corpus and retrieval settings. Benchmark leaderboards help shortlist models; they do not replace a workload-specific evaluation.

## Common mistakes

- **Using a model's input token embeddings as a semantic search index.** They were not necessarily trained to make sentence-level similarity useful.
- **Treating cosine scores as confidence.** A high score is not a calibrated probability of relevance.
- **Mixing incompatible vectors.** Query and document encoders, model versions, dimensions, and preprocessing must match the index.
- **Assuming normalization is harmless.** It changes vector magnitudes and may change ranking when using dot product.
- **Judging a RAG system only by answer fluency.** Measure retrieval, answer quality, and evidence attribution separately.
- **Assuming English results transfer to Persian.** Test the language and scripts your users actually send.

Embeddings make similarity search possible, but they do not remove the need to define relevance. Treat the model as one component in a retrieval pipeline and evaluate that pipeline against the failures that matter to your readers.

## Further reading

- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
- Pennington, Socher, and Manning, [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/).
- Bojanowski et al., [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/), the FastText subword model.
- Reimers and Gurevych, [Sentence-BERT](https://aclanthology.org/D19-1410/), for sentence-level representations designed for similarity comparison.
- Muennighoff et al., [MTEB: Massive Text Embedding Benchmark](https://aclanthology.org/2023.eacl-main.148/), a broad evaluation across embedding tasks.
