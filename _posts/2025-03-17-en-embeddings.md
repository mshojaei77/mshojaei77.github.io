---
layout: article
lang: en
title: "Word Embeddings to Retrieval Vectors: What an Embedding Means"
description: "A practical guide to static, contextual, sentence, and retrieval embeddings, with model-selection and evaluation advice for multilingual and Persian search."
date: 2025-03-17 20:30:00 +0000
permalink: /writing/embeddings/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Word Embeddings to Retrieval Vectors: What an Embedding Means

An embedding is a learned vector representation. The word “learned” matters: a vector only becomes useful for a task because of the data, objective, model, and comparison method used to produce it. Two nearby vectors are not automatically synonyms, and a general-purpose embedding model is not automatically a good retriever for your documents.

## Static word embeddings

Word2Vec and GloVe assign a fixed vector to each vocabulary item. Their training objectives use patterns of word occurrence and co-occurrence, so the vectors can capture useful distributional regularities. The same word receives the same vector in every sentence, however. A static representation cannot directly distinguish “bank” as a financial institution from “bank” beside a river.

The familiar vector analogy *king − man + woman ≈ queen* is an illustration of geometry in a particular learned space, not a reliable semantic operation. Results depend on the corpus, vocabulary, preprocessing, and evaluation. Do not treat a striking analogy example as evidence that every relation is encoded cleanly.

## Contextual token representations

Models such as ELMo and BERT produce token representations that depend on surrounding text. A word used in two different contexts can therefore have different hidden states. This is useful for tasks where a downstream model needs a context-sensitive representation for each token, such as sequence labeling or question answering.

A contextual token vector is not automatically a good sentence vector. Taking a hidden state from `[CLS]`, averaging token states, or reading a model's `pooler_output` gives a vector, but the pooling choice alone does not train that vector to rank semantic matches. Sentence similarity and retrieval usually need an objective, checkpoint, and pooling method suited to the target task.

## Sentence and retrieval embeddings

Sentence embedding models are trained so that vectors for related texts have a useful geometric relationship. Sentence-BERT used siamese and triplet structures for this purpose; SimCSE showed contrastive objectives with dropout or labeled sentence pairs. Those are examples, not one universal recipe used by every modern embedding model.

Retrieval models have an even narrower job: represent a query and candidate passages so that relevant passages rank above irrelevant ones. Some models encode queries and documents symmetrically; others use different instructions or prompts for each side. If a model expects “Represent this sentence for retrieval” or a query prefix, omitting it changes the input contract.

In a vector search system, similarity is computed with a metric such as dot product or cosine similarity. Cosine similarity compares direction after accounting for vector length; dot product also responds to magnitude. Normalizing vectors can make these rankings equivalent for a given pair of vectors, but it does not fix a poorly trained embedding space or an unsuitable task objective. Use the metric and normalization expected by the model and index.

## Why benchmark scores do not pick a universal winner

An embedding model may perform well on semantic textual similarity and poorly on retrieval, clustering, classification, or reranking. MTEB was introduced to evaluate multiple embedding tasks and found that no single method dominated across all of them in its study. Later multilingual benchmarks broadened language and task coverage, but their results still do not replace a test on your own queries and corpus.

For Persian, FaMTEB reports 63 datasets across seven tasks, including retrieval, reranking, classification, clustering, and semantic textual similarity. Its authors combine existing, translated, and newly created data. That breadth is useful for screening models, but no benchmark exactly represents every product's vocabulary, dialect, document structure, or query distribution.

## Evaluate the retrieval job, not just vector similarity

For RAG, construct query-document pairs that resemble actual use, including difficult negatives, short queries, rare names, and questions that cannot be answered from the corpus. Split near-duplicate documents and translated variants carefully so that evaluation does not accidentally reward memorization.

Measure retrieval with metrics that reflect the task. Recall@k asks whether relevant evidence appears in the first *k* results. MRR rewards an early first relevant result. nDCG accounts for graded relevance and rank. Inspect misses manually as well: a good average can hide failures on product names, Persian spelling variants, tables, or long documents.

Then evaluate the full retrieval pipeline. Chunking, metadata filters, hybrid lexical search, reranking, and query expansion can change the final evidence more than switching embedding models. Keep retrieval results separate from answer quality so you can identify which stage caused a failure.

## Persian and multilingual model selection

Check how the model handles Persian Yeh and Arabic Yeh, Persian and Arabic Kaf, نیم‌فاصله, digits, punctuation, and code-switched text. Apply the same normalization to documents and queries, and keep the unmodified source available for display and citation. Over-normalization can collapse distinctions or alter names.

Compare candidate models on the same index, query set, filters, and evaluation code. Record model revision, tokenizer, embedding dimension, instruction format, normalization, vector metric, and hardware. For production, also measure indexing time, storage, query latency, memory, license terms, and the cost of re-embedding when a model changes.

## The engineering takeaway

“Embedding” names a representation, not a guarantee of semantic understanding. Static word vectors, contextual token states, sentence embeddings, and retrieval vectors solve different problems. Pick the representation and objective for the task, follow the model's input instructions, and evaluate ranking on the data your system will actually search.

## Sources

- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
- Pennington, Socher, and Manning, [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/).
- Peters et al., [Deep Contextualized Word Representations](https://aclanthology.org/N18-1202/).
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
- Reimers and Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084).
- Gao, Yao, and Chen, [SimCSE](https://arxiv.org/abs/2104.08821).
- Muennighoff et al., [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316).
- Zinvandi et al., [FaMTEB: Massive Text Embedding Benchmark in Persian Language](https://aclanthology.org/2025.findings-emnlp.614/).
