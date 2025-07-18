---
title: "Embeddings"
nav_order: 4
parent: "Part I: Foundations"
grand_parent: "LLMs: From Foundation to Production"
description: "Learn how language models represent meaning through embeddings, from classic static embeddings like Word2Vec and GloVe to modern contextual embeddings from transformers."
keywords: "Embeddings, Word2Vec, GloVe, FastText, Contextual Embeddings, Semantic Search, Vector Similarity, Cosine Similarity"
---

# 4. Embeddings
{: .no_toc }

**Difficulty:** Beginner-Intermediate | **Prerequisites:** Linear Algebra, Python
{: .fs-6 .fw-300 }

Words are just symbols, so how do models capture their meaning? This chapter dives into embeddings—dense vector representations of text—that allow models to understand semantic relationships. We'll trace the evolution from static word vectors to the powerful contextual embeddings that underpin modern LLMs.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Static Embeddings</h4>
    <p>Classic algorithms like Word2Vec, GloVe, and FastText that generate a single, fixed vector for each word in the vocabulary.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Vector Arithmetic</h4>
    <p>The remarkable property of word embeddings that allows for semantic analogies, famously demonstrated by "king - man + woman = queen".</p>
  </div>
  <div class="concept-grid-item">
    <h4>Contextual Embeddings</h4>
    <p>Embeddings generated by transformer models (like BERT) that change based on the surrounding context, allowing them to capture nuances and polysemy.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Sentence Embeddings</h4>
    <p>Techniques for aggregating word or token embeddings to produce a single vector representation for an entire sentence or document.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Vector Similarity</h4>
    <p>Measuring the "closeness" of two embeddings in vector space using metrics like Cosine Similarity, Dot Product, and Euclidean Distance.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Semantic Search</h4>
    <p>Using embeddings to find documents that are semantically similar to a query, rather than just matching keywords.</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **Word2Vec from Scratch**: Implement the Skip-gram version of Word2Vec in PyTorch to generate embeddings from a text corpus and visualize the results.
2.  **Semantic Search with Sentence-BERT**: Build a semantic search engine for a corpus of documents (e.g., news articles) using a pre-trained Sentence-BERT model and FAISS for efficient vector indexing.
3.  **Embedding Space Visualization**: Use t-SNE or UMAP to create a 2D visualization of a pre-trained embedding space, exploring how words cluster based on semantic meaning.

---

## 🧠 Further Reading

- **[Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)**: The original Word2Vec paper.
- **[Jay Alammar, "The Illustrated Word2vec"](https://jalammar.github.io/illustrated-word2vec/)**: A fantastic visual explanation of the Word2Vec model.
- **[SentenceTransformers Documentation](https://www.sbert.net/)**: A practical library for creating and using sentence and text embeddings. 