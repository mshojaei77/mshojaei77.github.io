---
title: "Tokenization"
nav_order: 3
parent: "Part I: Foundations"
grand_parent: "LLMs: From Foundation to Production"
description: "An introduction to tokenization, the process of converting raw text into a sequence of tokens suitable for a language model, covering BPE, WordPiece, and SentencePiece."
keywords: "Tokenization, BPE, WordPiece, SentencePiece, Subword Tokenization, Vocabulary, OOV, Normalization, Pre-tokenization"
---

# 3. Tokenization
{: .no_toc }

**Difficulty:** Beginner | **Prerequisites:** Python Basics
{: .fs-6 .fw-300 }

Language models don't see text as we do. This chapter breaks down tokenization, the crucial first step in any NLP pipeline where raw text is converted into a sequence of tokens that the model can understand. We'll explore the trade-offs between different strategies and the algorithms that power modern tokenizers.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Token Fundamentals</h4>
    <p>The differences between character, word, and subword tokenization and the out-of-vocabulary (OOV) problem.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Normalization & Pre-tokenization</h4>
    <p>The preliminary steps of cleaning text, such as handling whitespace, punctuation, and Unicode normalization.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Byte-Pair Encoding (BPE)</h4>
    <p>A popular subword algorithm that iteratively merges the most frequent pairs of bytes, used by GPT-style models.</p>
  </div>
  <div class="concept-grid-item">
    <h4>WordPiece</h4>
    <p>A variant of BPE used by BERT-style models that merges tokens based on the likelihood of the training data.</p>
  </div>
  <div class="concept-grid-item">
    <h4>SentencePiece</h4>
    <p>A tokenization system that treats text as a raw stream of Unicode characters, making it language-agnostic.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Modern Tokenizer Libraries</h4>
    <p>An overview of practical, high-performance libraries like Hugging Face `tokenizers` and OpenAI's `tiktoken`.</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **BPE Tokenizer from Scratch**: Implement the Byte-Pair Encoding algorithm in pure Python to build a tokenizer from a small text corpus.
2.  **Tokenizer Comparison**: Train BPE, WordPiece, and Unigram tokenizers using the Hugging Face `tokenizers` library on the same dataset and compare their resulting vocabularies and tokenizations.
3.  **Vocabulary Analysis**: Explore the vocabulary of a pre-trained tokenizer like `gpt2` or `bert-base-uncased` to understand its structure and how it handles common and rare words.

---

## 🧠 Further Reading

- **[Hugging Face: "Summary of the tokenizers"](https://huggingface.co/docs/tokenizers/main/en/summary)**: A concise overview of the most common tokenization strategies.
- **[Sennrich et al. (2015), "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909)**: The paper that introduced Byte-Pair Encoding to NLP.
- **[Kudo & Richardson (2018), "SentencePiece: A simple and language independent subword tokenizer"](https://arxiv.org/abs/1808.06226)**: The paper introducing the SentencePiece library. 