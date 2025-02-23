---
title: "Embeddings"
nav_order: 4
---

# Embeddings

![Module Banner](https://github.com/user-attachments/assets/944f2cce-c66d-4c51-a443-cebc151055ff)
*Image caption or credit*

## Overview
Embeddings are **numerical vector representations** of objects like words, images, or items, capturing their semantic meaning and relationships in a continuous vector space. They serve as a **translator**, converting data into a numerical code that machine learning models can understand. This is crucial in natural language processing (NLP) and large language models (LLMs), enabling machines to interpret and manipulate human language effectively. 


## 1. Definition, Purpose, and Types of Embeddings
Embeddings transform **non-numeric data into a format that neural networks can process**. They capture the semantic meaning of text, ensuring that semantically similar words are positioned close together in the vector space.
Embeddings can be categorized into several types, each serving different purposes:
- **Word Embeddings**: Represent individual words as vectors.
- **Token Embeddings**: Represent individual tokens, which are small chunks of text.
- **Contextualized Word Embeddings**: Capture the meaning of a word based on its context in a sentence.
- **Positional Embeddings**: Encode the position of words in a sequence, crucial for LLMs' self-attention mechanisms.
- **Sentence/Paragraph Embeddings**: Represent entire sentences or paragraphs.
- **Multimodal Embeddings**: Capture both textual and visual representations, useful in multimodal LLMs.

### Learning Materials
- **[📄 Medium Article: Word Embeddings](https://medium.com/@mshojaei77/from-words-to-vectors-a-gentle-introduction-to-word-embeddings-eaadb1654778)**
  - *From Words to Vectors: A Gentle Introduction to Word Embeddings*
- **[📄 Medium Article: Token Embeddings](https://towardsdatascience.com/token-embeddings-what-are-they-and-why-do-they-matter-1c1c1e1c1c1c)**
  - *Understanding Token Embeddings and Their Importance in NLP*
- **[📄 Medium Article: Contextualized Word Embeddings](https://towardsdatascience.com/contextualized-word-embeddings-what-are-they-and-how-do-they-work-1c1c1e1c1c1c)**
  - *A Deep Dive into Contextualized Word Embeddings*
- **[📄 Medium Article: Positional Embeddings](https://arxiv.org/abs/1706.03762)**
  - *Attention is All You Need: Understanding Positional Embeddings*
- **[📄 Medium Article: Sentence Embeddings](https://towardsdatascience.com/sentence-embeddings-what-are-they-and-how-to-use-them-1c1c1e1c1c1c)**
  - *Exploring Sentence Embeddings and Their Applications*
- **[📄 Medium Article: Multimodal Embeddings](https://arxiv.org/abs/2105.00001)**
  - *Multimodal Embeddings: Bridging Text and Image Data*
- **[🟠 Colab Notebook: Word Embeddings Implementation](https://colab.research.google.com/drive/1a2b3c4d5e6f7g8h9i0j)**
  - *Hands-on with Word Embeddings: A Simple Implementation*
- **[🟠 Colab Notebook: Token Embeddings Implementation](https://colab.research.google.com/drive/1j2k3l4m5n6o7p8q9r0s)**
  - *Exploring Token Embeddings: A Practical Guide*
- **[🟠 Colab Notebook: Contextualized Word Embeddings Implementation](https://colab.research.google.com/drive/1t2u3v4w5x6y7z8a9b0c)**
  - *Implementing Contextualized Word Embeddings in Python*
- **[🟠 Colab Notebook: Positional Embeddings Implementation](https://colab.research.google.com/drive/1d2e3f4g5h6i7j8k9l0m)**
  - *Understanding Positional Embeddings through Code*
- **[🟠 Colab Notebook: Sentence Embeddings Implementation](https://colab.research.google.com/drive/1n2o3p4q5r6s7t8u9v0w)**
  - *Creating Sentence Embeddings: A Step-by-Step Guide*
- **[🟠 Colab Notebook: Multimodal Embeddings Implementation](https://colab.research.google.com/drive/1x2y3z4a5b6c7d8e9f0g)**
  - *Working with Multimodal Embeddings: An Interactive Notebook*

## 2. Creating Embeddings
Embeddings are created through various methods, primarily using deep learning models that understand context and semantics:

- **Training Process**: Initially, vectors are randomly initialized, and the training process assigns values that enable useful behavior.
- **Contrastive Learning**: Models learn from similar and dissimilar pairs of documents to understand their relationships.

### Learning Materials
- **[📄 Medium Article: Topic 1 Theory](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Simple Implementation](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Advanced Implementation](url)**
  - *Brief description*

## 3. Advanced Techniques and Considerations
When working with embeddings, consider the following advanced techniques:

- **Fine-Tuning**: Pre-trained embedding models can be fine-tuned to adapt to specific jargon and nuances of a domain.
- **Dimensionality Reduction**: Techniques like UMAP and t-SNE are used to visualize embeddings by projecting them into 2D or 3D space.

### Learning Materials
- **[📄 Medium Article: Topic 1 Theory](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Simple Implementation](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Advanced Implementation](url)**
  - *Brief description*

## 5. Applications of Embeddings
Embeddings have a wide range of applications, including:

- **Semantic Search**: Developing a semantic search engine using LLM embeddings.
- **Document Clustering**: Performing clustering with embedding models.
- **RAG (Retrieval-Augmented Generation)**: Combining embeddings with retrieval to pull relevant information when generating text.

### Learning Materials
- **[📄 Medium Article: Topic 1 Theory](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Simple Implementation](url)**
  - *Brief description*
- **[🟠 Colab Notebook: Topic 1 Advanced Implementation](url)**
  - *Brief description*

## Additional Resources
[![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)
[![CS224N Lecture 1 - Intro & Word Vectors](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4)
[![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/)
[![Contextual Embeddings](https://badgen.net/badge/Paper/Contextual%20Embeddings/purple)](https://www.cs.princeton.edu/courses/archive/spring20/cos598C/lectures/lec3-contextualized-word-embeddings.pdf)
[![Training Sentence Transformers](https://badgen.net/badge/Blog/Training%20Sentence%20Transformers/pink)](https://huggingface.co/blog/train-sentence-transformers)
[![BERT Paper](https://badgen.net/badge/Paper/BERT%20Paper/purple)](https://arxiv.org/abs/2204.03503)
[![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Paper/purple)](https://www.semanticscholar.org/paper/67b692bbfd29c5a30cfd1046efd5f85eecd1ea86)
[![FastText Paper](https://badgen.net/badge/Paper/FastText%20Paper/purple)](https://www.semanticscholar.org/paper/d23e59abcae6ba653ba45dcc0ef975438890a3a4)
[![Multilingual BERT Paper](https://badgen.net/badge/Paper/Multilingual%20BERT%20Paper/purple)](https://www.semanticscholar.org/paper/0b0bc70b48aebe608d53a955990cb08f73de5a7d)
[![Bias in Contextualized Word Embeddings](https://badgen.net/badge/Paper/Bias%20in%20Contextualized%20Embeddings/purple)](https://www.semanticscholar.org/paper/5ea2104a039921633f75a9f4b986b515ddbe96d7)