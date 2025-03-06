---
title: "Embeddings"
parent: Foundations
nav_order: 2
layout: default
---

# Embeddings

![Module Banner](https://github.com/user-attachments/assets/944f2cce-c66d-4c51-a443-cebc151055ff)
*Vector representations of data in multidimensional space*

## Overview
Embeddings are **numerical vector representations** that transform data into meaningful points in a high-dimensional space. Think of them as coordinates that capture the essence and relationships between objects - whether they're words, images, or any other type of data. These vectors serve as a sophisticated **translation layer**, converting complex information into a mathematical format that machine learning models can process and understand.

### Key Concepts
- **Dimensional Meaning**: Each dimension in the embedding space represents different features or aspects of the data
- **Similarity Metrics**: The closer two vectors are in the embedding space, the more semantically similar their corresponding items
- **Learned Representations**: Embeddings are typically learned from data, allowing them to capture nuanced relationships

### Types of Embeddings
1. **Word Embeddings**
   - Transform individual words into vectors (e.g., "cat" → [0.2, -0.5, 0.1])
   - Popular models: Word2Vec, GloVe, FastText
   - Capture semantic relationships like: king - man + woman ≈ queen

2. **Contextual Embeddings**
   - Generate dynamic vectors based on context
   - Same word can have different embeddings in different contexts
   - Examples: BERT, GPT, RoBERTa

3. **Sentence/Document Embeddings**
   - Represent entire text segments as single vectors
   - Preserve semantic meaning across longer contexts
   - Used for document similarity, clustering, and retrieval

### Learning Resources
- **[📄 Medium Article: Word Embeddings Deep Dive](https://medium.com/@mshojaei77/from-words-to-vectors-a-gentle-introduction-to-word-embeddings-eaadb1654778)**
  - *Comprehensive introduction to word vector representations*
- **[📄 Medium Article: Contextual Embedding Guide](https://medium.com/@mshojaei77/beyond-one-word-one-meaning-contextual-embeddings-187b48c6fc27)**
  - *Advanced concepts in context-aware embeddings*
- **[📄 Medium Article: Sentence Embedding Techniques](https://medium.com/@mshojaei77/beyond-words-mastering-sentence-embeddings-for-semantic-nlp-dc852b1382ba)**
  - *Modern approaches to sentence-level embeddings*
- **[🟠 Colab Notebook: Interactive Word2Vec Tutorial](https://colab.research.google.com/drive/1dVkCRF0RKWWSP_QQq79LHNYGhead14d0?usp=sharing)**
  - *Hands-on implementation with detailed explanations*


## Additional Resources

[![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)
[![CS224N Lecture 1 - Intro & Word Vectors](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4)
[![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/)
[![Word2vec from Scratch](https://badgen.net/badge/Blog/Word2vec%20from%20Scratch/pink)](https://jaketae.github.io/study/word2vec/)
[![Contextual Embeddings](https://badgen.net/badge/Paper/Contextual%20Embeddings/purple)](https://www.cs.princeton.edu/courses/archive/spring20/cos598C/lectures/lec3-contextualized-word-embeddings.pdf)
[![Training Sentence Transformers](https://badgen.net/badge/Blog/Training%20Sentence%20Transformers/pink)](https://huggingface.co/blog/train-sentence-transformers)
[![BERT Paper](https://badgen.net/badge/Paper/BERT%20Paper/purple)](https://arxiv.org/abs/2204.03503)
[![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Paper/purple)](https://www.semanticscholar.org/paper/67b692bbfd29c5a30cfd1046efd5f85eecd1ea86)
[![FastText Paper](https://badgen.net/badge/Paper/FastText%20Paper/purple)](https://www.semanticscholar.org/paper/d23e59abcae6ba653ba45dcc0ef975438890a3a4)
[![Multilingual BERT Paper](https://badgen.net/badge/Paper/Multilingual%20BERT%20Paper/purple)](https://www.semanticscholar.org/paper/0b0bc70b48aebe608d53a955990cb08f73de5a7d)
[![Bias in Contextualized Word Embeddings](https://badgen.net/badge/Paper/Bias%20in%20Contextualized%20Embeddings/purple)](https://www.semanticscholar.org/paper/5ea2104a039921633f75a9f4b986b515ddbe96d7)