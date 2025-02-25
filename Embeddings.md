---
title: "Embeddings"
nav_order: 4
---

# Embeddings

![Module Banner](https://github.com/user-attachments/assets/944f2cce-c66d-4c51-a443-cebc151055ff)
*Vector representations of data in multidimensional space*

## Overview
Embeddings are **numerical vector representations** of objects like words, images, or items, capturing their semantic meaning and relationships in a continuous vector space. They serve as a **translator**, converting data into a numerical code that machine learning models can understand. This is crucial in natural language processing (NLP) and large language models (LLMs), enabling machines to interpret and manipulate human language effectively. 


## 1. Definition, Purpose, and Types of Embeddings
Embeddings transform **non-numeric data into a format that neural networks can process**. They capture the semantic meaning of text, ensuring that semantically similar words are positioned close together in the vector space.
Embeddings can be categorized into several types, each serving different purposes:
- **Word Embeddings**: Represent individual words as vectors.
- **Contextual Embeddings**: Capture the meaning of a word based on its context in a sentence.
- **Sentence/Paragraph Embeddings**: Represent entire sentences or paragraphs.
- **Multimodal Embeddings**: Capture both textual and visual representations, useful in multimodal LLMs.

### Learning Materials
- **[📄 Medium Article: Static Embeddings](https://medium.com/@mshojaei77/from-words-to-vectors-a-gentle-introduction-to-word-embeddings-eaadb1654778)**
  - *From Words to Vectors: A Gentle Introduction to Word Embeddings*
- **[📄 Medium Article: Contextual Embeddings](https://medium.com/@mshojaei77/beyond-one-word-one-meaning-contextual-embeddings-187b48c6fc27)**
  - *Beyond "One-Word, One-Meaning": A Deep Dive into Contextual Embeddings*
- **[📄 Medium Article: Sentence Embeddings](https://towardsdatascience.com/sentence-embeddings-in-nlp-a-complete-guide-4d7c39c5196)**
  - *Exploring Sentence Embeddings and Their Applications*
- **[📄 Medium Article: Multimodal Embeddings](https://arxiv.org/abs/2103.00020)**
  - *Multimodal Embeddings: Bridging Text and Image Data*
- **[🟠 Colab Notebook: Hands-on Experiences with Embedding Models](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word_embeddings.ipynb)**
  - *Hands-on Experiences with Embeddings*

## 2. Training Embeddings
Embeddings are created through various methods, primarily using deep learning models that understand context and semantics:

- **Training Process**: Initially, vectors are randomly initialized, and the training process assigns values that enable useful behavior.
- **Contrastive Learning**: Models learn from similar and dissimilar pairs of documents to understand their relationships.
- **Fine-Tuning**: Pre-trained embedding models can be fine-tuned to adapt to specific jargon and nuances of a domain.
- **Dimensionality Reduction**: Techniques like UMAP and t-SNE are used to visualize embeddings by projecting them into 2D or 3D space.

### Learning Materials
- **[🟠 Colab Notebook: Word2Vec Implementation](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb)**
  - *Hands-on implementation of Word2Vec from scratch using TensorFlow*
- **[🟠 Colab Notebook: BERT Embeddings with Transformers](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/extract_bert_embeddings.ipynb)**
  - *Extracting and analyzing BERT embeddings with Hugging Face Transformers*
- **[🟠 Colab Notebook: Contrastive Learning for Sentence Embeddings](https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/training/contrastive/contrastive_learning_training.ipynb)**
  - *Training sentence embeddings using contrastive learning objectives*
- **[🟠 Colab Notebook: Fine-tuning Sentence Transformers](https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.ipynb)**
  - *Fine-tuning Sentence Transformers for semantic similarity tasks*
- **[🟠 Colab Notebook: Domain-Specific Embeddings](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)**
  - *Fine-tuning language models for domain-specific embedding generation*
- **[🟠 Colab Notebook: Visualizing Embeddings with UMAP](https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tensorboard_projector_plugin.ipynb)**
  - *Using UMAP and TensorBoard for dimensionality reduction and visualization of embeddings*
- **[🟠 Colab Notebook: t-SNE Visualization of Word Embeddings](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word_embeddings.ipynb#scrollTo=JgXok9E1NVhy)**
  - *Implementing t-SNE to visualize relationships between word vectors*
- **[🟠 Colab Notebook: Multilingual Embeddings Training](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification_multilingual.ipynb)**
  - *Working with and fine-tuning multilingual embedding models*

## 3. Applications of Embeddings
Embeddings have a wide range of applications, including:

- **Semantic Search**: Developing a semantic search engine using LLM embeddings.
- **Document Clustering**: Performing clustering with embedding models.
- **RAG (Retrieval-Augmented Generation)**: Combining embeddings with retrieval to pull relevant information when generating text.

### Learning Materials
- **[📄 Medium Article: Building Semantic Search](https://towardsdatascience.com/semantic-search-with-embeddings-efb59cb9ec4a)**
  - *Implementing Semantic Search with Vector Embeddings*
- **[🟠 Colab Notebook: Document Clustering with Embeddings](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/nlp/semantic-search/semantic-search.ipynb)**
  - *Clustering Documents Using Sentence Embeddings*
- **[🟠 Colab Notebook: RAG Implementation](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/question_answering/how_to/vector_db_qa.ipynb)**
  - *Building a Retrieval-Augmented Generation System with Embeddings*

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