
# Module 3: NLP Fundamentals

![NLP Fundamentals](image_url)
*Understanding the building blocks of Natural Language Processing*

## Overview
This module covers essential Natural Language Processing concepts and techniques, focusing on text processing, word representations, and language modeling fundamentals crucial for understanding LLMs.

## 1. Tokenization
Learn various tokenization methods to convert text into model-readable tokens. This fundamental NLP concept is crucial for understanding how language models process and interpret text data.

### Core Materials
- **[📘 Colab Notebook: BPE Implementation](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing)**
  - *Hands-on implementation of Simple Byte Pair Encoding*
- **[📘 Colab Notebook: Hugging Face Tokenizers](https://colab.research.google.com/drive/1mcFgQ9PX1TFyEAsFOnoS1ozeSz3vM6A1?usp=sharing)**
  - *Working with Hugging Face tokenization tools*
- **[📘 Colab Notebook: Custom Tokenizer](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing)**
  - *Building and customizing your own tokenizer*
- **[📘 Colab Notebook: New Tokenizer Training](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing)**
  - *Training a new tokenizer from scratch*
- **[📘 Colab Notebook: GPT Tokenizer](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)**
  - *Understanding GPT tokenization approach*
- **[📘 Colab Notebook: Tokenizer Comparison](https://colab.research.google.com/drive/1wVSCBGFm7KjJy-KugYGYETpncWsPgx5N?usp=sharing)**
  - *Comparing different tokenization strategies*

### Additional Resources
**Playgrounds:**

[![TikTokenizer](https://badgen.net/badge/Playground/TikTokenizer/blue)](https://tiktokenizer.vercel.app/)
[![Hugging Face Tokenizer](https://badgen.net/badge/Playground/HF%20Tokenizer/blue)](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)
[![OpenAI Tokenizer](https://badgen.net/badge/Playground/OpenAI%20Tokenizer/blue)](https://platform.openai.com/tokenizer)
[![Tokenizer Arena](https://badgen.net/badge/Playground/Tokenizer%20Arena/blue)](https://huggingface.co/spaces/Cognitive-Lab/Tokenizer_Arena)

**Learning Resources:**

[![GPT Tokenizer Implementation](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=zduSFxRajkE&t=4341s)
[![Tokenization Fundamentals](https://badgen.net/badge/Course/Tokenization%20Fundamentals/orange)](https://huggingface.co/learn/nlp-course/chapter2/4)
[![Stanford's CoreNLP](https://badgen.net/badge/Course/Stanford%20CS224N%20Subword%20Models/orange)](https://stanfordnlp.github.io/CoreNLP/tokenize.html)
[![SentencePiece Training Guide](https://badgen.net/badge/Docs/SentencePiece%20Training%20Guide/green)](https://github.com/google/sentencepiece#train-sentencepiece-model)
[![Tokenizer Shrinking Guide](https://badgen.net/badge/Guide/Tokenizer%20Shrinking%20Techniques/blue)](https://github.com/stas00/ml-engineering/blob/master/transformers/make-tiny-models.md)

## 2. Word Embeddings & Contextual Representations
Explore dense vector representations of words that capture semantic and syntactic relationships, from static embeddings to context-aware representations. This section covers fundamental concepts essential for modern language understanding systems.

### Core Materials
- **[📘 Colab Notebook: Word2Vec Implementation](https://colab.research.google.com/drive/yournotebooklink3)**
  - *Implementing Word2Vec from scratch*
- **[📘 Colab Notebook: GloVe Implementation](https://colab.research.google.com/drive/yournotebooklink4)**
  - *Working with GloVe embeddings*
- **[📘 Colab Notebook: Sentence Transformer Fine-tuning](https://colab.research.google.com/drive/yournotebooklink_sentence_transformers)**
  - *Fine-tuning sentence transformers*
- **[📘 Colab Notebook: BERT Embeddings Exploration](https://colab.research.google.com/drive/yournotebooklink_bert_exploration)**
  - *Exploring BERT embeddings*

### Additional Resources
**Learning Resources:**

[![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)
[![CS224N Lecture 1](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4)
[![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/)
[![Training Sentence Transformers](https://badgen.net/badge/Blog/Training%20Sentence%20Transformers/pink)](https://huggingface.co/blog/train-sentence-transformers)

**Research Papers:**

[![BERT Paper](https://badgen.net/badge/Paper/BERT%20Paper/purple)](https://arxiv.org/abs/2204.03503)
[![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Paper/purple)](https://www.semanticscholar.org/paper/67b692bbfd29c5a30cfd1046efd5f85eecd1ea86)
[![FastText Paper](https://badgen.net/badge/Paper/FastText%20Paper/purple)](https://www.semanticscholar.org/paper/d23e59abcae6ba653ba45dcc0ef975438890a3a4)
[![Multilingual BERT Paper](https://badgen.net/badge/Paper/Multilingual%20BERT%20Paper/purple)](https://www.semanticscholar.org/paper/0b0bc70b48aebe608d53a955990cb08f73de5a7d)
[![Bias in Embeddings](https://badgen.net/badge/Paper/Bias%20in%20Contextualized%20Embeddings/purple)](https://www.semanticscholar.org/paper/5ea2104a039921633f75a9f4b986b515ddbe96d7)

**Additional Guides:**

[![Oddly Satisfying Deep Learning](https://badgen.net/badge/Book/Oddly%20Satisfying%20Deep%20Learning/blue)](https://pythonandml.github.io/dlbook/content/word_embeddings/traditional_word_embeddings.html)
[![Instructor Embeddings](https://badgen.net/badge/Guide/Instructor%20Embeddings/blue)](https://huggingface.co/hkunlp/instructor-large)
[![Custom LLM Embedding Training](https://badgen.net/badge/Tutorial/Custom%20LLM%20Embedding%20Training/blue)](https://dagshub.com/blog/how-to-train-a-custom-llm-embedding-model/)
[![Word Embeddings Guide](https://badgen.net/badge/Guide/Word%20Embeddings%20Guide/blue)](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp)

## 3. Language Modeling Basics
Understanding fundamental concepts of statistical language modeling and sequence prediction. This section provides the foundation for understanding how modern language models work.

### Core Materials
- **[📘 Colab Notebook: N-Gram Language Modeling](https://colab.research.google.com/drive/yournotebooklink5)**
  - *Implementation of N-gram language models*
- **[📘 Colab Notebook: Probabilistic Language Modeling](https://colab.research.google.com/drive/yournotebooklink6)**
  - *Working with probabilistic language models*

### Additional Resources
[![N-Gram Language Modeling Guide](https://badgen.net/badge/Tutorial/N-Gram%20Language%20Modeling%20Guide/blue)](https://www.geeksforgeeks.org/n-gram-language-modeling/)
[![Dense LLM Lecture](https://badgen.net/badge/Video/Dense%20LLM%20Lecture/red)](https://youtu.be/9vM4p9NN0Ts)
[![Stanford CS224N](https://badgen.net/badge/Course/Stanford%20CS224N/orange)](https://web.stanford.edu/class/cs224n/)
[![Stanford CS229](https://badgen.net/badge/Course/Stanford%20CS229/orange)](https://cs229.stanford.edu/)