---
title: "NLP Fundamentals"
nav_order: 3
---

# NLP Fundamentals

# NLP Fundamentals

![NLP Fundamentals](image_url)

## Overview
This module covers essential Natural Language Processing concepts and techniques, focusing on text processing, word representations, and language modeling fundamentals crucial for understanding LLMs.

## 1. Tokenization Strategies
Learn various tokenization methods to convert text into model-readable tokens.

### Core Materials
**Hands-on Practice:**
- [BPE Implementation](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing)
- [Hugging Face Tokenizers](https://colab.research.google.com/drive/1mcFgQ9PX1TFyEAsFOnoS1ozeSz3vM6A1?usp=sharing)
- [Custom Tokenizer](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing)
- [New Tokenizer Training](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing)
- [GPT Tokenizer](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing)
- [Tokenizer Comparison](https://colab.research.google.com/drive/1wVSCBGFm7KjJy-KugYGYETpncWsPgx5N?usp=sharing)

### Key Concepts
- Byte Pair Encoding (BPE)
- WordPiece Tokenization
- Unigram Tokenization
- Custom Tokenizers
- Domain-specific Tokenization
- Vocabulary Optimization

### Tools & Frameworks
**Playgrounds:**
- [![TikTokenizer](https://badgen.net/badge/Playground/TikTokenizer/blue)](https://tiktokenizer.vercel.app/)
- [![Hugging Face Tokenizer](https://badgen.net/badge/Playground/HF%20Tokenizer/blue)](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)
- [![OpenAI Tokenizer](https://badgen.net/badge/Playground/OpenAI%20Tokenizer/blue)](https://platform.openai.com/tokenizer)
- [![Tokenizer Arena](https://badgen.net/badge/Playground/Tokenizer%20Arena/blue)](https://huggingface.co/spaces/Cognitive-Lab/Tokenizer_Arena)

**Libraries:**
- [![Hugging Face Tokenizers](https://badgen.net/badge/Library/HF%20Tokenizers/green)](https://github.com/huggingface/tokenizers)
- [![SentencePiece](https://badgen.net/badge/Library/SentencePiece/green)](https://github.com/google/sentencepiece)
- [![Tiktoken](https://badgen.net/badge/Library/Tiktoken/green)](https://github.com/openai/tiktoken)
- [![spaCy](https://badgen.net/badge/Library/spaCy/green)](https://spacy.io/)
- [![Mistral Tokenizer](https://badgen.net/badge/Library/Mistral%20Tokenizer/green)](https://docs.mistral.ai/guides/tokenization/)

### Additional Resources
- [![GPT Tokenizer Implementation](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=zduSFxRajkE&t=4341s) 
- [![Tokenization Fundamentals](https://badgen.net/badge/Course/Tokenization%20Fundamentals/orange)](https://huggingface.co/learn/nlp-course/chapter2/4) 
- [![Stanford's CoreNLP](https://badgen.net/badge/Course/Stanford%20CS224N%20Subword%20Models/orange)](https://stanfordnlp.github.io/CoreNLP/tokenize.html) 
- [![SentencePiece Training Guide](https://badgen.net/badge/Docs/SentencePiece%20Training%20Guide/green)](https://github.com/google/sentencepiece#train-sentencepiece-model) 
- [![Tokenizer Shrinking Guide](https://badgen.net/badge/Guide/Tokenizer%20Shrinking%20Techniques/blue)](https://github.com/stas00/ml-engineering/blob/master/transformers/make-tiny-models.md) 

## 2. Word Embeddings & Contextual Representations
Exploring dense vector representations of words that capture semantic and syntactic relationships, from static embeddings to context-aware representations.

### Core Materials
**Hands-on Practice:**
- [![Word2Vec Implementation](https://badgen.net/badge/Colab%20Notebook/Word2Vec%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink3)
- [![GloVe Implementation](https://badgen.net/badge/Colab%20Notebook/GloVe%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink4)
- [![Sentence Transformer Fine-tuning](https://badgen.net/badge/Colab%20Notebook/Sentence%20Transformer%20Fine-tuning/orange)](https://colab.research.google.com/drive/yournotebooklink_sentence_transformers)
- [![BERT Embeddings Exploration](https://badgen.net/badge/Colab%20Notebook/BERT%20Embeddings%20Exploration/orange)](https://colab.research.google.com/drive/yournotebooklink_bert_exploration)

### Key Concepts
**Static Word Embeddings:**
- Word2Vec Models (Skip-gram & CBOW)
- GloVe Embeddings
- Sparse Representations
- Cosine Similarity

**Contextual Word Embeddings:**
- Contextual Embeddings
- Transformer-based Models
- Vector Representations
- Semantic Similarity
- Word Embedding Models

**Advanced Concepts:**
- Subword Information (FastText)
- Multilingual and Cross-lingual Embeddings
- Domain-Specific Embeddings
- Bias Detection and Mitigation

### Additional Resources
**Learning Resources:**
- [![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)
- [![CS224N Lecture 1](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4)
- [![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/)
- [![Training Sentence Transformers](https://badgen.net/badge/Blog/Training%20Sentence%20Transformers/pink)](https://huggingface.co/blog/train-sentence-transformers)

**Research Papers:**
- [![BERT Paper](https://badgen.net/badge/Paper/BERT%20Paper/purple)](https://arxiv.org/abs/2204.03503)
- [![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Paper/purple)](https://www.semanticscholar.org/paper/67b692bbfd29c5a30cfd1046efd5f85eecd1ea86)
- [![FastText Paper](https://badgen.net/badge/Paper/FastText%20Paper/purple)](https://www.semanticscholar.org/paper/d23e59abcae6ba653ba45dcc0ef975438890a3a4)
- [![Multilingual BERT Paper](https://badgen.net/badge/Paper/Multilingual%20BERT%20Paper/purple)](https://www.semanticscholar.org/paper/0b0bc70b48aebe608d53a955990cb08f73de5a7d)
- [![Bias in Embeddings](https://badgen.net/badge/Paper/Bias%20in%20Contextualized%20Embeddings/purple)](https://www.semanticscholar.org/paper/5ea2104a039921633f75a9f4b986b515ddbe96d7)

**Additional Guides:**
- [![Oddly Satisfying Deep Learning](https://badgen.net/badge/Book/Oddly%20Satisfying%20Deep%20Learning/blue)](https://pythonandml.github.io/dlbook/content/word_embeddings/traditional_word_embeddings.html)
- [![Instructor Embeddings](https://badgen.net/badge/Guide/Instructor%20Embeddings/blue)](https://huggingface.co/hkunlp/instructor-large)
- [![Custom LLM Embedding Training](https://badgen.net/badge/Tutorial/Custom%20LLM%20Embedding%20Training/blue)](https://dagshub.com/blog/how-to-train-a-custom-llm-embedding-model/)
- [![Word Embeddings Guide](https://badgen.net/badge/Guide/Word%20Embeddings%20Guide/blue)](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp)

## 3. Language Modeling Basics
Understanding fundamental concepts of statistical language modeling and sequence prediction.

### Core Materials
**Hands-on Practice:**
- [![N-Gram Language Modeling](https://badgen.net/badge/Colab%20Notebook/N-Gram%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink5)
- [![Probabilistic Language Modeling](https://badgen.net/badge/Colab%20Notebook/Probabilistic%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink6)

### Key Concepts
- Language Modeling
- N-gram Models
- Probabilistic Models
- Next-word Prediction
- Model Architecture
- Training Approaches

### Additional Resources
- [![N-Gram Language Modeling Guide](https://badgen.net/badge/Tutorial/N-Gram%20Language%20Modeling%20Guide/blue)](https://www.geeksforgeeks.org/n-gram-language-modeling/)
- [![Dense LLM Lecture](https://badgen.net/badge/Video/Dense%20LLM%20Lecture/red)](https://youtu.be/9vM4p9NN0Ts)
- [![Stanford CS224N](https://badgen.net/badge/Course/Stanford%20CS224N/orange)](https://web.stanford.edu/class/cs224n/)
- [![Stanford CS229](https://badgen.net/badge/Course/Stanford%20CS229/orange)](https://cs229.stanford.edu/)
