---
title: "Tokenization Tutorial: Text to Tokens for LLMs"
layout: article
lang: en
permalink: /writing/tokenization/
nav_exclude: true
description: "Follow an LLM tokenization pipeline and compare character, word, subword and byte-level approaches, including the basic steps of BPE."
date: 2026-11-26 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Tokenization Tutorial 🔢: Text to Numbers Hack for LLM Smarts
Master tokenization guide – the bottleneck for AI language models. Turn text into tokens efficiently.

## Why Tokenization Matters in LLMs
- Arithmetic fails? Blame poor token splits.
- String manipulations? Depends on smart tokenization.

## Tokenization Pipeline
1. Normalize: Clean and standardize text.
2. Pre-tokenize: Split into words or subwords.
3. Model: Apply BPE explained or similar to get IDs.
4. Post-process: Add specials like [BOS], [EOS].

## Tokenization Paradigms
- Character-level: Universal but leads to long sequences.
- Word-level: Semantic but handles OOV poorly.
- Subword: BPE balance for efficiency.
- Byte-level: Raw UTF-8 for multilingual support.

## BPE Steps Explained
- Initialize with characters.
- Count pair frequencies.
- Merge most frequent pairs.
- Repeat until vocabulary size reached.

## My Tokenization Notes
- Token Theory
- [Colab: Tokenization Techniques](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing)

## Top Tokenization Resources
- [Mistral Tokenization Guide](https://docs.mistral.ai/guides/tokenization/)
- [Hugging Face Pipeline](https://huggingface.co/docs/tokenizers/en/pipeline)
- [Hugging Face Tokenizer Playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
- [Airbyte LLM Tokenization Guide](https://airbyte.com/data-engineering-resources/llm-tokenization)
- [Tiktokenizer](https://tiktokenizer.vercel.app/)
- [MIT Paper on Tokenization](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00557/128327)
- BPE Tutorial
- [BPE Original Paper](https://arxiv.org/abs/1508.07909)
- [Karpathy Video on Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE)
- [MinBPE GitHub Repo](https://github.com/karpathy/minbpe)

What's your trick for handling rare words in tokenization? Share! 🤓

Keywords: tokenization tutorial, BPE explained, LLM tokenization guide, text to tokens, subword tokenization
