---
layout: article
lang: en
title: "How Large Language Models Work: A Practical Introduction"
description: "Follow text through tokenization, transformer layers, next-token prediction, decoding, and post-training to understand what an LLM can and cannot do."
date: 2025-03-16 21:56:17 +0000
permalink: /writing/intro-to-llms/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# How Large Language Models Work: A Practical Introduction

An LLM does not look up a finished answer in a table. It turns the input into token IDs, processes them through learned layers, and predicts a distribution over the next token. Repeating that process produces text. This simple loop explains much of the model's behavior, but the deployed assistant also depends on its architecture, training data, decoding settings, system instructions, and tools.

## From text to token IDs

The tokenizer splits or encodes the input into pieces from a model-specific vocabulary. Each piece becomes an integer ID that is mapped to a learned vector. Punctuation, whitespace, uncommon names, code, and Persian spelling variants can all affect the token sequence. The exact tokenizer and chat template are part of the model contract; see the [tokenization guide](/writing/tokenization/) for the details.

The model adds information about token position and applies a stack of neural-network layers. In a Transformer, self-attention lets each position combine information from other allowed positions, while feed-forward sublayers transform the resulting representations. During generation, a causal mask prevents a decoder-only model from attending to future tokens.

## Transformer does not mean one architecture

The original Transformer was an encoder-decoder architecture for sequence-to-sequence tasks. Later models reused Transformer components in different arrangements:

- **Encoder-only models**, such as BERT, read a sequence bidirectionally and are commonly adapted for classification or representation tasks.
- **Decoder-only models**, such as GPT-style models, predict the next token from preceding context and are common for text generation.
- **Encoder-decoder models**, such as T5, encode an input and generate a target sequence, which supports text-to-text tasks.

These categories describe architecture, not a universal quality ranking. A model's training objective and task setup matter as much as the broad label.

## What next-token prediction teaches

Many generative language models are pretrained by predicting the next token in large text sequences. The objective supplies a learning signal across the sequence: the model adjusts its weights to assign higher probability to the observed next token, given previous context.

This training can produce useful patterns of syntax, facts, and task behavior, but it does not turn the weights into a clean database of verified facts. Text prediction rewards plausible continuation under the training objective. A fluent answer can still be unsupported, stale, or wrong, especially when the prompt asks for information outside the context or training distribution.

“Large” has no single parameter-count threshold that applies to every model. Parameter count is one scale measure; architecture, training data, compute, tokenization, and inference setup also affect what a model can do and what it costs to run.

## From a distribution to generated text

At each generation step, the model produces scores for possible next tokens. A decoding rule selects one of them. Greedy decoding chooses the highest-scoring token; sampling can draw among likely candidates, with settings such as temperature changing how concentrated the distribution is. The selected token is appended to the context and the model runs again until it reaches a stop condition or a length limit.

This is why a prompt can produce different outputs across runs and why generation length affects latency and cost. The model does not plan a complete response in one operation; it produces a sequence through repeated inference steps.

## A pretrained model is not yet a product assistant

Pretraining teaches statistical patterns from data. Post-training can then teach instruction following, preferred answer formats, refusal behavior, or domain-specific skills. Supervised fine-tuning uses examples of desired responses; preference methods and reinforcement learning use other feedback signals. The exact recipe varies by model, and those stages do not guarantee factual accuracy or safe behavior in every application.

An assistant application may add a system prompt, retrieval, tools, memory, policy checks, or a human approval step. These are system components around the model. Retrieval can provide current documents; a tool can perform a calculation or query a service; a policy layer can restrict an action. The model proposes or interprets, while the surrounding software determines what actually executes and how success is checked.

## Context windows and inference cost

The context window limits how much tokenized input and generated output the model can process in one request. A longer context is not the same as persistent memory: content outside the current context is not automatically available to the next request. Applications often need retrieval or explicit state storage for information that must survive across sessions.

For a decoder-only model, processing the prompt is commonly called **prefill**. Generating later tokens is **decode**. During decode, a key-value cache can avoid recomputing attention projections for earlier tokens, but the cache consumes memory and grows with sequence length and model configuration. Actual latency depends on the model, hardware, batch size, context length, and serving implementation.

## How to reason about an LLM system

When a model fails, locate the boundary before changing the model. Check whether the tokenizer preserved the input, the right context was retrieved, the prompt and tool schema were rendered correctly, the output format was parsed, and the application verified the result. A model benchmark alone does not test that entire path.

For a production decision, define the task and failure cost first. Then compare a language model with simpler options such as deterministic code, search, a classifier, or a constrained decision model. Use an LLM where language ambiguity or flexible generation is useful; keep rules, permissions, and outcome checks in software when they can be specified explicitly.

## The practical mental model

An LLM maps tokenized context to a next-token distribution and repeats that computation to generate a sequence. Its behavior reflects learned weights and inference settings, but the usable product is a larger system that supplies context, applies controls, and checks results. Understanding those boundaries makes it easier to choose a model, diagnose an error, and decide when a model is not the right tool.

## Sources

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
- Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683).
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
