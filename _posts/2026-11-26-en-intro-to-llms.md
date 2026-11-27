---
title: "Intro to LLMs: From Next-Token Prediction to Chat Assistants"
layout: article
lang: en
permalink: /writing/intro-to-llms/
nav_exclude: true
description: "An introduction to large language models: how token prediction works, how pretraining and post-training shape chat behavior, and where LLM outputs can fail."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Intro to Large Language Models

A language model assigns probabilities to text sequences. A large language model (LLM) is a language model trained at a scale that supports a broad range of language tasks; there is no universal parameter-count cutoff for “large.” The model does not receive words as people see them: a tokenizer maps text into token IDs, and the model processes those IDs. (See [how tokenization works](/writing/tokenization/).)

## The next-token objective

Many generative LLMs are autoregressive: they estimate the next token from the preceding context. Given a sequence, training presents each position with the earlier tokens and asks the model to assign high probability to the actual next token. In simplified notation, the probability of a sequence is factored as:

```text
p(token_1, ..., token_T) = product_t p(token_t | token_1, ..., token_(t-1))
```

For a toy prompt such as `The kettle began to`, a model might assign different probabilities to `whistle`, `boil`, or `sing`. These are example candidates, not measured model outputs. A training loss penalizes the model when it assigns too little probability to the token that occurred in the training text. Repeating this over many sequences adjusts the model's parameters.

At generation time, the model uses its current context to produce a probability distribution for the next token. A decoding method selects one token, appends it to the context, and repeats. Temperature and sampling rules can affect which token is selected. The model predicts tokens, not sentences all at once, and different decoding choices can yield different continuations.

## Why Transformers matter

Modern generative LLMs commonly use decoder-only Transformer architectures. Self-attention lets a position combine information from earlier positions; feed-forward layers transform each position's representation. A causal mask prevents the model from using future tokens when predicting the next one. The sequence of token IDs is converted to vectors, processed by Transformer layers, and projected into scores over the tokenizer's vocabulary.

The original Transformer paper described an encoder-decoder model for translation. Transformer is an architecture family, not a synonym for LLM. Some language models use encoder-only architectures and objectives such as predicting masked tokens; these models are useful for representation and classification tasks but are not trained in the same left-to-right generation setup.

## Training and chat behavior

Pretraining teaches broad patterns from large text collections using an objective such as next-token prediction. That objective alone does not guarantee that a model will follow instructions, cite reliable sources, or behave like an assistant. Later training can include supervised examples, preference feedback, or other objectives to shape how the model responds. The exact recipe varies by model.

For example, InstructGPT describes supervised fine-tuning on demonstrations followed by preference modeling and reinforcement learning from human feedback. That is one documented post-training pipeline, not a requirement that every chat model use the same steps.

## What an LLM can and cannot tell you

An LLM can produce fluent continuations, summarize text, transform formats, and generate code because its training has shaped parameters that encode useful patterns. Fluency is not proof that a claim is true. A model may produce an answer that sounds plausible but conflicts with evidence, especially when the prompt is ambiguous, information is missing, or its learned patterns do not cover the case.

The context window limits how much input a model can consider in a request; it is not the same as a durable memory or a guarantee that every included detail will affect the answer. Retrieval can provide external passages in context, while tools can obtain or act on information outside the model. Those components also need evaluation: retrieval may miss the right source, and tool output may be incomplete or wrong. A practical introduction to these system choices is [RAG and document parsing](/fa/writing/document-parsing/); that guide is a distinct Persian edition.

When using an LLM in an application, evaluate it on representative inputs. Check factual correctness, behavior on missing information, and consistency across model or prompt changes. For production systems, also consider privacy, access control, latency, cost, and failure handling.

## A useful learning path

To understand the mechanics, continue with [traditional language models](/writing/traditional-language-models/), [neural networks](/writing/neural-networks/), [tokenization](/writing/tokenization/), and [Transformers](/writing/transformers/). To build an application, study [embeddings](/writing/embeddings/) and the [compact AI engineering roadmap](/writing/mini-roadmap/).

## Further reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original encoder-decoder Transformer.
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), an autoregressive language-model study.
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), an example of masked-token pretraining.
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), an InstructGPT post-training study.
