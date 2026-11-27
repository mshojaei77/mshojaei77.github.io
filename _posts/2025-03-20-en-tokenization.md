---
layout: article
lang: en
title: "Tokenization: From Subwords to Model Input"
description: "How tokenizers turn text into model IDs, why BPE and SentencePiece differ, and what Persian text, chat templates, and token budgets mean in practice."
date: 2025-03-20 20:30:00 +0000
permalink: /writing/tokenization/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Tokenization: From Subwords to Model Input

A language model does not read a sentence as a string. Its tokenizer converts that string into token IDs, and the model processes those IDs. That conversion affects what the model can represent, how long an input is, and how much an API request costs. When a prompt behaves unexpectedly, the tokenizer is part of the debugging surface.

## The tokenizer is part of the model contract

The tokenizer maps text to a sequence of IDs from a fixed vocabulary. Those IDs are not words with universal meanings: they are entries in the vocabulary used when the model was trained. A different tokenizer can split the same text differently and assign different IDs, even if both tokenizers use a familiar algorithm such as BPE.

That is why a model checkpoint and its tokenizer should be treated as a pair. Loading an unrelated tokenizer can produce valid-looking IDs that refer to different vocabulary entries. The model will still run, but it will receive a different sequence from the one its training process used.

Tokenization also includes more than splitting. Implementations can normalize text, pre-segment it, add special tokens, and apply a model-specific chat template. The final input may therefore contain role markers or control tokens that are not visible in the message text. Inspect the exact tokenizer and template when counting tokens or debugging a prompt.

## Why use subword units?

A word-level vocabulary is simple to reason about, but it has a coverage problem: new names, spelling variants, and inflected forms may be missing. A character-level vocabulary covers more strings, but it can turn ordinary text into long sequences.

Subword tokenization is a compromise. Frequent text fragments can become single tokens while rarer words are represented by smaller pieces. The pieces are useful units for prediction and compression; they are not a guarantee that the model understands a word's morphology or meaning. A split that looks linguistically meaningful may simply reflect the tokenizer's training data and merge rules.

## How BPE learns and applies merges

Byte Pair Encoding (BPE) was adapted for neural machine translation as a way to represent rare words with subword units. In a simplified training picture, text begins as small symbols, adjacent pairs are counted, and frequent pairs are merged repeatedly until the target vocabulary or merge budget is reached.

Encoding uses the learned merge priorities to combine eligible neighboring symbols. It does not apply every rule as an arbitrary global replacement. Real tokenizers also have pre-tokenization and byte-handling rules, so a short teaching implementation is not a drop-in replacement for a model tokenizer. OpenAI's `tiktoken`, for example, exposes model-specific encodings and uses byte-level BPE; Hugging Face tokenizers expose their own model-specific preprocessing and special-token behavior.

## SentencePiece is a toolkit, not one algorithm

SentencePiece trains and applies subword tokenizers directly from raw text, without requiring a language-specific word segmenter first. It supports multiple model types, including BPE and Unigram. Its whitespace marker makes spaces representable in the token stream, which is useful when the tokenizer must reconstruct the original text.

So “BPE versus SentencePiece” is not always a comparison between two different segmentation algorithms. It may be a comparison between a particular BPE tokenizer and a SentencePiece implementation using BPE, or between BPE and SentencePiece's Unigram model. Check the vocabulary, normalization, pre-tokenizer, and model type rather than comparing library names alone.

## Persian text needs a tokenizer-specific check

Persian and Arabic characters that look similar can have different Unicode code points, including Yeh and Kaf variants. The zero-width non-joiner, Arabic and Persian digits, diacritics, punctuation, and mixed Latin text can also change token boundaries. A tokenizer may normalize some of these forms, preserve them, or have learned different frequencies for each one.

Do not normalize every document with one global rule just to lower the token count. First inspect the model's expected preprocessing and keep the original text available. If the application normalizes queries, apply the same documented transformation to indexed documents and queries, then evaluate retrieval and task quality. For text generation, normalization can also change spelling, names, legal quotations, or code-switching that the user expects to preserve.

## Measure tokens with the exact runtime path

For a reliable count, load the tokenizer associated with the model and apply the same chat template, special tokens, truncation policy, and tool definitions used by the application. Counting only the visible user message can miss system instructions, conversation history, structured output constraints, and tool schemas.

Token counts matter for context limits, latency, and billing, but the cheapest tokenization is not automatically the best. A normalization that saves a few tokens may damage retrieval or change a name. A tokenizer that splits Persian more finely may still produce good task results if the model was trained for it. Measure end-to-end quality on representative Persian and mixed-language inputs, then compare the token and cost impact under the same request format.

Fast implementations can improve preprocessing throughput, especially when tokenizing large corpora. Their advantage depends on the tokenizer, machine, input mix, batch size, and whether tokenization is actually the pipeline bottleneck. Do not reuse a speedup figure without its benchmark setup; measure the workload that matters to your system.

At the implementation level, [Hugging Face's `tokenizers` library](https://github.com/huggingface/tokenizers) uses Rust as its reference implementation and provides language bindings, including Python. That can reduce interpreter overhead in some pipelines, but the language used for the core does not supply a universal throughput guarantee. Compare equivalent preprocessing, batch sizes, hardware, and outputs; otherwise a benchmark may compare different work.

## A practical debugging checklist

- Confirm that the tokenizer files belong to the model checkpoint.
- Inspect the model's normalization, pre-tokenization, and special-token settings.
- Count the fully rendered chat input, not only the message text.
- Test Persian Yeh/Kaf variants, نیم‌فاصله, digits, punctuation, and mixed Persian-English text.
- Compare both token counts and downstream quality before changing normalization or vocabulary.
- Record the tokenizer revision and template alongside evaluation results.

## The engineering takeaway

Tokenization is a learned interface between text and model weights. BPE and SentencePiece describe parts of that interface, but the deployed tokenizer also includes vocabulary, normalization, special tokens, and formatting rules. Use the exact tokenizer with its model, inspect the final input IDs when behavior is surprising, and optimize token count only after checking that the transformation preserves the task.

## Sources

- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909).
- Kudo and Richardson, [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226).
- [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer).
- [Hugging Face `tokenizers` implementation](https://github.com/huggingface/tokenizers).
- [OpenAI `tiktoken` implementation](https://github.com/openai/tiktoken).
