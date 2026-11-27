---
title: "How LLM Tokenization Works: Subwords, BPE, and Text Cost"
layout: article
lang: en
permalink: /writing/tokenization/
nav_exclude: true
description: "Learn how tokenizers turn text into model input, how BPE differs from tokenization at inference time, and why multilingual text can use different token budgets."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# How LLM Tokenization Works

An LLM does not receive a sentence as words. A tokenizer converts text into a sequence of integer IDs, and the model looks up a learned vector for each ID. That conversion is part of the model interface: changing the tokenizer changes the sequence the model sees, even when the visible text is unchanged.

This matters in practical ways. Token counts affect context limits and, for token-priced APIs, input and output cost. A tokenizer can also split two languages or scripts very differently. Understanding the pipeline helps explain these effects and prevents a common mistake: assuming that all tokenizers simply split on spaces.

## From text to IDs

A tokenizer is usually a pipeline. Depending on the model, it may normalize text, split it into chunks, map chunks to subword pieces, and then add special tokens. The exact order and behavior belong to that tokenizer's saved configuration and vocabulary. Special tokens such as beginning-of-sequence or end-of-turn markers are model-specific; they are not universal punctuation rules.

The output is a list of IDs, for example:

```text
text:      "A small example."
pieces:    ["A", " small", " example", "."]
input_ids: [  32,  1402,      6179, 13]
```

Those pieces and numbers are illustrative. Another model may use different pieces and IDs for the same text. When debugging a prompt, inspect the tokenizer shipped with the exact model or API you are using.

## Why subwords are common

Character tokenization can represent arbitrary text but makes sequences long. Word tokenization gives common words one unit, but needs a large vocabulary and struggles with new spellings and inflections. Subword methods provide a compromise: common character sequences can become single pieces while less frequent forms are split into smaller pieces.

Byte-Pair Encoding (BPE) is one way to learn such a vocabulary. In a simplified training procedure, the tokenizer starts from a base alphabet, counts adjacent pairs in a corpus, merges a frequent pair, and repeats. The resulting merge table is saved. When encoding new text, BPE applies the learned merge rules; it does **not** recount pair frequencies in each prompt.

Other algorithms make different choices. WordPiece selects pieces using a likelihood-oriented objective, while Unigram tokenization learns a vocabulary of candidate pieces and a probabilistic segmentation model. SentencePiece is a toolkit that implements BPE and Unigram models and can train directly from raw sentences. “SentencePiece” and “BPE” therefore name different kinds of things: a toolkit and an algorithm.

Byte-level BPE uses byte values as a base alphabet, which allows arbitrary UTF-8 text to be represented without an unknown-character token. That guarantee does not mean every character maps to one token. A character may occupy several UTF-8 bytes, and the learned vocabulary determines whether those bytes are combined into larger pieces.

## A small BPE training illustration

Starting with the repeated string `low low lower`, a simplified character-level trainer might first merge `l` + `o` into `lo`, then `lo` + `w` into `low`. Real tokenizers also define pre-tokenization boundaries, end-of-word behavior, byte mappings, tie-breaking, and special tokens. This small illustration explains the merge idea, not a drop-in implementation of GPT-2 or another production tokenizer.

At inference time, the stored rules are applied deterministically. If the rules contain `l + o → lo` and `lo + w → low`, the encoder uses those learned priorities rather than learning new merges from the prompt.

## Token counts are model-dependent

Tokenization is one reason the same prompt can consume different numbers of tokens in different models. A script that appears as a single character to a person may be represented by several pieces. Persian text can be affected by Arabic and Persian character variants, half-spaces, punctuation, Latin names, and digits. Two strings that look similar may have different Unicode code points, so visually inspecting a sentence is not enough to diagnose a tokenization difference.

For reproducible comparisons, record the exact model and tokenizer revision, preserve the original text, and compare the encoded IDs. Useful measurements include tokens per character or word for a fixed corpus, but compression alone is not quality. A lower token count does not prove that the model understands that language better. Evaluate task quality as well as token usage.

Normalization deserves care. Replacing Arabic `ي` with Persian `ی`, removing half-spaces, or applying Unicode normalization can be useful in a data pipeline when the task calls for it. But do not silently normalize user input differently from the model's training and serving conventions. It can change names, code, quotations, or exact-match behavior. Keep normalization policy explicit and test it on representative mixed-script examples.

## Inspect a tokenizer instead of guessing

Use the tokenizer paired with the model. This example prints pieces and IDs; it uses `transformers==5.24.1` and the `bert-base-uncased` tokenizer. The first run downloads tokenizer files from the model repository. For repeatable or offline work, pin and cache the tokenizer repository revision as well as the package version.

The example can run without PyTorch because it only uses tokenizer utilities. In a Windows console that cannot display Persian characters, run Python in UTF-8 mode with `python -X utf8`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "سلام، this is a mixed-language example."
encoded = tokenizer(text, add_special_tokens=True)

print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print(encoded["input_ids"])
print("token count:", len(encoded["input_ids"]))
```

This is a demonstration, not a recommendation to use BERT's tokenizer with a different model. The tokenizer and model vocabulary must agree. For private or offline environments, download and pin the tokenizer files through the same approved process as the model, then record their revision.

## Practical checklist

- Count tokens with the tokenizer actually used at inference, not a generic word counter.
- Include special tokens and message-format overhead when estimating a request's context use.
- Test Persian, Arabic-script variants, half-spaces, digits, URLs, code, and mixed-language prompts from your workload.
- Keep model and tokenizer versions together in deployment and evaluation records.
- Treat token efficiency as one metric; compare answer quality, latency, and cost on the same workload.

Tokenization is a small component with effects throughout data preparation, model input, and serving. Once you can inspect its output, many otherwise surprising context and cost differences become measurable.

## Further reading

- Sennrich, Haddow, and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/), introducing BPE for subword units in machine translation.
- Kudo and Richardson, [SentencePiece](https://aclanthology.org/D18-2012/), a language-independent tokenizer and detokenizer.
- Hugging Face, [The tokenization pipeline](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html) and [tokenizer components](https://huggingface.co/docs/tokenizers/python/latest/components.html).
- Petrov, La Malfa, Torr, and Bibi, [Language Model Tokenizers Introduce Unfairness Between Languages](https://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html), on cross-language tokenization disparities.
