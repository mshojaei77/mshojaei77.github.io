---
title: "Traditional Language Models: N-Grams, Smoothing, and RNNs"
layout: article
lang: en
permalink: /writing/traditional-language-models/
nav_exclude: true
description: "Trace language modeling before Transformers, from n-gram probabilities and smoothing to neural, recurrent, and sequence-to-sequence models."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Traditional Language Models Before Transformers

A language model estimates how likely a sequence of tokens is, or predicts what may come next. Modern LLMs are one branch of this history. Earlier statistical and neural models make the core problem easier to see: use context to assign probabilities, then choose or score a continuation.

## N-gram models

An n-gram model approximates the next-word probability using only the previous `n - 1` words. A bigram model uses one previous word; a trigram uses two. This is a Markov assumption: the model treats a fixed-length suffix as sufficient context for its prediction.

For a bigram, the unsmoothed maximum-likelihood estimate is:

```text
p(next | previous) = count(previous, next) / count(previous)
```

For the tiny corpus `the cat sat` and `the cat ate fish`, after `the` the next word is `cat` both times, so the estimate assigns probability 1 to `cat`. But if a pair was never observed, its estimated probability is zero. A model that assigns zero probability to any word in a test sentence assigns zero probability to the whole sentence as well.

## Smoothing: a small example

Add-one (Laplace) smoothing adds one to every possible next-token count. It is useful for demonstrating the zero-count problem, although it is usually too crude for a strong modern n-gram model. This Python example uses only the standard library:

```python
from collections import Counter

sentences = [
    ["<s>", "the", "cat", "sat", "</s>"],
    ["<s>", "the", "cat", "ate", "fish", "</s>"],
]
bigrams = Counter(
    (left, right)
    for sentence in sentences
    for left, right in zip(sentence, sentence[1:])
)
vocabulary = sorted({token for sentence in sentences for token in sentence} - {"<s>"})
context = "the"
denominator = sum(bigrams[(context, word)] for word in vocabulary) + len(vocabulary)

for candidate in ("cat", "fish"):
    probability = (bigrams[(context, candidate)] + 1) / denominator
    print(candidate, round(probability, 3))
```

With this toy corpus the output is `cat 0.375` and `fish 0.125`. The counts are not a benchmark; they just show how smoothing gives an unseen continuation nonzero probability. More effective n-gram systems use methods such as interpolation and Kneser-Ney smoothing.

## Neural language models

Neural language models replaced exact word-count tables with learned parameters. A feed-forward model can look up vectors for a fixed context window, combine them through a neural network, and predict a distribution over the vocabulary. Bengio and colleagues' 2003 model used learned word representations and a neural network to generalize across contexts that did not have identical observed n-gram counts.

Recurrent neural networks (RNNs) update a hidden state as they read each token. In principle that state can carry information from earlier in the sequence, instead of limiting the model to a fixed n-gram window. In practice, learning long-range dependencies with ordinary recurrent backpropagation is difficult. LSTMs introduced gates and a memory path intended to help preserve information over longer intervals.

Encoder-decoder recurrent models later handled sequence-to-sequence tasks such as translation: an encoder processes the input sequence, and a decoder generates an output sequence. Attention improved this pattern by allowing the decoder to consult different encoded positions while generating. The Transformer replaced recurrence in its original design with attention and feed-forward layers.

## What these models teach us

N-grams make conditional probability, context, and smoothing concrete. Neural and recurrent language models show how distributed representations can generalize beyond exact count matches and carry information across sequence positions. These ideas remain useful when reading modern LLM work, even though current generative systems usually rely on Transformer architectures and much larger learned representations.

## Further reading

- Jurafsky and Martin, [Speech and Language Processing, Chapter 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/), August 2026 draft, including smoothing and perplexity.
- Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html).
- Hochreiter and Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- For the modern generative objective, continue to [the LLM introduction](/writing/intro-to-llms/).
