---
title: "How Transformers Work: Self-Attention, Masks, and Generation"
layout: article
lang: en
permalink: /writing/transformers/
nav_exclude: true
description: "A practical introduction to Transformer tensors, scaled dot-product attention, causal masking, encoder and decoder blocks, and the prefill and decode stages of LLM inference."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# How Transformers Work

Transformers turn a sequence of token IDs into vectors, repeatedly update those vectors with attention and feed-forward layers, then use the resulting states for a task such as predicting the next token. The architecture is often introduced through attention, but attention is one part of a larger block that also includes projections, residual connections, normalization, and a feed-forward network.

This tutorial follows a small decoder-only language model. The original Transformer was an encoder-decoder model for sequence transduction; encoder-only and decoder-only models use related components with different attention masks and training objectives.

## From token IDs to hidden states

Suppose a batch contains `B` sequences, each with `T` tokens. A tokenizer produces integer IDs, and an embedding table maps each ID to a vector of width `d_model`. Before the first block, the hidden state tensor therefore has shape:

```text
X: [batch, sequence, model_width] = [B, T, d_model]
```

The model also needs information about token order. Attention alone is permutation-equivariant: without position information or another ordering mechanism, changing token order does not tell the layer what came first. The original Transformer adds sinusoidal positional encodings; other architectures use learned positions or rotary position embeddings (RoPE). Position schemes are model design choices, not interchangeable tokenizer features.

## Self-attention with shapes

Each attention head projects the current hidden states into queries, keys, and values:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

For a single head of width `d_k`, their shapes are `[B, T, d_k]`. Each query compares with all keys to produce weights over positions. Those weights form a weighted sum of the value vectors:

```text
Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + M) V
```

The scale `sqrt(d_k)` keeps dot-product magnitudes from growing with the head width. The mask `M` controls which positions are visible. For a batch, the attention scores have shape `[B, heads, T, T]`; each query position has a score for each key position.

With multiple heads, each head has its own learned projections. The outputs are concatenated and projected back to the model width. This lets the layer compute several attention patterns in parallel; it does not guarantee that an individual head has a simple human-readable role.

## Causal masking and next-token prediction

A decoder-only language model predicts the next token from tokens that came before it. During training, the causal mask prevents position `i` from using positions greater than `i`. The model can still process all training positions in parallel because the full input sequence is known; the mask blocks information flow from future positions.

For a short sequence, the allowed attention pattern looks like this (`✓` means the query may read that key):

```text
          key position
query       1   2   3   4
    1       ✓   ·   ·   ·
    2       ✓   ✓   ·   ·
    3       ✓   ✓   ✓   ·
    4       ✓   ✓   ✓   ✓
```

At inference, generation is sequential: the next token depends on the tokens already generated. For encoder-decoder models, the decoder also has cross-attention that reads encoder outputs. Decoder-only models do not have that encoder/cross-attention path.

## What an attention block contains

A typical Transformer block combines attention with a position-wise feed-forward network (FFN), residual connections, and normalization. At each layer, attention moves information across sequence positions; the FFN transforms each position's features. Residual paths help preserve and update the representation across depth. Exact ordering varies: the 2017 architecture uses post-layer normalization, while many modern decoder models use pre-layer normalization or related variants.

The original paper's encoder and decoder contain different sublayers. A decoder-only LLM is not simply the complete original encoder-decoder stack with the encoder removed; it is a model family built around causal self-attention and its own normalization, position, and output conventions.

## Prefill, decode, and the KV cache

Inference commonly has two stages. During **prefill**, the model processes the prompt and computes key and value tensors for its layers. During **decode**, it generates one or a few new tokens at a time. A key-value cache stores the previous keys and values so the model can reuse them instead of recomputing those projections for every earlier token.

The cache uses memory proportional to the number of layers, cached sequence positions, and key/value width (also multiplied by batch size and numeric precision). Longer prompts and more concurrent sequences therefore increase memory use. Multi-query attention and grouped-query attention share key/value heads across query heads to reduce cache size in some architectures. These are inference-relevant design choices, but their actual latency and capacity effects depend on the model, serving engine, hardware, and workload.

Attention over a length-`T` sequence forms pairwise scores with `T × T` entries per head, so its compute and score storage grow quadratically with sequence length in the standard formulation. Optimized kernels can reduce memory traffic and improve practical performance without changing that underlying pairwise pattern. The KV cache avoids recomputing past keys and values during autoregressive decoding, but it does not make the context free: the model still attends over cached positions for each generated token.

## A useful mental model

For a decoder-only LLM, trace one token through the system:

1. The tokenizer maps text to IDs.
2. The embedding table maps each ID to a vector; position information is added or applied.
3. Each block uses masked self-attention to mix information across earlier positions, then an FFN to transform each position.
4. The final hidden state is projected to vocabulary logits; a decoding rule selects the next token.
5. The tokenizer decodes generated IDs back into text.

This view connects model architecture to practical debugging. Tokenizer differences alter the input sequence. Causal masks define what the model can condition on. The KV cache trades memory for reuse during generation. When throughput or latency is poor, measure prompt processing and token generation separately instead of treating “inference” as one operation.

## Further reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original encoder-decoder Transformer.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), introducing RoPE.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), on grouped-query attention.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), an exact attention algorithm designed to reduce memory traffic.
