---
title: "Neural Networks"
nav_order: 2
parent: Blog
layout: default
---



# Neural Networks — From First Principles to Front-Line Research

*A friendly but thorough expedition across the core concepts that power today’s AI, from the earliest perceptrons to Pointer Networks and beyond.*

---

## 1 Why Learn This Stuff?

Deep-learning libraries hide most of the mathematics behind a single `model.fit()` call, but the *ideas* underneath still matter. They tell you:

* **Which model to reach for** (CNN, RNN, Transformer?) and **why**.
* **What to try first** when training stalls (raise the learning-rate warm-up, add weight-decay, clip gradients, …).
* **How to spot red-flags** (exploding activations, vanishing gradients, over-fitting) *before* your GPU budget vanishes.

This guide starts at rock-bottom—individual neurons—and builds, layer by layer, toward the high-level architectures that run chatbots, recommend videos, and solve combinatorial puzzles.

Grab a mug of coffee; we’re going deep.

---

## 2 Neural-Network Fundamentals & Architecture Design

### 2.1 The Computational Graph

A feed-forward network is nothing more than a big composite function

$$
f_\theta(\mathbf{x}) \;=\; \phi_L\!\bigl(W_L\,\phi_{L-1}\bigl(\dotsb\,\phi_1(W_1\mathbf{x}+b_1)\bigr)+b_L\bigr),
$$

where

* **$W$** and **$b$** are learnable weights and biases,
* **$\phi$** is an *activation* (Section 3), and
* **$L$** is the number of hidden layers.

Because every operation is differentiable, we can compute exact gradients of a loss $\mathcal{L}$ with respect to *all* parameters.

### 2.2 Depth, Width, and Parameter Count

* **Depth** (more layers) lets the network build hierarchical features—edges → corners → objects.
* **Width** (more neurons per layer) increases capacity at each level.
* **Parameter sharing** is the trick behind convolutions and recurrent layers: one weight matrix is reused across space or time, cutting memory costs.

Practical tips:

| Concern                                        | Rule of thumb                                                     |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| Training loss stuck?                           | Make it **wider** or **deeper** *and* raise learning rate.        |
| Validation loss rising but train loss falling? | Add regularization (dropout, weight-decay) or gather more data.   |
| GPU out of memory?                             | Use smaller batch-size, 8-bit weights, or gradient checkpointing. |

---

## 3 Activation Functions – Injecting Non-Linearity

| Function             | Formula                                                       | Pros                                 | Cons                                | Use-cases                     |
| -------------------- | ------------------------------------------------------------- | ------------------------------------ | ----------------------------------- | ----------------------------- |
| Sigmoid              | $\sigma(x)=1/(1+e^{-x})$                                      | Probabilities; smooth                | Saturates; not zero-centred         | Output layer for binary tasks |
| Tanh                 | $\tanh(x)$                                                    | Zero-centred                         | Still saturates                     | Legacy RNNs                   |
| **ReLU**             | $\max(0,x)$                                                   | Cheap; mitigates vanishing gradients | “Dead” neurons when $x<0$ too often | All modern CNN/RNN FFN blocks |
| Leaky-ReLU, PReLU    | Small slope left of zero                                      | Fix dead neurons                     | Extra hyper-parameter               | GANs                          |
| **GELU** (GPT-style) | $0.5x\left[1+\operatorname{erf}\!\bigl(x/\sqrt2\bigr)\right]$ | Smooth, stochastic                   | Slightly slower                     | Transformers                  |

*Guiding principle:* prefer ReLU variants unless your paper reviewer demands something exotic.

---

## 4 Gradients & Back-Propagation

### 4.1 The Chain Rule Engine

Back-propagation *is* the chain rule, applied from the output back to every parameter:

$$
\frac{\partial \mathcal{L}}{\partial W_l} \;=\;
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}\,
\frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}}\,
\dotsb
\frac{\partial \mathbf{h}_{l+1}}{\partial W_l}.
$$

Modern frameworks (PyTorch, TensorFlow, JAX) build the graph as you write Python, then run automatic differentiation (`.backward()`).

### 4.2 Vanishing & Exploding Gradients

Repeated multiplications of Jacobians can shrink (eigenvalues < 1) or blow up ( > 1) the gradient.

Fixes:

* **Good initialization**: Xavier (fan-avg) for tanh/sigmoid, He (fan-in) for ReLU.
* **Normalization layers**: Batch-Norm, Layer-Norm, RMS-Norm.
* **Skip connections**: Residual blocks → gradient highways.

---

## 5 Loss Functions & Regularization Strategies

### 5.1 Popular Losses

* **Cross-Entropy**: classification and language-model token prediction.
* **MSE** (ℓ2): regression.
* **Huber**: robust regression (quadratic near 0, linear for outliers).
* **CTC**: sequence alignment without explicit segmentation (speech).

### 5.2 Keeping Generalization on Track

| Technique             | One-liner                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| **L2 weight-decay**   | Adds $\lambda\|\theta\|_2^2$ to the loss; combats large weights.                                    |
| **Dropout**           | Randomly zeroes activations during train; set `p=0` at eval time.                                   |
| **Early stopping**    | Halt when validation loss stops improving.                                                          |
| **Label smoothing**   | Replaces one-hot targets with 0.9 / 0.1 distribution; stabilizes training of very wide classifiers. |
| **Data augmentation** | MixUp, random crop, token masking, back-translation, …                                              |

---

## 6 Optimization Algorithms

### 6.1 Classics

* **Batch Gradient Descent**: conceptually pure, practically slow.
* **SGD**: updates on mini-batches; add **momentum** to accelerate through ravines.
* **Nesterov Momentum**: “look-ahead” gradient for extra stability.

### 6.2 Adaptive Methods

| Optimizer   | Secret sauce                                               | Typical defaults     |
| ----------- | ---------------------------------------------------------- | -------------------- |
| **Adam**    | Exponential moving averages of grad (m) & squared grad (v) | `β1=0.9, β2=0.999`   |
| **AdamW**   | Decouples weight-decay from Adam update                    | Use for Transformers |
| **RMSProp** | Only tracks second moment                                  | Legacy RNN code      |

Cutting-edge:

* **Lion**: updates by sign of EMA of gradients → lower memory.
* **Sophia-G**: diagonal Hessian approximation for gigantic LLMs.

### 6.3 Learning-Rate Tricks

1. **Warm-up**: linear ramp over first \~ 1000 steps.
2. **Cosine decay** or **linear decay**: anneal down to a floor.
3. **Cyclic schedulers**: periodically reset LR to escape local minima.

---

## 7 Recurrent Neural Networks (RNNs)

An RNN reuses the same cell at every time-step:

$$
\mathbf{h}_t=\phi\!\bigl(W_{xh}\mathbf{x}_t + W_{hh}\mathbf{h}_{t-1}+b_h\bigr).
$$

This parameter sharing yields *O(hidden²)* weights regardless of sequence length.

### 7.1 Back-Propagation Through Time (BPTT)

1. **Unroll** the graph for $T$ steps.
2. Run standard back-prop along the unrolled chain.
3. Optionally **truncate** to a sliding window (TBPTT) to save memory.

### 7.2 The Curse of Long-Term Dependencies

With plain tanh cells, gradients fade after \~20 – 30 tokens. Two celebrated fixes:

* **Gradient clipping**: scale back ‖g‖ if it exceeds a threshold (e.g., 1.0).
* **Gated cells**: LSTM and GRU.

---

## 8 Long Short-Term Memory (LSTM)

![lstm diagram](https://i.imgur.com/8e1f2Yf.png)

*(If image fails to load: think “conveyor-belt plus three gates”.)*

Equations:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(W_{xi}\mathbf{x}_t+W_{hi}\mathbf{h}_{t-1}+b_i) \\
\mathbf{f}_t &= \sigma(W_{xf}\mathbf{x}_t+W_{hf}\mathbf{h}_{t-1}+b_f) \\
\mathbf{o}_t &= \sigma(W_{xo}\mathbf{x}_t+W_{ho}\mathbf{h}_{t-1}+b_o) \\
\tilde{\mathbf{c}}_t &= \tanh(W_{xc}\mathbf{x}_t+W_{hc}\mathbf{h}_{t-1}+b_c) \\
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t).
\end{aligned}
$$

The **cell state** $\mathbf{c}_t$ carries information unimpeded; gates learn which bits to erase, write, or expose.

Variants: **GRU** (two gates), **peephole LSTM**, **Bidirectional LSTM** (look-ahead context).

---

## 9 Attention — Letting Information Flow Freely

### 9.1 Encoder–Decoder Attention

Instead of forcing a single fixed vector bottleneck, the decoder can *query* all encoder states:

$$
\text{Attention}(\mathbf{q},K,V)=\operatorname{softmax}\!\left(\frac{\mathbf{q}K^\top}{\sqrt{d_k}}\right)V.
$$

Here **Q** is the decoder’s current hidden state, **K**/**V** are encoder outputs. The softmax weights sum to 1, forming a context vector that adapts at every output step.

### 9.2 Self-Attention & Transformers

*Apply the same trick *within* the sequence.* Each token attends to every other token in parallel. Add **positional encodings** so order isn’t lost, stack many layers, and you have the Transformer—state-of-the-art for language, vision, protein folding, you name it.

Key insights:

* **Multi-Head**: learn $h$ attention patterns in parallel, then concatenate.
* **Feed-Forward Network**: two 1 × 1 convolutions (linear layers) applied token-wise; most parameters live here.
* **Layer-Norm + Residuals**: stabilize gradients in 100-layer stacks.

---

## 10 Pointer Networks — When the Output *Is* the Input

In tasks like sorting, TSP, or text span extraction, the answer isn’t chosen from a fixed vocabulary but from *positions* in the input.

Pointer Networks reuse attention *weights* directly as a probability distribution over input indices:

1. Encode the sequence with an RNN or Transformer.
2. At each decoder step, compute attention scores $\alpha$.
3. **Softmax** the scores; that distribution *is* the output token.

Because the output length can differ from the input, and because the “vocabulary” scales with sequence length, Ptr-Nets fit problems that were awkward for vanilla seq-to-seq.

---

## 11 A Quick Glance at Front-Line Research

| Theme                                        | What it does                                                                                  | One paper to Google    |
| -------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------- |
| **Mixture-of-Experts (MoE)**                 | Activates only a tiny subset of very large model for each token → trillion-scale sparse LLMs. | “Switch Transformers”  |
| **Retrieval-Augmented Generation (RAG)**     | Pulls fresh facts from an external database before answering.                                 | “REALM”                |
| **Efficient Attention**                      | Low-rank or local windows to make O(n) memory instead of O(n²).                               | “Flash-Attention”      |
| **Sign-based optimizers (Lion)**             | Cut optimizer state in half, still converge.                                                  | “Lion: Symbolic SGD”   |
| **Continual / Domain-adaptive pre-training** | Keep training the foundation model on more focused data.                                      | “BioGPT”, “Code Llama” |

---

## 12 Putting It All Together — A Minimal Recipe

```python
model = TransformerLM(
    vocab_size=50_000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    dropout=0.1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = cosine_with_warmup(optimizer, warmup_steps=1_000, total_steps=100_000)

for batch in dataloader:
    logits = model(batch["input"])
    loss = F.cross_entropy(logits.flatten(0,1), batch["target"].flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); optimizer.zero_grad(); scheduler.step()
```

Everything you’ve learned surfaces here:

* **Embedding + Positional encodings** in the model.
* **Cross-entropy** loss.
* **AdamW** with **weight-decay**, **warm-up**, **cosine decay**.
* **Gradient clipping** for stability.

Swap the Transformer for an LSTM, change the loss to CTC, or replace the trainer with Lion—the principles remain.

---

## 13 Where to Go Next

1. **Re-implement** a tiny LSTM language model from scratch; handwritten grads build intuition.
2. **Dive into “Attention Is All You Need”**; re-derive scaled dot-product attention on paper.
3. **Tackle a toy combinatorial task** (e.g., sorting numbers) with a Pointer Network.
4. Join an open-source LLM project; read the training scripts and *match each line* to concepts in this tutorial.

Armed with this knowledge, the imposing walls of deep-learning research start to look like a well-charted maze: twisty, yes, but navigable. Happy exploring—and may your gradients be ever in your favor!
