---
title: "Supervised Fine-Tuning"
nav_order: 9
parent: "Part II: Building & Training Models"
grand_parent: "LLMs: From Foundation to Production"
description: "A guide to Supervised Fine-Tuning (SFT), the process of adapting a pre-trained LLM for specific tasks, with a focus on parameter-efficient methods like LoRA and QLoRA."
keywords: "Supervised Fine-Tuning, SFT, Instruction Tuning, PEFT, LoRA, QLoRA, Adapters, Model Merging"
---

# 9. Supervised Fine-Tuning
{: .no_toc }

**Difficulty:** Advanced | **Prerequisites:** Pre-training Basics
{: .fs-6 .fw-300 }

Supervised Fine-Tuning (SFT) is the primary method for adapting a base pre-trained model into a helpful assistant that can follow instructions. This chapter covers the mechanics of SFT, with a strong focus on parameter-efficient techniques that make it possible to fine-tune large models on consumer hardware.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Full Fine-Tuning vs. PEFT</h4>
    <p>The trade-offs between updating all of a model's weights versus using Parameter-Efficient Fine-Tuning (PEFT) methods.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Low-Rank Adaptation (LoRA)</h4>
    <p>A popular PEFT technique that freezes the pre-trained model weights and injects trainable low-rank matrices into the Transformer layers.</p>
  </div>
  <div class="concept-grid-item">
    <h4>QLoRA</h4>
    <p>A highly efficient variant of LoRA that combines it with 4-bit quantization and other memory-saving tricks to fine-tune massive models on a single GPU.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Instruction Tuning</h4>
    <p>The process of fine-tuning a model on a dataset of instruction-response pairs to teach it to follow user commands.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Domain Adaptation</h4>
    <p>Fine-tuning a model on a large corpus of text from a specific domain (e.g., medical, legal) to improve its expertise in that area.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Model Merging</h4>
    <p>Techniques for merging the weights of two or more fine-tuned models to create a new model that combines their capabilities.</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **Fine-tuning with LoRA**: Use the Hugging Face `peft` library to fine-tune a model like Llama 3 on an instruction dataset using LoRA.
2.  **Fine-tuning with QLoRA**: Fine-tune a large model (e.g., 7B parameters) on a consumer GPU by implementing QLoRA with 4-bit quantization.
3.  **Model Merging**: Fine-tune two models on different tasks (e.g., one on code, one on poetry) and merge them using a library like `mergekit`.

---

## 🧠 Further Reading

- **[Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)**: The original paper introducing the LoRA technique.
- **[Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)**: The paper that introduced QLoRA, making large-model fine-tuning much more accessible.
- **[Hugging Face PEFT Library](https://huggingface.co/docs/peft/index)**: The go-to library for implementing various parameter-efficient fine-tuning methods. 