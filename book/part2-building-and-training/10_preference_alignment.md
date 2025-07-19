---
title: "Preference Alignment"
nav_order: 10
parent: "Part II: Building & Training Models"
grand_parent: "LLMs: From Foundation to Production"
description: "An overview of preference alignment techniques used to make LLMs safer, more helpful, and more aligned with human values, covering RLHF, PPO, and DPO."
keywords: "Preference Alignment, RLHF, Reinforcement Learning, PPO, DPO, KTO, Reward Model, Constitutional AI"
---

# 10. Preference Alignment
{: .no_toc }

**Difficulty:** Expert | **Prerequisites:** Reinforcement Learning Basics
{: .fs-6 .fw-300 }

An SFT model knows how to follow instructions, but it doesn't have a sense of what makes a *good* response. This chapter covers preference alignment, the set of techniques used to fine-tune a model to better match human preferences, making it more helpful, harmless, and reliable.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Reinforcement Learning from Human Feedback (RLHF)</h4>
    <p>The classic RLHF pipeline: train a reward model on human preference data, then use that reward model to fine-tune the LLM with a reinforcement learning algorithm like PPO.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Reward Modeling</h4>
    <p>The process of training a model to predict which of two responses a human would prefer, which serves as the reward function for the RL algorithm.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Proximal Policy Optimization (PPO)</h4>
    <p>The reinforcement learning algorithm most commonly used in RLHF to update the LLM's policy (its weights) to maximize the score from the reward model.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Direct Preference Optimization (DPO)</h4>
    <p>A newer, simpler, and more stable method that achieves the goal of RLHF without explicitly training a reward model or using a complex RL algorithm.</p>
  </div>
  <div class="concept-grid-item">
    <h4>DPO Variants (KTO, IPO)</h4>
    <p>Recent variations on DPO that modify the objective to improve performance or handle different types of preference data.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Constitutional AI</h4>
    <p>A technique pioneered by Anthropic where the human feedback is replaced or supplemented by AI feedback, guided by a set of principles or a "constitution."</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **Reward Model Training**: Train a reward model on a public preference dataset (like Anthropic's H&H) to predict which response is more helpful or harmless.
2.  **Fine-tuning with DPO**: Use the TRL library to fine-tune a model using DPO on a preference dataset, and compare the results to the original SFT model.
3.  **RLHF with PPO**: (Advanced) Implement the full RLHF pipeline using a pre-trained reward model and PPO to align a model.

---

## 🧠 Further Reading

- **[Ouyang et al. (2022), "Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155)**: The original InstructGPT paper from OpenAI that popularized the RLHF process.
- **[Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290)**: The paper that introduced DPO.
- **[The TRL Library](https://huggingface.co/docs/trl/index)**: The primary Hugging Face library for implementing SFT, RLHF, and DPO. 