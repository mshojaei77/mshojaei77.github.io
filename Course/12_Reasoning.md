---
layout: default
title: Reasoning
parent: Course
nav_order: 12
---

# Reasoning

**📈 Difficulty:** Intermediate | **🎯 Prerequisites:** Prompt engineering

## Key Topics
- **Reasoning Fundamentals and System 2 Thinking**
  - Deliberate vs Intuitive Reasoning
  - Multi-Step Problem Solving
  - Logical Reasoning Patterns
- **Chain-of-Thought (CoT) Supervision and Advanced Prompting**
  - CoT Prompting Techniques
  - Zero-shot and Few-shot CoT
  - Tree-of-Thoughts (ToT)
- **Reinforcement Learning for Reasoning (RL-R)**
  - Reward-based Reasoning Training
  - Trajectory-level Optimization
  - DeepSeek-R1 and o1 Methodologies
- **Process/Step-Level Reward Models (PRM, HRM, STEP-RLHF)**
  - Step-by-Step Reward Modeling
  - Process Reward Model Training
  - STEP-RLHF Implementation
- **Self-Reflection and Self-Consistency Loops**
  - Self-Critique Systems
  - Iterative Refinement
  - Confidence Scoring
- **Deliberation Budgets and Test-Time Compute Scaling**
  - Dynamic Token Allocation
  - Compute-Quality Trade-offs
  - Scaling Laws for Reasoning
- **Synthetic Reasoning Data and Bootstrapped Self-Training**
  - Synthetic Rationale Generation
  - Self-Training Pipelines
  - Quality Filtering

## Skills & Tools
- **Techniques:** CoT, Tree-of-Thoughts, ReAct, MCTS, RL-R, Self-Reflection, Bootstrapped Self-Training
- **Concepts:** System 2 Thinking, Step-Level Rewards, Deliberation Budgets, Planner-Worker Architecture
- **Frameworks:** DeepSeek-R1, OpenAI o1/o3, Gemini-2.5, Process Reward Models
- **Evaluation:** GSM8K, MATH, HumanEval, Conclusion-Based, Rationale-Based, Interactive, Mechanistic
- **Tools:** ROSCOE, RECEVAL, RICE, Verifiable Domain Graders

## 🔬 Hands-On Labs

**1. Chain-of-Thought Supervision and RL-R Training Pipeline**
Implement complete CoT supervision pipeline that teaches models to emit step-by-step rationales during fine-tuning. Build reinforcement learning for reasoning (RL-R) systems that use rewards to favor trajectories reaching correct answers. Compare supervised CoT vs RL-R approaches on mathematical and coding problems.

**2. Process-Level Reward Models and Step-RLHF**
Build step-level reward models (PRM) that score every reasoning step rather than just final answers. Implement STEP-RLHF training that guides PPO to prune faulty reasoning branches early and search deeper on promising paths. Create comprehensive evaluation frameworks for process-level reward accuracy.

**3. Self-Reflection and Deliberation Budget Systems**
Develop self-reflection systems where models judge and rewrite their own reasoning chains. Implement deliberation budget controls that allow dynamic allocation of reasoning tokens. Create test-time compute scaling experiments showing accuracy improvements with increased reasoning budgets.

**4. Synthetic Reasoning Data and Bootstrapped Self-Training**
Build synthetic reasoning data generation pipelines using stronger teacher models to create step-by-step rationales. Implement bootstrapped self-training where models iteratively improve by learning from their own high-confidence reasoning traces. Create quality filtering and confidence scoring mechanisms. 