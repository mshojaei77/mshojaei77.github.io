---
layout: default
title: Supervised Fine-Tuning (SFT)
parent: Course
nav_order: 9
---

# Supervised Fine-Tuning (SFT)

**📈 Difficulty:** Advanced | **🎯 Prerequisites:** Pre-training basics

## Key Topics
- **Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters)**
  - Low-Rank Adaptation (LoRA) Theory
  - Quantized LoRA (QLoRA) Implementation
  - Adapter Layers and Bottleneck Architectures
  - PEFT vs Full Fine-Tuning Trade-offs
- **Full Fine-Tuning vs PEFT Trade-offs**
  - Memory and Compute Requirements
  - Performance Comparisons
  - Use Case Selection Criteria
- **Instruction Tuning and Chat Model Training**
  - Instruction Following Capabilities
  - Chat Template Integration
  - Response Quality Optimization
- **Domain Adaptation and Continual Learning**
  - Domain-Specific Fine-Tuning
  - Catastrophic Forgetting Mitigation
  - Continual Learning Strategies
- **Model Merging and Composition**
  - SLERP, TIES-Merging, DARE
  - Multi-Task Model Creation
  - Capability Preservation

## Skills & Tools
- **Libraries:** PEFT, Hugging Face Transformers, Unsloth, Axolotl
- **Concepts:** LoRA, QLoRA, Model Merging, Domain Adaptation
- **Tools:** DeepSpeed, FSDP, Gradient checkpointing
- **Modern Techniques:** QLoRA, DoRA, AdaLoRA, IA3

## 🔬 Hands-On Labs

**1. Parameter-Efficient Fine-Tuning with LoRA and QLoRA**
Implement comprehensive parameter-efficient fine-tuning using LoRA and QLoRA techniques. Fine-tune models like CodeLlama for code generation tasks, focusing on resource optimization and performance retention. Compare different PEFT methods and optimize for consumer GPU constraints.

**2. Domain-Specific Model Specialization**
Create specialized models for specific domains through targeted fine-tuning strategies. Implement instruction tuning to improve model following capabilities and handle catastrophic forgetting in continual learning scenarios. Optimize hyperparameters for different model sizes and tasks.

**3. Advanced Model Merging and Composition**
Fine-tune separate models for different tasks and combine them using advanced merging techniques (SLERP, TIES-Merging, DARE). Create multi-task models that maintain capabilities across different domains. Implement evaluation frameworks for merged model performance.

**4. Memory-Efficient Fine-Tuning for Limited Hardware**
Develop memory-efficient training pipelines that enable fine-tuning large models on consumer GPUs. Implement 4-bit quantization, gradient checkpointing, and other optimization techniques. Create comprehensive analysis of memory usage and training efficiency. 