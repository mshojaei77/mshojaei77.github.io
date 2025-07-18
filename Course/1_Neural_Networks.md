---
title: "Neural Networks"
nav_order: 1
parent: Course
layout: default
---


## 1. Neural Networks Foundations for LLMs
![image](https://github.com/user-attachments/assets/9c70c637-ffcb-4787-a20c-1ea5e4c5ba5e)
**📈 Difficulty:** Intermediate | **🎯 Prerequisites:** Calculus, linear algebra

### Key Topics
- **What is a Neural Network?**
  - Mathematical Definition: Input → Output Functions
  - Neurons as Number Holders (Activations)
  - Basic Network Structure: Nodes and Connections
- **Core Components**
  - Layers: Input, Hidden, Output
  - Weights and Biases
  - Network Topology Basics
- **Perceptrons and Multi-layer Networks (MLPs)**
  - Single Perceptron Fundamentals
  - Multi-layer Perceptron Design
  - Feedforward Information Flow
- **Network Structure Design**
  - Layer Sizing Considerations
  - Connection Patterns
  - Universal Approximation Theorem
- **Activation Functions**
  - Sigmoid, Tanh, ReLU Functions
  - Modern Activations: GELU, Swish
  - Function Properties and Use Cases
- **Forward and Backward Propagation**
  - Forward Pass Computation
  - Gradient Computation and Chain Rule
  - Backpropagation Algorithm Implementation
  - Automatic Differentiation
- **Loss Functions**
  - Mean Squared Error (MSE)
  - Cross-entropy Loss
  - Huber Loss and Other Variants
- **Training Mechanics**
  - Training vs Validation Performance
  - Model Evaluation and Performance Metrics
  - Basic Optimization with Gradient Descent
- **Optimization Algorithms**
  - Stochastic Gradient Descent (SGD)
  - Adam, AdamW, RMSprop Optimizers
  - Learning Rate Scheduling
  - Gradient Clipping and Normalization
- **Regularization Strategies**
  - Understanding Overfitting: Training vs Unseen Data Performance
  - L1/L2 Regularization Techniques
  - Dropout and Batch Normalization
  - Early Stopping and Validation Strategies
  - Data Augmentation Techniques

### Skills & Tools
- **Frameworks:** PyTorch, JAX, TensorFlow
- **Concepts:** Automatic Differentiation, Mixed Precision (FP16/BF16), Gradient Clipping
- **Tools:** Weights & Biases, Optuna, Ray Tune
- **Modern Techniques:** Mixed Precision Training, Gradient Accumulation
### 🔬 Hands-On Labs

**1. Perceptron From Scratch — Iris Flower Classifier**
Build a single-layer perceptron in pure NumPy to separate *Iris setosa* from the other species. Implement the perceptron learning rule, normalise features, and plot the decision boundary as training progresses. *Outcome*: students grasp linear separability, weight-update intuition, and how preprocessing affects convergence.

**2. Tiny Autograd & Activation Playground**
Fork Karpathy’s **micrograd** (≈100 LOC) and extend it to support ReLU, GELU, and Swish, then train a two-layer net on XOR and the 8×8 mini-MNIST digits from `sklearn`. Key tasks: write forward & backward functions, validate gradients with `torch.autograd.gradcheck`, and graph loss curves for each activation. *Outcome*: deep understanding of the chain rule, automatic differentiation, and the practical impact of activation choice.

**3. Regularisation Lab — Taming Overfitting on Fashion-MNIST**
Port your Lab 2 network to PyTorch, then add L2 weight-decay, dropout, and batch-normalisation. Log training/validation metrics to **Weights & Biases**, enable early stopping on a held-out set, and compare runs side-by-side. *Outcome*: students see how each regulariser shifts loss curves and can articulate why certain combinations generalise better.

**4. Optimiser & LR-Schedule Showdown**
Train a three-layer MLP on the 10 k-image CIFAR-10-tiny subset. Benchmark SGD + momentum, Adam, and AdamW while sweeping learning-rate schedules (step decay vs. cosine annealing) with **Optuna**; keep `‖g‖₂ ≤ 1` via gradient clipping. *Outcome*: a notebook (or Colab) and W\&B dashboard demonstrating which optimiser/schedule pair achieves the highest validation accuracy and a reflection on how LR tuning and clipping stabilise training.