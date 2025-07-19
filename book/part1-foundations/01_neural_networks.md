---
title: "Chapter 1: Neural Networks"
nav_order: 1
parent: "Part I: Foundations"
grand_parent: "LLMs: From Foundation to Production"
---


## 1. Neural Networks Foundations for LLMs
![image](https://github.com/user-attachments/assets/9c70c637-ffcb-4787-a20c-1ea5e4c5ba5e)
**📈 Difficulty:** Intermediate | **🎯 Prerequisites:** Calculus, linear algebra
### Key Topics

#### **What is a Neural Network?**
- Mathematical Definition: Input → Output Functions
- Neurons as Number Holders (Activations)
- Basic Network Structure: Nodes and Connections

**📚 Learning Resources:**
- [3Blue1Brown – "But what is a neural network?](https://www.youtube.com/watch?pp=0gcJCfwAo7VqN5tD&v=aircAruvnKk)
- [An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)

#### **Core Components**
- Layers: Input, Hidden, Output
- Weights and Biases
- Network Topology Basics

**📚 Learning Resources:**
- **Neural Networks: Zero-to-Hero by Andrej Karpathy** Code-along sessions building MLPs from scratch. ([YouTube Playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ))
- **MIT 6.S191 – Intro to Deep Learning** Layer types and topologies with Colab notebooks. ([YouTube](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI))

#### **Perceptrons and Multi-layer Networks (MLPs)**
- Single Perceptron Fundamentals
- Multi-layer Perceptron Design
- Feedforward Information Flow

**📚 Learning Resources:**
- **Cornell CS4780 slides – "The Perceptron Algorithm"** Derives update rule and geometric intuition. ([PDF](https://www.cs.cornell.edu/courses/cs4780/2023fa/slides/Perceptron_no_animation.pdf))
- **Pablo Insente blog** Bridges single-layer perceptron ➜ MLP with NumPy + PyTorch code. ([Posts](https://pabloinsente.github.io/the-multilayer-perceptron))
- **Technion CS236605 Posts 3** Hands-on back-prop for MLPs with PyTorch. ([Notebook](https://vistalab-technion.github.io/cs236605/tutorials/tutorial_03/))

#### **Network Structure Design**
- Layer Sizing Considerations
- Connection Patterns
- Universal Approximation Theorem

**📚 Learning Resources:**
- **Lilian Weng – "Are Deep Nets Dramatically Over-fitted?"** Width vs. depth tradeoffs under UAT. ([Lil'Log](https://lilianweng.github.io/posts/2019-03-14-overfit/))
- **Hornik et al. (1989)** Original Universal Approximation Theorem paper. ([arXiv](https://arxiv.org/pdf/2101.09181))

#### **Activation Functions**
- Sigmoid, Tanh, ReLU Functions
- Modern Activations: GELU, Swish
- Function Properties and Use Cases

**📚 Learning Resources:**
- **GELU Paper** Hendrycks & Gimpel 2016 - default in Transformers/LLMs. ([arXiv](https://arxiv.org/abs/1606.08415))
- **Swish Paper** Ramachandran et al. 2017 - discovered by NAS. ([arXiv](https://arxiv.org/abs/1710.05941))
- **Distill.pub Feature Visualization** Visual intuition for activation responses. ([Distill](https://distill.pub/2017/feature-visualization))

#### **Forward and Backward Propagation**
- Forward Pass Computation
- Gradient Computation and Chain Rule
- Backpropagation Algorithm Implementation
- Automatic Differentiation

**📚 Learning Resources:**
- **3Blue1Brown – "Back-propagation, Intuitively"** Gradient flow without messy indices. ([YouTube](https://www.youtube.com/watch?pp=0gcJCfwAo7VqN5tD&v=Ilg3gGewQ5U))
- **Colah's blog** Breaks chain-rule math into tiny graph transforms. ([Posts](https://colah.github.io/posts/2015-08-Backprop/))
- **PyTorch AutoGrad Guide** Practical autodiff for modern frameworks. ([Docs](https://pytorch.org/docs/stable/autograd.html))

#### **Loss Functions**
- Mean Squared Error (MSE)
- Cross-entropy Loss
- Huber Loss and Other Variants

**📚 Learning Resources:**
- **PyTorch Loss Functions Documentation:**
  - [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
  - [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
  - [HuberLoss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)

#### **Training Mechanics**
- Training vs Validation Performance
- Model Evaluation and Performance Metrics
- Basic Optimization with Gradient Descent

**📚 Learning Resources:**
- **Keras EarlyStopping** Canonical pattern for validation monitoring. ([Docs](https://keras.io/api/callbacks/early_stopping/))
- **Karpathy – "A Recipe for Training Neural Networks"** Battle-tested checklist for training. ([Posts](https://karpathy.github.io/2019/04/25/recipe/))

#### **Optimization Algorithms**
- Stochastic Gradient Descent (SGD)
- Adam, AdamW, RMSprop Optimizers
- Learning Rate Scheduling
- Gradient Clipping and Normalization

**📚 Learning Resources:**
- **Ruder 2016 – "Overview of Gradient Descent Optimization"** Comprehensive survey of SGD variants. ([Posts](https://www.ruder.io/optimizing-gradient-descent/))
- **Adam Paper** Kingma & Ba 2014 original ICLR paper. ([arXiv](https://arxiv.org/abs/1412.6980))
- **AdamW Paper** Loshchilov & Hutter 2017 - decoupled weight decay. ([arXiv](https://arxiv.org/abs/1711.05101))
- **PyTorch Scheduler Docs** StepLR, CosineAnnealingLR implementations. ([Docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html))
- **Gradient Clipping** Handles exploding gradients. ([PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html))

#### **Regularization Strategies**
- Understanding Overfitting: Training vs Unseen Data Performance
- L1/L2 Regularization Techniques
- Dropout and Batch Normalization
- Early Stopping and Validation Strategies
- Data Augmentation Techniques

**📚 Learning Resources:**
- **Dropout Paper** Srivastava et al. 2014 original technique. ([arXiv](https://arxiv.org/abs/1207.0580))
- **Batch Normalization** Ioffe & Szegedy 2015 - crucial for modern training. ([arXiv](https://arxiv.org/abs/1502.03167))
- **TensorFlow Data Augmentation Posts** On-the-fly transforms for images. ([TensorFlow](https://www.tensorflow.org/tutorials/images/data_augmentation))
- **Torchvision v2 Transforms** Unified API for multi-modal augmentation. ([PyTorch](https://docs.pytorch.org/vision/main/transforms.html))


---


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