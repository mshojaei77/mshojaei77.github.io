---
title: "Neural Network Architectures"
nav_order: 4
parent: "Neural Networks"
grand_parent: "Part I: Foundations"
---

# Neural Network Architectures

While all networks are built from neurons and layers, their topology—how they are connected—is specialized for different data types.

## Feedforward Networks (Fully Connected)

The simplest topology, where every neuron in one layer connects to every neuron in the next. They are general-purpose approximators, good for tabular data. This is the architecture we've primarily discussed so far in this chapter.

Key characteristics:
- Information flows in one direction only (forward)
- Each neuron in a layer connects to every neuron in the next layer
- No loops or cycles in the network
- Well-suited for classification and regression tasks with structured data

## Convolutional Neural Networks (CNNs)

The superstars of computer vision. They use special convolutional layers with shared weights to detect local features in grid-like data (like images) in a way that is translation-invariant.

Key innovations:
- **Local receptive fields**: Each neuron only looks at a small region of the input
- **Shared weights**: The same filter is applied across the entire input, drastically reducing parameters
- **Pooling layers**: Downsample the representation, making the network more robust to small translations
- **Translation invariance**: A feature can be detected regardless of where it appears in the image

CNNs have revolutionized image classification, object detection, segmentation, and many other computer vision tasks. Their hierarchical feature learning is particularly effective for visual data, where patterns exist at multiple scales and positions.

## Recurrent Neural Networks (RNNs)

Designed for sequential data like text or time series. They have connections that loop back on themselves, giving them a form of "memory" to process inputs in context.

Key features:
- **Memory state**: RNNs maintain an internal state that gets updated with each input
- **Parameter sharing across time**: The same weights are applied at each time step
- **Ability to handle variable-length sequences**: Can process sequences of different lengths
- **Challenges with long-range dependencies**: Basic RNNs struggle with long sequences due to vanishing/exploding gradients

Advanced variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) address the limitations of basic RNNs by introducing gating mechanisms that control information flow, allowing the network to learn longer-range dependencies.

## Residual Networks (ResNets)

A key innovation that uses "skip connections" to allow gradients to bypass layers. This makes it possible to train extremely deep networks without suffering from the vanishing gradient problem.

The core insight of ResNets is simple but profound: instead of trying to learn a direct mapping H(x), the network learns a residual function F(x) = H(x) - x. The original input x is then added back through a skip connection: H(x) = F(x) + x.

This seemingly small change has enormous implications:
- Gradients can flow directly through the skip connections, mitigating the vanishing gradient problem
- Networks can be much deeper (hundreds or even thousands of layers)
- Training becomes more stable and often converges faster
- Performance typically improves as depth increases (up to a point)

ResNets and their variants have become fundamental building blocks in many state-of-the-art architectures, including those used in computer vision and, more recently, in transformers for NLP.

<img width="792" height="430" alt="Four neural network architectures comparison: feedforward network with linear layer progression, CNN with convolutional and pooling layers for image processing, RNN with recurrent connections for sequential data, and ResNet block with skip connections bypassing convolutional layers" src="https://github.com/user-attachments/assets/de185435-7d48-4054-a08b-04f39ec39916" />

**Figure 4.1:** Common Neural Network Architectures. Comparison of four fundamental architectures: Feedforward (fully connected), CNN (convolutional), RNN (recurrent), and ResNet (residual), each optimized for different data types and tasks.

## Transformers

While not covered in detail in this chapter, it's worth mentioning Transformers as they represent the current state-of-the-art architecture for natural language processing and are the foundation of Large Language Models (LLMs).

Key innovations:
- **Self-attention mechanism**: Allows the model to weigh the importance of different parts of the input sequence
- **Parallelization**: Unlike RNNs, transformers process the entire sequence in parallel
- **Positional encoding**: Preserves sequence order information without recurrence
- **Scaled to enormous sizes**: Modern transformers can have billions or even trillions of parameters

Transformers will be covered in much greater detail in subsequent chapters, as they form the backbone of modern LLMs.

## Conclusion

Neural networks are not black magic; they are elegant mathematical systems built from a few core, surprisingly simple principles. This chapter has armed you with the foundational concepts:

-   Networks are built from simple **neurons** organized into **layers**.
-   **Non-linear activation functions** are the secret sauce that gives networks their power.
-   Learning is an iterative dance of **forward propagation**, **loss calculation**, **backpropagation**, and **parameter updates**.
-   **The Chain Rule** is the engine of backpropagation, distributing blame for errors.
-   **Regularization** is the discipline that prevents **overfitting** and forces generalization.
-   **Initialization** sets the stage for effective learning.
-   Different **network architectures** are specialized for different types of data and tasks.

With this foundation, you are ready to tackle the transformer architectures that power modern large language models. The dragons may still be there, but now you know their names.

---

### Exercises
-   Implement a simple feedforward network from scratch using only NumPy.
-   Compare the performance of a CNN vs. a fully connected network on an image classification task.
-   Implement a basic RNN for a sequence prediction task and observe how it handles dependencies of different lengths.

### Online Resources
-   **[Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)**: A hands-on, code-first guide to building neural networks from scratch.
-   **[PyTorch Tutorials](https://pytorch.org/tutorials/)**: Official documentation and guides for the PyTorch framework. 