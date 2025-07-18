---
title: "Neural Networks Fundamentals"
nav_order: 2
parent: Blog
layout: default
---

# Neural Network Fundamentals

Neural networks are the computational backbone of modern artificial intelligence, powering everything from a self-driving car's vision system to the large language models that can write poetry and code. They are inspired by the human brain's structure but are, at their core, powerful mathematical tools for finding patterns in data.

This comprehensive tutorial will guide you through the fundamental concepts of neural networks. We will start with a high-level definition, break the network down into its core components—neurons, layers, weights, and biases—and finally, explore how these components are arranged into different architectures.

## What is a Neural Network? A Mathematical View

At its most fundamental level, a **neural network** is a sophisticated, parameterised mathematical function that learns to map an input to a desired output.

Formally, we can describe a network as a function f:

**y = f_θ(x)**

Where:
*   **x** is the input vector (e.g., the pixel values of an image, or a sequence of words). It exists in a space with dimension d_in.
*   **y** is the output vector (e.g., a probability that the image is a cat, or the next word in a sentence). It exists in a space with dimension d_out.
*   **θ** represents all the learnable **parameters** of the network—its **weights and biases**.

The goal of "training" a neural network is to find the optimal set of parameters θ so that the function f_θ correctly approximates the unknown, underlying relationship between the training examples (x, y) we provide it.

## The Building Blocks: Neurons and Activations

To understand how the function f_θ works, we must look at its smallest unit: the neuron.

### Neurons as Number Holders
A **neuron** is the fundamental processing unit of a network. It can be intuitively understood as a "number holder" that performs a simple two-step computation:

1.  It receives one or more numerical inputs from the previous layer's neurons (or the raw input data).
2.  It computes a weighted sum of these inputs, adds a **bias**, and then passes this result through a non-linear **activation function**.

The final output of this computation, called the **activation**, is then passed as input to neurons in the next layer.

Mathematically, the activation a_j of a single neuron j is calculated as:

**a_j = σ(Σ w_ji * x_i + b_j)**

Where:
*   x_i are the input values to the neuron.
*   w_ji are the **weights**, which scale the importance of each input x_i.
*   b_j is the **bias**, a learnable offset that allows the neuron to shift its output.
*   σ (sigma) is the non-linear **activation function** (e.g., ReLU, Sigmoid, or Tanh), which introduces the non-linearity that allows networks to learn complex patterns.

## Assembling the Network: Layers and Connections

Individual neurons are organized into layers, and these layers are stacked to form a full network. Information flows directionally through these layers in what is called a **forward pass**.

### Basic Network Structure
A neural network's structure is an **acyclic directed graph**: nodes (neurons) are arranged in layers, and signals (activations) move strictly "forward" from one layer to the next, preventing infinite computational loops.

A typical network consists of three types of layers:

*   **Input Layer**: This is not a computational layer but a passive one that holds the raw features of your data (e.g., pixel intensities, word indices, sensor readings). Its size is determined by the number of features in your input, d_in.
*   **Hidden Layers**: These are the intermediate layers between the input and output. Each hidden layer takes the activations from the previous layer as input and extracts progressively higher-level features and representations from the data. The number and size of hidden layers (the network's "depth" and "width") largely determine the network's capacity to learn.
*   **Output Layer**: This is the final layer that transforms the final hidden representation into a task-specific output, such as class probabilities for a classification problem or a single continuous value for a regression task.

### The Mathematics of a Layer
When we move from a single neuron to a full layer, we can express the computation more efficiently using vectors and matrices. The forward pass from layer ℓ-1 to layer ℓ is calculated as:

**h^(ℓ) = σ(W^(ℓ) * h^(ℓ-1) + b^(ℓ))**

Where:
*   h^(ℓ-1) is the vector of activations from the previous layer.
*   W^(ℓ) is the **weight matrix** for layer ℓ, where each entry W_ij contains the weight connecting neuron i in layer ℓ to neuron j in layer ℓ-1.
*   b^(ℓ) is the **bias vector** for layer ℓ, containing the bias for each neuron.
*   σ(·) is the activation function, applied element-wise to the result.
*   h^(ℓ) is the resulting vector of activations for the current layer ℓ.

Chaining these blocks together, from the input layer to the output layer, forms the entire network function f_θ.

## The Learnable Parts: Weights and Biases

The parameters θ that the network learns during training are its weights and biases.

*   **Weights (W)**: These are learnable scaling factors on every connection between neurons. They determine the strength and direction of influence one neuron has on another. A high positive weight means the input is strongly excitatory, while a high negative weight means it's strongly inhibitory. In code, they often live in tensors like PyTorch's `nn.Linear.weight` (with shape *out_features × in_features*).

*   **Biases (b)**: These are learnable, per-neuron offsets that allow the activation function to be shifted left or right. This added flexibility is often crucial for the model to fit the data properly.

During training, an **optimizer** (like SGD or Adam) uses a **loss function** to measure the network's error. It then calculates the gradient of this error with respect to all weights and biases and updates them in the direction that minimizes the error. This process is known as **backpropagation** and **gradient descent**.

## Network Architecture: An Overview of Topologies

Topology describes "who talks to whom" inside the model. While the fully-connected layer described above is the default, specialized topologies encode useful assumptions (inductive biases) about the data, leading to more efficient and powerful models.

| Topology                           | Characteristic Connections                   | Typical Use Case                      |
| ---------------------------------- | -------------------------------------------- | ------------------------------------- |
| **Fully-Connected (Feed-Forward)** | Dense, no cycles; every neuron connects to every neuron in the next layer. | Tabular data, simple baselines.       |
| **Convolutional (CNN)**            | Local receptive fields and shared weights across space. | Images, audio, and other grid-like signals. |
| **Recurrent (RNN, LSTM, GRU)**     | Connections that form cycles through time, with shared weights at each step. | Sequences, time-series data, language. |
| **Residual / Skip-Connections**    | Additive identity paths that bypass one or more layers. | Very deep networks, to stabilize training and gradients. |

Choosing the right topology is a key part of model design and directly affects the parameter count, computational cost, and overall performance of the network.

## Where to Go Next

You now have a solid foundation in what neural networks are and how they work. The key takeaways are:
*   A network is a **parameterized function** composed of **layers**.
*   Layers are made of **neurons**, which compute a **weighted sum + bias** followed by an **activation function**.
*   **Weights and biases** are learned from data to minimize a **loss function**.
*   The network's **topology** encodes assumptions about the data.

Once you are comfortable with these building blocks, the natural next step is to dive into the mechanics of training, including loss functions, backpropagation, optimizers, and regularization.

For a practical, hands-on next step, we highly recommend Andrej Karpathy's classic blog post, **[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)**, which provides an invaluable practitioner's guide.

### References and Further Reading
[1] Schmidhuber, J. (2015). *Deep Learning in Neural Networks: An Overview*. [arXiv:1404.7828](https://arxiv.org/abs/1404.7828)
[2] Stanford CS231n Course Notes. *Lecture 4: Neural Networks and Backpropagation*. [[PDF]](https://cs231n.stanford.edu/slides/2024/lecture_4.pdf)
[3] Karpathy, A. (2019). *A Recipe for Training Neural Networks*. [Andrej Karpathy Blog](https://karpathy.github.io/2019/04/25/recipe/)
[4] Zhou, V. *A Beginner's Introduction to Neural Networks*. [Victor Zhou's Blog](https://victorzhou.com/blog/intro-to-neural-networks/)
[5] Lado, J., et al. (2020). *Introduction to deep learning*. [arXiv:2003.03253](https://arxiv.org/abs/2003.03253)
[6] PyTorch Documentation. *torch.nn.Linear*. [[Docs]](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)