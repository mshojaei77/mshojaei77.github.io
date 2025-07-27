---
title: "The Learning Process"
nav_order: 2
parent: "Neural Networks"
grand_parent: "Part I: Foundations"
---

# The Learning Process

"Training" a neural network means finding the best set of values for all its weights and biases. This isn't a one-shot deal; it's an iterative process of refinement.

## Cost Functions
The entire goal of training is to make the network as accurate as possible. We quantify its performance with a **cost function** (or **loss function**). You can think of this as a "lousiness score"—a single number that brutally measures how wrong the network's predictions are compared to the actual, correct answers. A perfect network has a loss of 0. The goal of training is to find the specific set of weights and biases that results in the lowest possible cost across the entire training dataset.

## Gradient Descent
How do we find the settings that minimize this cost? We use an algorithm called **Gradient Descent**. Imagine the cost function as a huge, hilly, multi-dimensional landscape where "altitude" is the cost and your "position" is defined by the current values of the network's thousands or millions of weights and biases.

1.  You start at a random position (randomly initialized weights and biases).
2.  You calculate the **gradient**, which is a vector that points in the direction of the *steepest ascent*—the direction where the cost increases fastest.
3.  You take a small step in the **exact opposite direction** of the gradient. This is the "downhill" direction.
4.  By repeating this process, you iteratively "roll down the hill" of the cost landscape. Eventually, you settle in a valley—a point where the cost is at a minimum.

## Training Loop
This "rolling downhill" process is organized into an iterative process called the **training loop**.

<img width="4025" height="1545" alt="Training loop flowchart showing four steps: forward propagation feeding input through network, loss calculation comparing prediction vs target, backward propagation calculating gradients, parameter update adjusting weights, with convergence check determining if process continues or completes" src="https://github.com/user-attachments/assets/18cac33e-31af-4bf2-8eda-a1ebfe54f870" />

**Figure 2.1:** Neural Network Training Loop. The iterative four-step process of training: (1) Forward propagation, (2) Loss calculation, (3) Backward propagation (backpropagation), and (4) Parameter update, repeated until convergence.


1.  **Forward Propagation**: We feed input data into the network. It flows through the layers, each performing its calculation, until the final layer spits out a prediction.
2.  **Loss Calculation**: We compare the network's prediction to the actual, true target using our loss function.
3.  **Backward Propagation (Backpropagation)**: This is the computational heart of learning. Here, we calculate the gradient of the loss with respect to every single weight and bias in the network.
4.  **Parameter Update**: The optimizer uses the calculated gradients to update all the weights and biases, nudging them in the downhill direction.

This four-step dance is repeated thousands or millions of times until the network's predictions are consistently accurate.

## Backpropagation
Backpropagation is the algorithm that makes gradient descent feasible by efficiently computing the gradients for all parameters. At its core, it relies on two key ideas: **derivatives as measures of influence** and the **chain rule**.

A **derivative** measures the **sensitivity** of a function's output to a tiny change in one of its inputs. For a neural network, the derivative of the final loss with respect to a single weight tells us how "influential" that weight is. A positive gradient means increasing the weight increases the loss (bad), while a negative gradient means increasing the weight decreases the loss (good).

To calculate these derivatives for millions of parameters, backpropagation uses the **chain rule**. It's like a gossip chain in reverse. It starts at the very end (the loss) and propagates the error signal *backward* through the network, layer by layer. At each step, it uses the chain rule to determine how much each neuron, weight, and bias contributed to the overall error. This provides the exact "nudge" required for every parameter to move the cost downhill.

For incredibly deep models like LLMs, this backward pass of distributed blame is the only feasible way to train them.

A critical detail in practice is that gradients from different paths in the network must be **summed up** for any given parameter. If you don't reset the stored gradients to zero before each backward pass (`zero_grad()`), you'll be accumulating gradients from previous training batches, which corrupts the optimization step and derails learning.

## Computation Graph
To truly understand how backpropagation is automated, it's helpful to visualize the entire process as a **computation graph**. This is a Directed Acyclic Graph (DAG) where each node represents a value (a scalar, vector, or tensor) and each edge represents an operation (e.g., `+`, `*`, `tanh`).

The **forward pass** builds this graph. As you execute your code, every operation and its resulting value are recorded. For example, the expression `L = f(d * w + b)` creates a small graph showing how the inputs `d`, `w`, and `b` combine to produce the final loss `L`.

The **backward pass** is then simply the process of traversing this graph *backwards* from the final output node (the loss). At each node, it calculates the local gradients (how the output of an operation changes with respect to its inputs) and uses the chain rule to pass the "upstream" gradient back to the nodes that fed into it.

This graph structure is the fundamental principle behind modern autograd engines like those in PyTorch and TensorFlow. While this chapter discusses scalar values for simplicity, these professional libraries perform the exact same process on **tensors** (multi-dimensional arrays), allowing them to compute gradients for millions of parameters with astonishing efficiency, especially on GPUs.

## Stochastic Gradient Descent
Calculating the gradient based on the *entire* training dataset for every single step (known as **Batch Gradient Descent**) is accurate but computationally prohibitive for large datasets.

Instead, we use **Stochastic Gradient Descent (SGD)**. SGD dramatically speeds up training by estimating the gradient based on just a small, random subset of the data called a **mini-batch** (e.g., 100 images instead of 50,000). It's like a "drunk man stumbling downhill"—less direct and more wobbly than the batch method, but it moves much faster and is often more effective at escaping shallow valleys in the cost landscape.

## Loss Functions

The choice of loss function is tailored to the task:

*   **Mean Squared Error (MSE)**: The go-to for regression tasks where the output is a continuous value (e.g., predicting the price of a vintage computer).
*   **Cross-Entropy Loss**: The standard for classification. It measures the dissimilarity between the predicted probabilities and the true, one-hot encoded labels.

## Optimization Algorithms
The optimizer is the chauffeur for our learning process. It uses the gradients to decide how to update the parameters. While SGD defines the strategy (using mini-batches), specific algorithms improve upon it.

*   **Adam (Adaptive Moment Estimation)**: The sophisticated, self-correcting GPS. It's the default choice for most deep learning tasks. Adam adapts the learning rate for each parameter individually and uses momentum (an accumulation of past gradients) to accelerate the journey.
*   **AdamW**: An improved version of Adam that decouples weight decay (a regularization technique) from the main optimization step, which often leads to better generalization, especially in transformers.

The **learning rate** is a critical hyperparameter that controls how big of a step the optimizer takes. Too large, and it might leap right over the optimal solution. Too small, and training will take an eternity. **Learning rate scheduling**, which adjusts the learning rate during training, is a key technique for peak performance.

<img width="3897" height="567" alt="Optimization process diagram showing current weights flowing to gradient calculation, then to optimizer (SGD/Adam/AdamW) with learning rate input, then to weight update using formula w = w - η∇L(w), resulting in updated weights" src="https://github.com/user-attachments/assets/e8d188bb-f0cf-422c-ba96-2f8592986e28" />

**Figure 2.2:** Optimization Process in Neural Networks. The weight update mechanism showing how current weights are modified using gradients computed via backpropagation, processed by an optimizer, and scaled by the learning rate η.

---

### Exercises
-   Derive the gradient of a single neuron with respect to its weights and bias, assuming a sigmoid activation function and MSE loss.
-   Build a training loop with backpropagation for a binary classification task on synthetic data.
-   Visualize the decision boundary learned by a network on a 2D classification problem. How does it change as the network trains?

### Further Reading
-   Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*.
-   Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv*.

### Online Resources
-   **[Karpathy's micrograd GitHub](https://github.com/karpathy/micrograd)**: Minimal autograd & neural-net engine (≈150 LOC) perfect for digging into backprop under the hood.
-   **[Zero-to-Hero micrograd lecture notebooks](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)**: Companion code to the lecture series that birthed *micrograd*. 