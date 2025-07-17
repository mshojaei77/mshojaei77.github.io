---
title: "Neural Networks"
nav_order: 2
parent: Blog
layout: default
---

# Neural Network Fundamentals

At its core, a **neural network is simply a mathematical expression** or function that takes in a set of numbers as input and produces a set of numbers as output. While the term "neural" suggests biological inspiration, in practice, a **"neuron" is just a thing that holds a number** - specifically, a number between 0 and 1 called its **activation**.

**Artificial Neural Networks (ANNs)** are computational models designed to process information through interconnected computational units. They're a powerful approach to solving complex problems by decomposing them into simpler elements and allowing these elements to create emergent complex behavior.

At their core, ANNs are characterized by two main components: a **set of nodes** and **connections between these nodes**.

* **Nodes** (artificial neurons) act as computational units that receive inputs, process them, and produce outputs. Each neuron holds an activation value and performs operations like summing weighted inputs.
* **Connections** dictate the flow of information between nodes. In **feedforward networks**, information flows unidirectionally from input to output without cycles, while other architectures may allow bidirectional flow.

The network processes information in distinct **layers**:
* **Input Layer:** Neurons whose activations correspond to the input data (e.g., 784 neurons for a 28×28 pixel image, each holding a grayscale value from 0 to 1)
* **Output Layer:** Neurons representing the network's final prediction (e.g., 10 neurons for digit classification, where the brightest activation indicates the network's choice)
* **Hidden Layers:** Intermediate layers that perform hierarchical feature extraction, ideally learning to identify abstract patterns or "subcomponents" - a second layer might detect edges, a third might combine edges into shapes, and subsequent layers build increasingly complex representations

The ability of a network to produce **global behavior that cannot be observed in its individual elements** is known as **emergence**, making networks a powerful tool for various applications. This emergent behavior arises from the complex interactions of nodes through their connections.

## Key Components

**Neurons (Units):** Each artificial neuron receives **inputs** (analogous to synapses), which are **multiplied by weights** representing signal strength. These weighted inputs are summed with a **bias** term, then processed by an **activation function** to determine the neuron's output.

**Weights and Biases:** 
* **Weights** control the strength of connections between neurons. Higher weights indicate stronger influence, while negative weights indicate inhibition.
* **Biases** are additional parameters added to the weighted sum that influence the "trigger happiness" of a neuron, making it more or less likely to activate regardless of inputs.
* These are the **learnable parameters** that training algorithms adjust to improve network performance.
* **Scale of Parameters:** Even a simple digit recognition network with two hidden layers of 16 neurons each can have **around 13,000 total weights and biases** - this massive parameter space is what gives neural networks their expressive power.

**Information Flow:** The fundamental principle is that **activations in one layer determine the activations of the next**. This determination is governed by the weights and biases, which can be efficiently computed using **matrix multiplication** for faster computation.

**Layers:** Neurons are organized into layers:
* **Input layer:** receives raw data (e.g., 784 neurons for a 28×28 pixel image)
* **Hidden layers:** perform intermediate processing and feature extraction through hierarchical decomposition
* **Output layer:** produces final predictions (e.g., 10 neurons for digit classification 0-9)

**Connections:** Each edge carries a signal (activation) from one neuron to another, modulated by its weight. In **feedforward networks**, information flows forward from input to output without cycles.

**The Learning Process:** **Learning** in a neural network refers to the process of **finding a valid setting for all these weights and biases** to solve the problem at hand. The **forward pass** computes outputs layer by layer (input→hidden→output), then we measure error and use **backpropagation** to update weights. Over time, the network "learns" to map inputs to desired outputs by adjusting these parameters. This complex function can be compactly represented using **matrix multiplication**, making the code simpler and faster as libraries optimize these operations.

## Architecture Design

Designing an ANN involves defining its structure, including the number and arrangement of neurons, their connections, and the rules governing their interactions. The layered structure allows for **hierarchical decomposition of the input**, where deeper layers build up increasingly complex features instead of directly interpreting raw data.

**Layer Sizing Considerations:**
* **Input and Output Layers:** The input layer size is dictated by the input data (e.g., 784 for a 28×28 image), and the output layer size is determined by the number of possible predictions (e.g., 10 for digits).
* **Hidden Layers:** The number of hidden layers and neurons within them are often **arbitrary choices or hyperparameters** that require experimentation.

**Multi-Layer Perceptrons (MLPs):** The "plain vanilla" neural network, where layers are sequentially stacked and each neuron in one layer is fully connected to all neurons in the next layer. MLPs are used for various tasks, including character-level language modeling and classification.

**Handling Categorical Inputs: Embedding Layers**
For categorical inputs (characters, words), direct integer indices can't be plugged into neural networks effectively:

* **One-hot Encoding:** Converts an integer (e.g., index 13 for 'm') into a vector of all zeros except for a '1' at the 13th position.
* **Embedding Lookup Tables:** An **embedding layer** (or lookup table `C`) maps these discrete indices to **dense, lower-dimensional feature vectors** (e.g., 27 characters into a 2-dimensional space or 17,000 words into a 30-dimensional space).
* **Training Benefits:** These embedding vectors are initially random but are **tuned during backpropagation**, allowing words/characters with similar meanings or contexts to cluster together in the embedding space.
* **Efficiency:** Conceptually equivalent to a linear layer that receives one-hot encoded inputs, but much more efficient.

**Context Length Handling:** Simple bigram models using lookup tables **"blow up" exponentially** as context length increases (e.g., 27×27 for two characters of context). MLPs, with their ability to process continuous embeddings and learn complex relationships, are more scalable for longer contexts.

**Hierarchical Information Fusion (WaveNet-like Architecture):** Instead of flattening all input context into a single large vector, deeper models can **progressively fuse information hierarchically**:
* Combining information from consecutive elements in stages (e.g., two characters, then two bigrams, then two four-grams)
* Building up understanding of context slowly and effectively
* Seen in architectures like WaveNet, which uses "dilated causal convolution layers" for efficient sequential data processing

**Non-linearities:** Essential for network expressiveness. Without non-linear activation functions, stacked linear layers collapse into a single linear transformation, severely limiting representational power.

**Data Splits:** Datasets are divided into three parts:
* **Training Set:** Used to optimize model parameters (weights and biases)
* **Development (Validation) Set:** Used to tune hyperparameters
* **Test Set:** Used for final, unbiased evaluation after training and tuning are complete

If training loss is much lower than validation/test loss, the model is **overfitting**. If both remain high, it may be **underfitting**.

## Activation Functions

An **activation function** is a mathematical function that applies a non-linear transformation to the weighted sum of inputs and bias in a neuron. Its primary role is to introduce non-linearity into the network, enabling it to learn complex patterns and approximate arbitrary functions. Without non-linear activation functions, a deep neural network would behave like a single linear layer, severely limiting its representational power.

Common activation functions include:

**Sigmoid Function:** Also known as the logistic curve, the sigmoid function squashes its input into a range between 0 and 1. Very negative inputs result in outputs close to 0, very positive inputs result in outputs close to 1, and there's a smooth, S-shaped transition around 0:

```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Example usage:
x = np.array([-3.0, 0.0, 2.0])
print(sigmoid(x))  # [0.0474, 0.5, 0.8808]
```

Sigmoid outputs can be interpreted as probabilities (useful for binary classification) and is smooth/differentiable. Early neural networks frequently used sigmoid functions, partly motivated by a biological analogy of neurons being either inactive or active.

**Problem with Sigmoid:** Its **"flat tails"** (where inputs are very positive or very negative) cause the derivative to be very close to zero. This leads to **vanishing gradients**, effectively "killing" the gradient signal during backpropagation and slowing down or stopping learning for earlier layers. Modern networks rarely use sigmoid for this reason.

**Tanh (Hyperbolic Tangent):** Similar to sigmoid, tanh is also a squashing function, but it maps inputs to a range between -1 and 1. It has a smooth, S-shaped curve, with outputs close to -1 for large negative inputs and close to 1 for large positive inputs. It's zero-centered, which often speeds convergence compared to sigmoid. Its derivative is `1 - output^2`.

**Problem with Tanh:** Also suffers from the **vanishing gradient problem** due to its flat tails, making it unsuitable for deep networks.

**ReLU (Rectified Linear Unit):** ReLU is defined as `max(0, x)`. It outputs the input directly if it's positive, and zero otherwise:

```python
def relu(x):
    return np.maximum(0, x)
# Example usage:
x = np.array([-2.0, 0.0, 3.5])
print(relu(x))  # [0.0, 0.0, 3.5]
```

**Advantages of ReLU:** Much easier to train than sigmoid/tanh due to a non-zero derivative for positive inputs, and is very fast to compute.

**Problem with ReLU: "Dead ReLU":** If a neuron's pre-activation (weighted sum) is consistently negative, it will always output zero, and its gradient will be exactly zero. Such a "dead" neuron will never learn or update its weights. This can happen at initialization or during training with high learning rates.

### Challenges with Activation Functions

**Saturation and Vanishing Gradients:** Both sigmoid and tanh functions suffer from **saturation**, where the "tails" of the function (very positive or very negative inputs) become very flat. When a neuron's activation falls into these flat regions, the derivative (local gradient) becomes very close to zero. During backpropagation, this near-zero gradient is multiplied with the incoming gradient, effectively **"killing" or "vanishing" the gradient** for earlier layers. This makes it difficult for weights connected to saturated neurons to learn.

**Dead Neurons:** For ReLU, there's a flat region below zero. If a ReLU neuron's pre-activation is always negative for all training examples, it will continuously output zero. In this "dead" state, its gradient will be exactly zero, meaning its weights and bias will never update. This can occur during initialization or training if a high learning rate causes the neuron to be permanently inactive.

**Importance of Well-Behaved Activations:** Ideally, the **distribution of activations** throughout the network should be **roughly Gaussian (zero mean, unit standard deviation)**. This is critical for effective training:

* **Too Small Activations:** If activations are too small, they might fall into the inactive regions of activation functions, leading to poor gradient flow.
* **Too Large Activations:** If activations are too large, they can cause saturation (flat tails), leading to vanishing gradients and slow learning.
* **Proper Initialization:** Strategies like **Kaiming initialization** aim to preserve this desirable distribution and prevent saturation or shrinking activations from the start.
* **Normalization Techniques:** Methods like **Batch Normalization** actively maintain well-behaved activation distributions during training.

The goal is to keep activations in the "sweet spot" where activation functions have meaningful gradients and can effectively propagate learning signals throughout the network.

## Gradients and Backpropagation

**Gradients** are a fundamental concept in neural network training. The **gradient** is a multi-dimensional slope that tells us **how to efficiently decrease the cost function**. For each weight and bias in the network, the gradient provides two key pieces of information:

* **Sign:** Indicates whether the parameter should be nudged up or down to decrease the cost.
* **Relative Magnitude:** Tells us the "importance" or "bang for your buck" of changing that specific parameter; larger magnitudes mean a small nudge will have a greater impact on the cost.

In neural networks, we're interested in the gradient of the **loss function** with respect to the network's **weights and biases**. This gradient indicates how sensitive the loss is to changes in these parameters, guiding how to adjust them to most efficiently decrease the loss.

**Backpropagation** is the **core algorithm for efficiently computing these gradients**. It's the workhorse behind how neural networks learn and operates in a **supervised learning** setting, where the algorithm is provided with examples of inputs and their desired outputs. The intuitive idea behind backprop is to propagate "desired nudges" or "sensitivities" backward through the network.

### The Mechanism of Backpropagation

Backpropagation relies heavily on the **chain rule from calculus**. It's a systematic way to apply the chain rule efficiently through the network's computation graph. Here's the formal process:

**1. Forward Pass:** The input data flows through the network, activations and all intermediate values are computed, and a **computation graph** (or "expression graph") is implicitly built, tracking all operations and their dependencies.

**2. Compute Loss:** A single scalar value, the **loss**, is calculated based on the network's output and the true labels.

**3. Initialize Gradient:** The `grad` attribute of the final output node (the loss) is manually set to **1.0** (representing `dL/dL`, the derivative of the loss with respect to itself).

**4. Topological Sort:** The nodes in the computation graph are ordered topologically, ensuring that a node's backward function is called only after all nodes that depend on its output have had their backward functions called.

**5. Backward Pass (Chain Rule Application):** The algorithm iterates through the nodes in **reverse topological order**, from the loss back to the input parameters. For each node:
   * It retrieves the **global gradient** (the derivative of the final loss with respect to the node's output), which has been accumulated from subsequent operations.
   * It applies the **local derivative** (the derivative of the node's output with respect to its own inputs).
   * It then **multiplies the global gradient by the local derivative** (the chain rule) and **adds** this product to the `grad` attribute of the node's input(s). This is expressed as `input.grad += local_derivative * output.grad`. The `+=` is crucial to accumulate gradients from multiple paths in the computation graph.
   * This process effectively "chains" the gradient signal backward, populating the `grad` attribute for all intermediate values and, crucially, for the **weights and biases**.

**Backpropagation Working Backward:** Starting from understanding how sensitive the cost function is to the activations in the output layer, it works backward, layer by layer, to determine how much each weight and bias in previous layers contributed to that sensitivity. For a single neuron's contribution, the "nudges" from multiple subsequent neurons are added together, and the desired changes (gradients) are averaged across all training examples.

**Key Operations:**
* **Addition:** The gradient is distributed equally to the inputs
* **Multiplication:** The local derivative is the *other* term being multiplied
* **Matrix Multiplication:** Involves matrix multiplication with transposed matrices during backpropagation
* **Summation and Broadcasting:** Forward summation becomes backward broadcasting (replication), and forward broadcasting becomes backward summation

**Analytic Derivatives:** For complex operations like Softmax followed by Cross-Entropy Loss, or Batch Normalization, it's often more efficient to analytically derive a single, simplified backward formula rather than backpropagating through each atomic piece. This leads to faster and more numerically stable computations.

**Autograd Engines:** Modern deep learning frameworks like PyTorch have **autograd engines** that automatically handle the creation of the computational graph and the application of the chain rule. This abstracts away manual gradient derivation, but understanding the internals is crucial for debugging and optimization.

### Training Process

The training process follows these steps:

1. **Forward Pass:** Input data flows through the network, computing predictions and loss
2. **Error Calculation:** Compare network output to desired targets to compute error
3. **Backward Pass:** Apply backpropagation to compute gradients for all parameters
4. **Parameter Update:** Adjust weights and biases using gradients to minimize loss
5. **Repeat:** Continue this process across multiple epochs until convergence

**Critical Implementation Details:**
* **Zeroing Gradients:** Before each backward pass, gradients must be reset to zero (or `None`) to prevent accumulation from previous steps
* **Gradient Diagnostics:** Monitor gradient statistics (histograms, magnitudes) to detect issues like vanishing/exploding gradients
* **Update-to-Data Ratio:** The ratio of gradient updates to parameter values should ideally be around 10^-3 on a log scale for stable training

The goal of backpropagation is to **adjust the network's weights and biases to minimize error**. Starting with randomly initialized weights, the algorithm iteratively adjusts them until the error is minimal, meaning the network has "learned" the training data.

## Loss Functions and Regularization Strategies

**Loss functions** (also called **cost functions**) are central to neural network training. A **loss function** quantifies how well a neural network is performing by taking the network's approximately 13,000 weights and biases as input and outputting a **single number representing the "lousiness"** of those parameters based on the network's behavior across the training data. 

The primary **goal of training is to minimize this loss function**. A lower loss indicates that the network's predictions are closer to the desired outputs. A good loss function must be **smooth** to enable gradient descent optimization.

### Common Loss Functions

**Mean Squared Error (MSE):** This loss function is typically used for **regression** tasks. It calculates the sum of the squared differences between the network's output activations and the desired target values:

```python
import numpy as np
y_true = np.array([3.0, 5.0])
y_pred = np.array([2.5, 4.8])
mse = np.mean((y_pred - y_true)**2)
print(mse)  # e.g. 0.145
```

Squaring the difference ensures that errors, whether positive or negative, contribute positively to the total loss, and larger differences result in disproportionately larger loss.

**Negative Log Likelihood (NLL):** This loss is commonly used for **classification** tasks, especially in language modeling where the goal is to predict the probability distribution of the next character or word. It is derived from the principle of **maximizing the likelihood** of the observed training data given the model's parameters.

* **Likelihood:** The likelihood of a model's predictions is the product of the probabilities it assigns to each correct outcome. Higher likelihood means better performance.
* **Log-Likelihood:** For numerical stability (multiplying many small probabilities leads to very tiny numbers), the **log-likelihood** is used instead. Since log(a×b) = log(a) + log(b), the log-likelihood is the sum of logarithms of individual probabilities.
* **Negative Log-Likelihood:** To align with the goal of *minimization*, the **negative log-likelihood** is used. **Maximizing likelihood is equivalent to minimizing negative log likelihood**. The lowest possible NLL is zero (when all probabilities for correct outcomes are 1).
* **Interpretation:** A lower NLL means the network is assigning **higher probabilities to the correct labels** for the training examples.

**Cross-Entropy Loss:** Widely used for classification, often with Softmax activation in the output layer. Cross-entropy combines the logarithm and normalization steps efficiently and is designed to be numerically stable:

```python
p = np.array([0.9, 0.2, 0.7])  # predicted probabilities
y = np.array([1,   0,   1])    # true labels
loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
```

Cross-entropy grows large if the model is confident and wrong, encouraging the model to predict true classes with high probability. For multi-class problems, categorical cross-entropy is used with softmax activation.

**Intuitive Gradient Behavior:** The gradients of cross-entropy loss have an intuitive interpretation: they **"pull down" the probabilities of incorrect characters and "pull up" the probability of the correct character**, with the force being proportional to how confident and wrong the network was.

**Numerical Stability:** For numerical stability and efficiency, deep learning libraries provide highly optimized implementations (e.g., `torch.nn.functional.cross_entropy`) that fuse several operations and handle extreme values.

### Regularization Strategies

**Regularization** refers to techniques used to prevent a model from **overfitting** the training data. Overfitting occurs when a model learns the training examples too well, including noise, leading to poor performance on new data (memorization vs. generalization).

**Model Smoothing (for simple models):** In count-based language models, zero probabilities for unseen bigrams lead to infinite loss. **Adding "fake counts"** (e.g., adding 1 to all counts) ensures that no probability is exactly zero, "smoothing out" the distribution and preventing infinite loss. This is analogous to regularization in neural networks.

**L2 Regularization (Weight Decay):** Adds a penalty to the loss function based on the magnitude of weights. The sum of squared weights is added to the original loss, multiplied by a small **regularization strength** hyperparameter:

```python
lambda_reg = 0.01
loss = mse + lambda_reg * np.sum(W**2)  # W = weight matrix
```

This **"spring force" or "gravity force"** pushes weights towards zero, encouraging a simpler model and **reducing overfitting**. It's functionally equivalent to model smoothing in simple cases, where incentivizing weights to be near zero leads to more uniform probability distributions and prevents weights from growing too large and capturing noise.

**L1 Regularization:** Adds a penalty based on the absolute values of weights (λ·∑|w|). This can drive some weights exactly to zero, effectively performing feature selection by zeroing out irrelevant weights.

**Dropout:** During training, randomly "drop" a fraction of neurons (set output to zero) on each forward pass. With dropout rate 0.5, each neuron has a 50% chance of being ignored. This prevents neurons from co-adapting too strongly and acts like an ensemble of many thinner networks.

**Batch Normalization as Regularizer:** Surprisingly, **Batch Normalization** has been found to have a regularization effect. By coupling computation across examples within a batch (normalizing by batch statistics), it introduces slight "jitter" or noise into activations, effectively acting as **data augmentation** and making it harder for the network to overfit.

**Model Smoothing:** In count-based models, adding "fake counts" ensures no sequence has zero probability. In neural networks, initializing weights close to zero or using L2 regularization serves a similar purpose, making initial distributions more uniform.

The effectiveness of neural networks heavily depends on having **lots of training data**. Large datasets provide the examples necessary for training robust models that generalize well to new, unseen data.

## Optimization Algorithms

Once a neural network's architecture is defined and a loss function is chosen, **optimization algorithms** are used to iteratively adjust the network's parameters (weights and biases) to minimize the loss.

### Gradient Descent

**Gradient Descent** is the foundational optimization algorithm. The core idea is to **repeatedly nudge the network's parameters (weights and biases) by some multiple of the negative gradient** of the loss function. Since the gradient points towards the steepest increase in loss, the **negative gradient** points in the direction of the steepest decrease. This process allows the network to **"step downhill"** on the complex "cost surface" (representing the loss function in its high-dimensional input space) and **converge towards a local minimum**.

**Learning Rate (Step Size):** Each update step is scaled by a **learning rate**, a small positive constant. The learning rate is a critical **hyperparameter**:
* If too small → training will be very slow
* If too large → optimization might become unstable, overshoot the minimum, or cause loss to "explode"

When the slope flattens near a minimum, smaller step sizes help prevent overshooting.

**Finding a Good Learning Rate:** Start with a very low rate and exponentially increase it over a few steps while monitoring the loss, looking for a "valley" where the loss decreases efficiently before becoming unstable.

**Learning Rate Decay:** Often, the learning rate is **decayed** over time, gradually reducing it during training. This allows for larger, faster steps early in training and finer adjustments as the network approaches a minimum.

### Stochastic Gradient Descent (SGD)

Evaluating the gradient over an entire dataset (tens of thousands of examples) is computationally expensive. **Stochastic Gradient Descent (SGD)** addresses this by:

* **Mini-Batches:** Instead of using all training examples, the data is randomly shuffled and divided into small subsets called **mini-batches** (e.g., 100 examples).
* The gradient is computed and parameters are updated based on a single mini-batch.
* **Benefits:** This offers a **significant computational speedup** despite providing only an *approximation* of the true gradient.

While the gradient from a mini-batch is noisy (an approximation), taking many small, quick steps is often more efficient than fewer, precise steps with the full dataset. The training trajectory resembles a **"drunk man stumbling aimlessly down a hill but taking quick steps"** rather than a perfectly calculated path.

**Mini-batch size** is another hyperparameter that influences training stability and speed.

**Learning Rate Diagnostics:** Practitioners often search for a reasonable learning rate by trying exponentially spaced values and observing the loss curve. A useful diagnostic is to monitor the **"update-to-data ratio"** (the ratio of the update magnitude to the parameter's magnitude). A ratio around `1e-3` (negative 3 on a log scale) is often considered good; if it's much lower, training is too slow, and if higher, it's too fast.

**Example:** Minimize a simple function $f(w)=w^2$ with gradient descent:

```python
w = 5.0
lr = 0.1
for i in range(20):
    grad = 2*w           # derivative of w^2 is 2w
    w = w - lr * grad    # update step
print(w)  # approaches 0
```

### Weight Initialization

The initial random values of weights and biases are crucial for training success.

**Proper Initialization:** Poor initialization can significantly hinder or prevent training, especially in deeper networks. If weights are too large, activations can saturate, leading to vanishing gradients. If too small, activations might shrink to zero.

**Goal:** Ensure activations throughout the network remain **well-behaved** with **zero mean and unit standard deviation** to preserve gradient flow during backpropagation. If weights are too large, activations can saturate, leading to vanishing gradients. If too small, activations might shrink to zero.

**Initialization Schemes:**
* **Xavier/Glorot initialization:** Often `sqrt(1/fan_in)` for Tanh/Sigmoid activations
* **He/Kaiming initialization:** `sqrt(2/fan_in)` for ReLU (accounts for ReLU clamping negative values to zero)

These schemes set weight scale based on the number of input connections (`fan_in`) to maintain desirable activation distributions and prevent saturation or shrinking activations from the start.

**Bias Initialization:** Often initialized to zeros, but sometimes small random numbers are used to introduce slight "diversity" and avoid symmetry issues.

### Batch Normalization

Introduced in 2015, **Batch Normalization (BatchNorm)** was a highly influential innovation that **significantly stabilized the training of deep neural networks**.

**Core Idea:** It explicitly normalizes the activations of a layer to have a desired distribution (ideally, **zero mean and unit variance, like a Gaussian**). This prevents activations from becoming too small (inactive) or too large (saturated), issues that hinder gradient flow.

**Mechanism:**
* For each mini-batch, BatchNorm calculates the **mean and standard deviation** of the activations for each neuron independently across the batch.
* It then **normalizes** these activations by subtracting the mean and dividing by the standard deviation (`(x - mean) / std_dev`).
* Critically, it introduces two **learnable parameters**: a **`gain` (gamma)** and a **`bias` (beta)**. These parameters scale and shift the normalized activations, allowing the network to undo the normalization if it deems necessary during training. These are trained via backpropagation.
* During training, BatchNorm also maintains a **`running_mean` and `running_variance`** using an exponential moving average. These running statistics are **not updated via backpropagation** but rather "on the side".
* For **inference (test time)**, the `running_mean` and `running_variance` are used instead of calculating statistics per batch. This ensures deterministic outputs for individual examples.

**Placement:** BatchNorm layers are typically "sprinkled" throughout the network, commonly placed **after linear or convolutional layers and before non-linearities**.

**Interactions:** When BatchNorm is used, the **bias parameter in the preceding linear layer becomes redundant** because BatchNorm has its own learnable bias which effectively "subtracts out" any prior bias.

**Impact:**
* **Stability:** BatchNorm makes neural network training much more stable and robust to poor weight initializations.
* **Regularization:** Unexpectedly, BatchNorm acts as a regularizer. Because the normalization depends on the mini-batch statistics, it introduces a slight "jitter" or "noise" into the activations for any given example depending on what other examples are in its batch. This effectively augments the data and makes it harder for the network to overfit.
* **Enables Deeper Networks:** Mitigates vanishing/exploding gradients across many layers
* **Higher Learning Rates:** Allows more aggressive learning rate settings

**Challenges and Alternatives:**
* The dependence on batch statistics means that examples within a batch are mathematically coupled, leading to non-deterministic outputs if the batch content changes. This **"train-test mismatch"** (using batch stats in training vs. running stats in inference) can sometimes be a source of subtle bugs.
* Because of these issues, newer normalization techniques like **Layer Normalization, Instance Normalization, and Group Normalization** have emerged and become more common, as they do not couple examples across the batch. However, BatchNorm was revolutionary at its time for enabling the training of much deeper networks.

### Modern Optimizers

Beyond vanilla SGD, advanced optimizers improve training stability and convergence:

* **Momentum:** Accumulates velocity (exponential average of past gradients) to smooth updates and accelerate learning in relevant directions
* **Adam:** Combines momentum and per-parameter learning rates, maintaining moving averages of both gradients and their squares. Adapts each weight's learning rate individually and typically works well without much tuning
* **RMSProp:** Adapts learning rates based on recent gradient magnitudes

These optimizers often converge faster and more reliably on complex problems by adapting learning rates or using momentum to navigate the loss landscape more effectively.

**References:** Neural networks are modeled as interconnected neurons with weights and biases. Activation functions like sigmoid, tanh, and ReLU introduce non-linearity. Backpropagation uses the chain rule to compute gradients for training. Common loss functions (MSE, cross-entropy) measure model error, and regularization methods (L1/L2, dropout) help prevent overfitting. Basic optimizers (gradient descent, SGD, Adam) iteratively adjust weights to minimize loss.
