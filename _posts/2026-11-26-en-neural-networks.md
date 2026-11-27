---
title: "Neural Networks: Forward Pass, Loss, and Backpropagation"
layout: article
lang: en
permalink: /writing/neural-networks/
nav_exclude: true
description: "Review neural network layers, activations, loss, gradient descent, and backpropagation with a small reproducible numerical example."
date: 2026-11-26 00:00:00 +0000
last_modified_at: 2026-11-27 00:00:00 +0000
article_kind: tutorial
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# Neural Networks: From Forward Pass to Learning

A neural network is a parameterized function built by composing layers. Each layer applies a linear transformation, adds a bias, and may apply a nonlinear activation. Training adjusts the parameters so the network's predictions reduce a chosen loss on examples.

For one layer, the forward calculation is:

```text
z = W x + b
h = activation(z)
```

Here `x` is the input vector, `W` and `b` are learned parameters, and `h` is the layer output. A stack of only linear layers is still one linear transformation, so hidden layers normally include nonlinear activations such as ReLU or GELU. A Transformer block also contains attention and a feed-forward network; the [Transformer tutorial](/writing/transformers/) connects these operations to language models.

## A one-neuron example

Consider a single ReLU neuron with two inputs, a target value, and squared-error loss. We can calculate one forward pass and one gradient-descent update using only Python's standard library:

```python
def predict(x1, x2, w1, w2, bias):
    return max(0.0, w1 * x1 + w2 * x2 + bias)

x1, x2 = 1.0, 2.0
target = 1.0
w1, w2, bias = 0.5, -0.25, 0.1
learning_rate = 0.1

prediction = predict(x1, x2, w1, w2, bias)
error = prediction - target
loss = 0.5 * error ** 2

# The pre-activation is positive, so ReLU's derivative here is 1.
grad_w1 = error * x1
grad_w2 = error * x2
grad_bias = error

w1 -= learning_rate * grad_w1
w2 -= learning_rate * grad_w2
bias -= learning_rate * grad_bias

new_prediction = predict(x1, x2, w1, w2, bias)
new_loss = 0.5 * (new_prediction - target) ** 2

print(round(loss, 4), round(new_loss, 4))  # 0.405, 0.0648
```

This example deliberately has one trainable unit and one update. It is not an LLM training procedure. It shows the basic loop: compute a prediction, measure error, compute parameter gradients, and adjust parameters in the direction that lowers the loss.

## What backpropagation does

For a network with many layers, the loss depends on early parameters through a chain of operations. The chain rule lets us calculate how a small change to each parameter would change the loss. Backpropagation applies that rule from the output toward earlier layers, reusing intermediate derivatives rather than perturbing each parameter one at a time.

For the neuron above, with positive pre-activation and loss `0.5 * (prediction - target)^2`, the gradient for `w1` is:

```text
d(loss)/d(w1) = (prediction - target) * d(prediction)/d(w1)
               = (prediction - target) * x1
```

That derivative is `-0.9` for this example, so subtracting `learning_rate * gradient` increases `w1`. If a ReLU neuron's pre-activation is negative, its local derivative is zero; this simple example would then send no gradient through that unit. Real networks use many units and may use other activation functions.

## Loss, gradients, and generalization

The loss function defines what the training process is trying to improve. Mean squared error is common for regression; cross-entropy is common for classification and next-token prediction. The optimizer uses gradients to update parameters. A learning rate controls update size: too large can make optimization unstable, while too small can make progress slow.

Lower training loss does not automatically mean better behavior on new inputs. A model can memorize examples or fit noise. Training, validation, and test data serve different purposes: use training data to fit parameters, validation data to compare choices, and hold out test data for a final estimate. Repeatedly tuning against the test set leaks information into the development process.

Initialization, normalization, residual connections, and optimizer choices can affect whether deep networks train well, but there is no single recipe that fits every architecture. In LLMs, the same basic calculus is applied to very large parameter tensors; frameworks such as PyTorch build a computation graph and calculate gradients automatically.

## Further reading

- PyTorch, [The Fundamentals of Autograd](https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html), on automatic differentiation and backpropagation.
- Goodfellow, Bengio, and Courville, [Deep Learning, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html).
- Rumelhart, Hinton, and Williams, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0).
