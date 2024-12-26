# Gradient Descent

**Gradient Descent** is an optimization technique used to minimize a function, typically the **loss function** in machine learning, by iteratively adjusting model parameters like weights $`w`$ and bias $`b`$. 
The goal is to find the values that reduce the error.


### Key Concepts
- **Gradient**: The gradient is a generalization of the slope to multiple dimensions. In one dimension, the slope is simply the derivative, but in higher dimensions,
  the gradient is a **vector** of partial derivatives with respect to each parameter. It tells you the direction of the steepest ascent.
  To minimize the function, we move in the opposite direction, steepest descent, that's why we have name **Gradient Descent**.

- **Learning Rate (α)**: A small positive value that determines the size of the steps taken during the gradient update. In more rigorous words: "The learning rate
  determines the magnitude of the changes to make to the weights and bias during each step of the gradient descent process."


## Epochs in Gradient Descent

An **epoch** refers to one complete pass through the entire training dataset. Each epoch consists of multiple steps where the model's parameters are updated 
based on the gradient of the loss function.

## Until Convergence

The phrase "until convergence" means the process continues until the model's parameters stop changing significantly. This happens when:
- The change in the loss function between epochs is small.
- The gradients become very close to zero.

At this point, the model has found the best parameters (or is very close to it), and further updates don't improve performance significantly.
