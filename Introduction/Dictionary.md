<details>
  <summary>Feature vector</summary>

### Feature vector

A **feature vector** is an ordered list of numerical values that represent the characteristics or properties (features) of an example in a dataset. Each value corresponds to a specific feature, and together, the vector provides a mathematical representation of the example that machine learning algorithms can process.

---

#### Example:

Imagine you are building a model to predict whether a person is likely to develop diabetes. Each person in your dataset is represented by a feature vector:

| Feature                  | Value   |
|--------------------------|---------|
| Age (in years)           | 45      |
| Body Mass Index (BMI)    | 28.5    |
| Glucose Level (mg/dL)    | 120     |
| Exercise Hours per Week  | 3       |

The feature vector for this individual would be:

$`x_i = [45, 28.5, 120, 3]`$

So basically the feature vector in our case is just 4-dimensional vector, which is treated as point in a high-dimensional space.

Feature vectors provide a standardized way to represent data points so that machine learning models can analyze and learn patterns from them.
</details>

<details>
  <summary>Hyperplane/Decision Boundary</summary>

### Hyperplane

*In linear classification algorithms the hyperplane is the same thing as decision boundary*

A **hyperplane** is a flat subspace in a higher-dimensional space that divides the space into two or more regions. In machine learning, hyperplanes sometimes are the same thing as decision boundaries, and decision boundary is used in algorithms to separate data points into different classes.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/717a5724-9631-4f6b-bfba-740429ed4b61">

In a **2D space**, a hyperplane is a **line**:
- $`2x_1 + 3x_2 - 5 = 0`$ represents a line dividing the plane into two regions.

In a **3D space**, a hyperplane is a **plane**:
- $`x_1 + 2x_2 + 3x_3 - 6 = 0`$ represents a plane splitting the 3D space.

In **higher dimensions**, it’s difficult to visualize, but the concept remains the same.

<details>
  <summary>Mathematical Definition</summary>


A hyperplane in a $`D`$-dimensional space is defined by the equation:

$`w_1x_1 + w_2x_2 + \dots + w_Dx_D + b = 0`$

Where:
- $`w_1, w_2, \dots, w_D`$ are the weights (coefficients) of the features.
- $`x_1, x_2, \dots, x_D`$ are the feature values of a data point.
- $`b`$ is the bias (intercept term).

Both weight and bias establish the hyperplane's orientation and position within the input space.

The hyperplane separates the space into regions based on the sign of the equation:
- $`w \cdot x + b > 0`$ on one side.
- $`w \cdot x + b < 0`$ on the other.

</details>

In non-linear models (e.g., Neural Networks, k-Nearest Neighbors) the decision boundary may not be a hyperplane - it could be a curved or irregular surface depending on the data and the model. For example a neural network might create a non-linear decision boundary that adapts to the data's complex shape.

Hyperplane is purely mathematical, while decision boundary is contextual:
- A hyperplane is always flat (linear) and mathematically defined.
- A decision boundary can be linear (a hyperplane) or non-linear, depending on the model.

---

</details>

<details>
  <summary>Margin</summary>

### Margin

The **margin** is the distance between the decision boundary (e.g., a hyperplane) and the closest data points from each class in a classification problem. It is a key concept in machine learning algorithms like **Support Vector Machines (SVMs)**.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/3846f0ec-b7d7-4a03-806f-1ea83462147f">

---

#### Why Is Margin Important?

1. **Generalization**:
   - A larger margin often leads to better generalization, meaning the model performs better on unseen data.

2. **Overfitting**:
   - A small margin increases the risk of overfitting, where the model becomes too sensitive to the training data.

3. **Robustness**:
   - Models with larger margins are less sensitive to small perturbations in the data.

---

</details>

<details>
  <summary>Optimization</summary>

**Optimization** is like the engine that makes machine learning work. At its core, it's all about finding the best values for a model's parameters (like weights and biases) so it performs well on a given task.

</details>

<details>
  <summary>Linear Models</summary>

### Linear Models

Linear models are one of the simplest types of machine learning algorithms. These models make predictions by finding a straight-line (or hyperplane in higher dimensions) relationship between the input features and the output.

---

#### Advantages of Linear Models:
- Easy to interpret (e.g., the coefficients show feature importance).
- Computationally efficient and fast to train.
- Works well when the relationship between features and the target is approximately linear.

#### Disadvantages of Linear Models:
- Struggles with non-linear relationships.
- Sensitive to outliers unless regularization techniques are used.

---

### When to Use Linear Models:
- When your data is linearly separable or has a roughly linear relationship.
- When you need a quick, interpretable model.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/f02e53a0-1c84-4640-97b0-3a369d9af74a">

</details>

<details>
  <summary>Kernels</summary>

### Kernels

Kernels are mathematical functions that enable machine learning algorithms, like Support Vector Machines (SVMs), to handle **non-linear data**. They work by implicitly mapping the original data into a higher-dimensional space where a linear decision boundary can be used.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/e0347f10-0552-4a22-81bc-438747522270">

---

#### Why Kernels Matter:
- They allow algorithms like SVMs to create non-linear decision boundaries.
- Kernels let you handle complex datasets without manually adding features or transforming data.

---

### When to Use Kernels:
- When your data is not linearly separable in the original feature space.
- When you suspect complex relationships between features but don’t want to explicitly define transformations.

</details>

<details>
  <summary>Parameters vs Hyperparameters</summary>
  <img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/0335df3c-cbf7-44fe-8133-35154b988807">
</details>

