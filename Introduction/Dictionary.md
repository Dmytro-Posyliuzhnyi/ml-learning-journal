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
