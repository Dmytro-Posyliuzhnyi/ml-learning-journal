
<details>
  <summary>Types of Machine Learning</summary>

## Types of Machine Learning

### 1. Supervised Learning
- **Dataset:** Contains labeled examples $`(x_i, y_i)`$ where $`x_i`$ is a feature vector, and $`y_i`$ is the corresponding label.
- **Description:**
  
  A **labeled example** may look in the following way:
  - $`x_i`$: A feature vector representing the characteristics of a patient. (e.g., blood sugar level, cholesterol level, weight, height, habits, etc).
  - $`y_i`$: The label or target outcome associated with $`x_i`$. It could be a **binary label** (e.g., $`y_i = 1`$ if the patient has a specific disease, $`y_i = 0`$ otherwise),
                 **multiclass label** (e.g., $`y_i = \text{"diabetes"}`$, $`y_i = \text{"hypertension"}`$), **real number** (e.g., the predicted severity of a condition), etc.
- **Real Life Examples of a Supervised Learning:**
  - Email classification: $`\text{"{spam, not-spam}"}`$
  - Predicting a probability (e.g., cancer diagnosis).
- **Goal:** Train a model to map feature vectors $`x`$ to their labels $`y`$.
 
<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/4c0b97b1-41e6-412a-85e5-51abe79cd688">

---

### 2. Unsupervised Learning
- **Dataset:** Contains only unlabeled examples $`x_i`$.
- **Description:** Unlike supervised learning, there are no labels $`y_i`$ to guide the learning process. Instead, the goal is to identify patterns, structure, or relationships within the data.
- **Real Life Examples of an Unsupervised Learning:**
  - **Clustering:** Grouping similar data points (e.g., grouping customers by purchase history or behavior.).
  - **Dimensionality Reduction:** Simplifying data while retaining its essence (e.g., reducing the size of images while retaining important details.).
  - **Outlier Detection:** Identifying anomalies in the data (e.g., fraud detection).
- **Goal:** Discover patterns or structure in the data.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/89869f38-0b5e-4454-b276-f25fc698d768">

---

### 3. Semi-Supervised Learning
- **Dataset:** Contains both labeled and unlabeled examples.
- **Description:** The algorithm uses the unlabeled examples to learn the structure or distribution of the data, which helps improve the model's predictions on labeled examples.
  This approach assumes that the unlabeled data provides additional information about the problem, such as the underlying data distribution.
- **Real Life Examples of a Semi-Supervised Learning:**

  **Medical Image Analysis:**
   - **Scenario:** Labeling medical images (e.g., MRI scans) requires expert radiologists, making labeled data scarce and expensive.
   - **Solution:** Use a small set of labeled images (e.g., with disease annotations) along with a large set of unlabeled images to improve diagnostic accuracy.
   - **Example:** Identifying tumors in MRI scans with minimal labeled data.
- **Goal:** Leverage unlabeled data to improve model performance.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/551f4da7-f05e-4e94-9a69-8e864e8e5002">

---

### 4. Reinforcement Learning
- **Dataset:** Not explicitly provided; the machine interacts with an environment.
- **Description:** It is a type of learning where an agent learns to make decisions by interacting with an environment.
  The agent observes the current state of the environment, takes actions, and receives rewards or penalties based on those actions.
  Over time, the agent learns a strategy, called a **policy**, to maximize rewards.
- **Real Life Examples of the Reinforcement Learning:**
  **Game Playing:**
   - **Scenario:** Training agents to play complex games.
   - **Example:** AlphaGo, which uses reinforcement learning to master the game of Go by learning strategies that outperform human experts.
- **Goal:** Learn a policy (function) that selects optimal actions to maximize long-term rewards in a sequential decision-making process.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/9c887f3b-4f6d-4bcb-888d-680a254bc7d8">

</details>

<details>
  <summary>Classification Algorithms</summary>

## Classification Algorithms

**Classification algorithms** are a subset of machine learning algorithms used to assign a label or category to input data. They are widely used for tasks like spam detection, image recognition, medical diagnosis, etc.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/598627d5-9115-48bf-af04-36263dc3effa">

### Key Concepts:

1. **Decision Boundary**:
   - A **decision boundary** is a surface that separates the feature space into regions corresponding to different classes.
   - It can take various forms:
     - **Linear (straight line or hyperplane):** Algorithms like Logistic Regression and Linear SVM.
     - **Non-linear (curved):** Algorithms like Kernel SVM and Neural Networks.
     - **Complex shapes:** Algorithms like Decision Trees and ensemble methods (e.g., Random Forests).
   - The decision boundary defines the accuracy of the model by determining how well it separates classes in the dataset.

2. **How Classification Algorithms Differ**:
   - **Form of Decision Boundary:**
     - Each algorithm uses a unique method to compute the decision boundary based on the training data.
   - **Training Speed:**
     - Some algorithms train quickly, while others take longer due to complexity.
   - **Prediction Speed:**
     - Algorithms also differ in how fast they make predictions.

</details>

