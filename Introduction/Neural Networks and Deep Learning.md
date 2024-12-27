# Neural Network Overview

## What is a Neural Network?  
A **Neural Network** is a mathematical function structured to process and analyze data. Its design consists of nested functions, 
giving it a multi-layered architecture..

## Structure and Flow

1. **Input Layer**:  
   The starting point where the network receives the feature vector (e.g., $`[x₁, x₂, ...]`$).  

2. **Hidden Layers**:  
   These are intermediate layers where the real magic happens:  
   - Each layer is a function that evaluates the input data by applying a **linear transformation** (weights and biases) followed by an **activation function**.  
   - Neurons in the same layer detect different patterns in the data due to their unique weights and biases.  

   **Flow**:  
   - The input is passed to the first hidden layer.  
   - The layer processes the input and outputs a vector.  
   - This vector becomes the input to the next layer.  

3. **Output Layer**:  
   The final layer combines and interprets the processed data from the previous layers.  
   - For **classification**, it assigns probabilities to different classes.  
   - For **regression**, it predicts a continuous value.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/498c72da-5943-4409-b60e-91d9ca2cbfc9">

### Current State
I plan to add more detailed information about, Neural Networks but for now, I'm working on developing a stronger intuition behind neural networks 
and their mathematical internals.
