
# Linear Regression

Linear regression is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. 
The variable you are using to predict the other variable's value is called the independent variable. It works by finding the best-fit line (or hyperplane for multiple variables) 
that shows the relationship between these variables.

<img width="500" alt="Page 1" src="https://github.com/user-attachments/assets/9bebe1ac-cd79-4575-a837-c73bf364875d">

For simple linear regression (one feature) the equation look in the following way: $`y = wx + b`$, where:
  - $`y`$: Predicted value (output)
  - $`x`$: Input feature (independent variable)
  - $`w`$: Weight (slope of the line)
  - $`b`$: Bias (intercept, where the line crosses the y-axis)

The primary goal in training a linear regression model is to minimize the cost function.

**How It Works**


1. **Fit the Model**:
    - Use historical data to calculate the parameters ($`w`$ and $`b`$) that minimize the cost function.

2. **Make Predictions**:
    - Plug new input values ($`x`$) into the regression equation to predict $`y`$.

3. **Example**:
    Let's take a look at trivial example which can give us some intuition being the linear regression. Suppose we want to predict a student's exam score ($`y`$)
    based on hours studied ($`x`$).

    Data:
    | Hours Studied ($`x`$) | Exam Score ($`y`$) |
    |------------------------|-------------------|
    | 2                      | 50                |
    | 4                      | 60                |
    | 6                      | 70                |
    | 8                      | 80                |

    Model:
    $`y = 5x + 40`$

    Prediction for 7 hours studied:
    $`y = 5(7) + 40 = 75`$

You can think of Linear Regression as a function that finds the best-fit line through historical data, allowing you to predict, with high accuracy, 
values for points that were not part of the training data. Essentially, it fills in the gaps based on the existing data points. 
For example, if I give you a Cartesian plane with 10 points plotted on it and tell you that these points are the result of some underlying function, 
Linear Regression can draw a line that closely approximates the original function. This ensures that the $`y`$-values of the line are not too far from the 
original function's values.
