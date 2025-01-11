
# Polynomial Regression

In case if the data is more complex than a straight line, we still can use a linear model to fir non-linear data. A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called **polynomial regression**. 

Let's take a look at this non-linear and noisy dataset:

<img width="600" alt="Page 1" src="https://github.com/user-attachments/assets/a504bf66-e128-4df3-a1dc-74990e00ddbd">

```python
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
```

Clearly, straight line will never fit this data properly. We can use *scikit-learn* *Polynomial Features* class to transform the training data, adding square (second-degree polynomial) of each feature in the training set as a new feature (in this case there is just one feature):

```python
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

Now the *X_poly* variable contains the original feature value and the square of this value. We can use this variable with *Linear Regression* class from *scikit-learn* and see that estimations are quite close to the original function:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

This model estimates $`\hat{y}=0.56x_1^2 + 0.93x_1 + 1.78`$ when in fact the original function was: $`y=0.5x_1^2 + 1.0x_1 + 2.0 + \text{Gaussian noise}`$

<img width="600" alt="Page 1" src="https://github.com/user-attachments/assets/21000194-f334-4fb9-8247-633a3e9441f9">

Polynomial regression is also capable of finding the relationships between features, which is something a plain linear regression model cannot do. This is made possible by the fact that *PolynomialFeatures* also adds all combinations of features up to the given degree. For example if there were two features *a* and *b*, *PolynomialFeatures* with degree 3 would not onl add the features $a^2$, $`a^2`$, $`b^2`$, $`b^3`$, but also the combinations $`ab`$, $`a^2b`$, and $`ab^2`$. We need to be careful of the combinatorial explosion that can occur when the number of features is large.


