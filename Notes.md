
*Date: January 11, 2025*

*Time: 00:42 AM*

Improved my understanding of *gradient descent* and took some notes on *batch gradient descent*.

*Time: 11:56 AM*

Took notes on *stochastic gradient descent* and *mini-batch gradient descent*.

*Time: 3:03 PM*

Took notes on *polynomial regression*.

---

*Date: January 10, 2025*

*Time: 10:02 PM*

So yesterday I was struggling to get the intuition behind the *normal equation*, *ordinary least squares*, and how *LinearRegression* in *scikit-learn* is connected to all of that. Today, I was finally able to connect the different concepts in my mind, and I explained it in [4. Optimization Techniques.md](https://github.com/Dmytro-Posyliuzhnyi/ml-learning-journal/blob/main/Introduction/4.%20Optimization%20Techniques.md) file. I'm actually excited when something clicks in my mind, and it proves what I've been talking about in my math notes: that with consistency, all misunderstandings disappear, and gathering as much information as possible helps to connect things and develop a better intuition.

---

*Date: January 9, 2025*

*Time: 9:56 PM*

Today, I started Chapter 4 as I planned yesterday. I noted some basic information on the **normal equation**, but the book mentioned that the *Linear Regression* class in *scikit-learn* uses the **Least Squares** function for optimization because it's more efficient. I won't lie - I don’t fully understand it yet. First of all, I need to grasp the difference between **least squared error**, **mean squared error**, and other functions I've learned about that sound similar to me. Also, I need to understand whether **Least Squares** is a *closed-form* solution and why it's more efficient than the normal equation.

---

*Date: January 8, 2025*

*Time: 11:02 PM*

Today, I finished Chapter 3. It focused on classification, and we built a model using the MNIST dataset. There was a lot of information about different performance metrics for the model, as well as experimentation with various algorithms and visualizations of the metrics. 

The next chapter is about the internals of training models. I've already been exploring the internals of some algorithms in the *Hundred-Page Machine Learning* book, and for some of them, I was able to gain a really solid mathematical understanding. Let's see what will be in this book.

---

*Date: January 4, 2025*

*Time: 02:22 AM*

Today, I finally completed Chapter 2, where I was creating a model to predict real estate prices. At the end of the day, the predictions are quite poor, but the book expects them to be that way. As of now, 
I don't have a broad enough understanding to get the exact reason, so I may revisit and improve the model later, or at least understand the root cause of the issue.

I also had a problem understanding whether data transformations should be applied to the test set, and after researching for a while, I found a great 
[answer on StackOverflow](https://stackoverflow.com/questions/68284264/does-the-pipeline-object-in-sklearn-transform-the-test-data-when-using-the-pred).

<img width="900" alt="Page 1" src="https://github.com/user-attachments/assets/7a84c13e-2fad-496b-b353-c3af3061f954">

So it seems like, transformations should be applied. Fortunately, `predict` function handles it and nothing besides calling it seems to be needed, at least in the primitive cases I've been exploring.
