
*Date: January 13, 2025*

*Time: 10:16 PM*

Today, I finished Chapter 4. The last couple of pages were related to *Logistic Regression*. As I already mentioned, before starting Chapter 5, I will try to implement my own regression model. Yesterday, I mentioned that I had an idea, but I decided it would not be suitable at this point for learning purposes. So, most likely, I will just search for a publicly available dataset that I like. 

---

*Date: January 12, 2025*

*Time: 1:25 PM*

Learned more about regularization. Specifically about *L2* regularization (Ridge regression). Explanation that helped me to get intuition behind *L2* regularization: [Overfitting: L2 regularization](https://developers.google.com/machine-learning/crash-course/overfitting/regularization)

*Time: 3:05 PM*

Added notes on *Lasso* and *Elastic Net Regression*. I noticed that recently I've been copying a lot of notes from books, which feels like "notes for the sake of notes." I should avoid this approach and focus on gaining a deeper intuition behind what I learn, and only then explain it in my own words.

*Time: 8:39 PM*

Added notes on *early stopping*. The next topic in this chapter is on *Logistic Regression*. I've got a nice idea for the pet project I'm gonna create after this chapter to retain information I've learned.

---

*Date: January 11, 2025*

*Time: 00:42 AM*

Improved my understanding of *gradient descent* and took some notes on *batch gradient descent*.

*Time: 11:56 AM*

Took notes on *stochastic gradient descent* and *mini-batch gradient descent*.

*Time: 3:03 PM*

Took notes on *polynomial regression*.

*Time: 5:00 PM*

I am having some difficulties getting the intuition behind the *learning curves*. I will take a short break and then get back to brainstorming.

*Time: 9:48 PM*

Took notes on *learning curves*. Understood them well enough. Seems like I was just tired. 

After Chapter 4 I want to start a pet project, to sum up what I've learned so far. Most likely I will focus on regression related problem.  

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
