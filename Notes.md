
*Date: 4 January 2025*

*Time: 02:22 AM*

Today, I finally completed Chapter 2, where I was creating a model to predict real estate prices. At the end of the day, the predictions are quite poor, but the book expects them to be that way. As of now, 
I don't have a broad enough understanding to get the exact reason, so I may revisit and improve the model later, or at least understand the root cause of the issue.

I also had a problem understanding whether data transformations should be applied to the test set, and after researching for a while, I found a great 
[answer on StackOverflow](https://stackoverflow.com/questions/68284264/does-the-pipeline-object-in-sklearn-transform-the-test-data-when-using-the-pred).

<img width="900" alt="Page 1" src="https://github.com/user-attachments/assets/7a84c13e-2fad-496b-b353-c3af3061f954">

So it seems like, transformations should be applied. Fortunately, `predict` function handles it and nothing besides calling it seems to be needed, at least in the primitive cases I've been exploring.
