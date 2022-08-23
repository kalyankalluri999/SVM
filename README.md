# 1. Explain SVM to a non-technical person.

Explanation: Suppose you have to construct a bidirectional road. Now you have to make a dividing line. The optimal approach would be to make margins on the sides and draw an equidistant line from both the margins.

![image](https://user-images.githubusercontent.com/93079874/186146922-07c8dd22-60dc-408b-b041-02447c91bd09.png)

This is exactly how SVM tries to classify points by finding an optimal centre line (technically called as hyperplane).


# 2. Can you explain SVM?

Explanation: Support vector machines is a supervised machine learning algorithm which works both on classification and regression problems. It tries to classify data by finding a hyperplane that maximizes the margin between the classes in the training data. Hence, SVM is an example of a large margin classifier.

The basic idea of support vector machines:

a) Optimal hyperplane for linearly separable patterns
b) Extend to patterns that are not linearly separable by transformations of original data to map into new space(i.e the kernel trick)

# 3. What do know about Hard Margin SVM and Soft Margin SVM?

Explanation: If a point Xi satisfies the equation Yi(WT*Xi +b) ≥ 1, then Xi is correctly classified else incorrectly classified. So we can see that if the points are linearly separable then only our hyperplane is able to distinguish between them and if any outlier is introduced then it is not able to separate them. So these type of SVM is called hard margin SVM (since we have very strict constraints to correctly classify each and every data point).

To overcome this, we introduce a term ( ξ ) (pronounced as Zeta)

if ξi= 0, the points can be considered as correctly classified.

if ξi> 0 , Incorrectly classified points.


# 4. What is Hinge Loss?

Explanation: Hinge Loss is a loss function which penalises the SVM model for inaccurate predictions.

If Yi(WT*Xi +b) ≥ 1, hinge loss is ‘0’ i.e the points are correctly classified. When

Yi(WT*Xi +b) < 1, then hinge loss increases massively.

As Yi(WT*Xi +b) increases with every misclassified point, the upper bound of hinge loss {1- Yi(WT*Xi +b)} also increases exponentially.

Hence, the points that are farther away from the decision margins have a greater loss value, thus penalising those points.

![image](https://user-images.githubusercontent.com/93079874/186147642-0b32cc3a-eed8-434a-aef0-3d7dc8f34ac7.png)

We can formulate hinge loss as max[0, 1- Yi(WT*Xi +b)]


# 5. Explain the Dual form of SVM formulation?

Explanation: The aim of the Soft Margin formulation is to minimize

![image](https://user-images.githubusercontent.com/93079874/186147887-a0453607-5741-4acb-829f-bd6c90a18a43.png)

subject to

![image](https://user-images.githubusercontent.com/93079874/186147926-ec3a2260-fe9e-473e-8a9f-b51deb5f9fd7.png)

This is also known as the primal form of SVM.

The duality theory provides a convenient way to deal with the constraints. The dual optimization problem can be written in terms of dot products, thereby making it possible to use kernel functions.

It is possible to express a different but closely related problem, called its dual problem. The solution to the dual problem typically gives a lower bound to the solution of the primal problem, but under some conditions, it can even have the same solutions as the primal problem. Luckily, the SVM problem happens to meet these conditions, so you can choose to solve the primal problem or the dual problem; both will have the same solution.

![image](https://user-images.githubusercontent.com/93079874/186148009-a4c130b6-71b5-4ae6-8be2-fafb37946713.png)

# 6. What’s the “kernel trick” and how is it useful?

Explanation: Earlier we have discussed applying SVM on linearly separable data but it is very rare to get such data. Here, kernel trick plays a huge role. The idea is to map the non-linear separable data-set into a higher dimensional space where we can find a hyperplane that can separate the samples.

![image](https://user-images.githubusercontent.com/93079874/186148386-85d995c3-b184-4b0d-871f-50cd03e0b755.png)

It reduces the complexity of finding the mapping function. So, Kernel function defines the inner product in the transformed space. Application of the kernel trick is not limited to the SVM algorithm. Any computations involving the dot products (x, y) can utilize the kernel trick.


# 7. What is Polynomial kernel?

Explanation: Polynomial kernel is a kernel function commonly used with support vector machines (SVMs) and other kernelized models, that represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables, allowing learning of non-linear models.

For d-degree polynomials, the polynomial kernel is defined as:

![image](https://user-images.githubusercontent.com/93079874/186148506-adf930c5-3983-4c88-8ba2-aafbbdc4a7cc.png)


# 8. What is RBF-Kernel?

Explanation:

The RBF kernel on two samples x and x’, represented as feature vectors in some input space, is defined as

![image](https://user-images.githubusercontent.com/93079874/186148583-07abfbf8-1917-42d1-a1a3-b1a5ac622a62.png)

||x-x’||² recognized as the squared Euclidean distance between the two feature vectors. sigma is a free parameter.


# 9. Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?

Explanation: This question applies only to linear SVMs since kernelized can only use the dual form. The computational complexity of the primal form of the SVM problem is proportional to the number of training instances m, while the computational complexity of the dual form is proportional to a number between m² and m³. So, if there are millions of instances, you should use the primal form, because the dual form will be much too slow.

# 10. Explain about SVM Regression?

Explanation: The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. First of all, because the output is a real number it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM.

![image](https://user-images.githubusercontent.com/93079874/186148753-c1f22ffb-6bbc-4e0a-9da1-56eb8632a673.png)


# 11. Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm.

Explanation:

The main reason to use an SVM instead is that the problem might not be linearly separable. In that case, we will have to use an SVM with a non-linear kernel (e.g. RBF).
Another related reason to use SVMs is if you are in a higher-dimensional space. For example, SVMs have been reported to work better for text classification.

# 12. What is the role of C in SVM? How does it affect the bias/variance trade-off?

Explanation:

![image](https://user-images.githubusercontent.com/93079874/186148936-c65be83a-ccee-4061-9642-169690e38ffb.png)

In the given Soft Margin Formulation of SVM, C is a hyperparameter.

C hyperparameter adds a penalty for each misclassified data point.

Large Value of parameter C implies a small margin, there is a tendency to overfit the training model.

Small Value of parameter C implies a large margin which might lead to underfitting of the model.


# 13.SVM being a large margin classifier, is it influenced by outliers?

Explanation: Yes, if C is large, otherwise not.

# 14. In SVM, what is the angle between the decision boundary and theta?

Explanation: Decision boundary is a plane having equation Theta1*x1+Theta2*x2+……+c = 0, so as per the property of a plane, it’s coefficients vector is normal to the plane. Hence, the Theta vector is perpendicular to the decision boundary.

# 15. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?

Explanation:
Logistic Regression is computationally more expensive than SVM — O(N³) vs O(N²k) where k is the number of support vectors.
The classifier in SVM is designed such that it is defined only in terms of the support vectors, whereas in Logistic Regression, the classifier is defined over all the points and not just the support vectors. This allows SVMs to enjoy some natural speed-ups (in terms of efficient code-writing) that is hard to achieve for Logistic Regression.

# 16. What is the difference between logistic regression and SVM without a kernel?

Explanation: They differ only in the implementation . SVM is much more efficient and has good optimization packages.

# 17. Can any similarity function be used for SVM?

Explanation: No. It has to have to satisfy Mercer’s theorem.

# 18. Does SVM give any probabilistic output?

Explanation: SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
