# Support Vector Machines Classifier

Hello friends,

Support Vector Machines are supervised machine learning algorithms that are used for classification and regression purposes. In this repo, I have built a Support Vector Machines classifier to predict a Pulsar Star with the given attributes. So, let's get started.

<a class="anchor" id="0.1"></a>
# **Table of Contents**


1.	[Introduction to Support Vector Machines](#1)
2.	[Support Vector Machines intuition](#2)
3.	[Kernel trick](#3)
4.	[SVM Scikit-Learn libraries](#4)
5.	[Dataset description](#5)
6.	[Import libraries](#6)
7.	[Import dataset](#7)
8.	[Exploratory data analysis](#8)
9.	[Declare feature vector and target variable](#9)
10.	[Split data into separate training and test set](#10)
11.	[Feature scaling](#11)
12.	[Run SVM with default hyperparameters](#12)
13.	[Run SVM with linear kernel](#13)
14.	[Run SVM with polynomial kernel](#14)
15.	[Run SVM with sigmoid kernel](#15)
16.	[Confusion matrix](#16)
17.	[Classification metrices](#17)
18.	[ROC - AUC](#18)
19.	[Stratified k-fold Cross Validation with shuffle split](#19)
20.	[Hyperparameter optimization using GridSearch CV](#20)
21.	[Results and conclusion](#21)
22. [References](#22)
                                                                                                                                                                                               
# **1. Introduction to Support Vector Machines** <a class="anchor" id="1"></a>

[Table of Contents](#0.1)


**Support Vector Machines** (SVMs in short) are machine learning algorithms that are used for classification and regression purposes. SVMs are one of the powerful machine learning algorithms for classification, regression and outlier detection purposes. An SVM classifier builds a model that assigns new data points to one of the given categories. Thus, it can be viewed as a non-probabilistic binary linear classifier.

The original SVM algorithm was developed by Vladimir N Vapnik and Alexey Ya. Chervonenkis in 1963. At that time, the algorithm was in early stages. The only possibility is to draw hyperplanes for linear classifier. In 1992, Bernhard E. Boser, Isabelle M Guyon and Vladimir N Vapnik suggested a way to create non-linear classifiers by applying the kernel trick to get maximum-margin hyperplanes. The current standard was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

SVMs can be used for linear classification purposes. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using the **kernel trick**. It enable us to implicitly map the inputs into high dimensional feature spaces.


# **2. Support Vector Machines intuition** <a class="anchor" id="2"></a>

[Table of Contents](#0.1)


Now, we should be familiar with some SVM terminology. 


### Hyperplane

A hyperplane is a decision boundary which separates between given set of data points having different class labels. The SVM classifier separates data points using a hyperplane with the maximum amount of margin. This hyperplane is known as the `maximum margin hyperplane` and the linear classifier it defines is known as the `maximum margin classifier`.


### Support Vectors

Support vectors are the sample data points, which are closest to the hyperplane.  These data points will decide the orientation and position of the separating line or hyperplane better by calculating margins.


### Margin

A margin is a separation gap between the two lines on the closest data points. It is calculated as the perpendicular distance from the line to support vectors or closest data points. In SVMs, we try to maximize this separation gap so that we get maximum margin.

The following diagram illustrates these concepts visually.


### Margin in SVM

![Margin in SVM](https://static.wixstatic.com/media/8f929f_7ecacdcf69d2450087cb4a898ef90837~mv2.png)


### SVM Under the hood

In SVMs, our main objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum margin hyperplane in the following 2 step process –

1.	Generate hyperplanes which segregates the classes in the best possible way. There are many hyperplanes that might classify the data. We should look for the best hyperplane that represents the largest separation, or margin, between the two classes.

2.	So, we choose the hyperplane so that distance from it to the support vectors on each side is maximized. If such a hyperplane exists, it is known as the **maximum margin hyperplane** and the linear classifier it defines is known as a **maximum margin classifier**. 


The following diagram illustrates the concept of **maximum margin** and **maximum margin hyperplane** in a clear manner.


### Maximum margin hyperplane

![Maximum margin hyperplane](https://static.packt-cdn.com/products/9781783555130/graphics/3547_03_07.jpg)


### Problem with dispersed datasets


Sometimes, the sample data points are so dispersed that it is not possible to separate them using a linear hyperplane. 
In such a situation, SVMs uses a `kernel trick` to transform the input space to a higher dimensional space as shown in the diagram below. It uses a mapping function to transform the 2-D input space into the 3-D input space. Now, we can easily segregate the data points using linear separation.


### Kernel trick - transformation of input space to higher dimensional space

![Kernel trick](http://www.aionlinecourse.com/uploads/tutorials/2019/07/11_21_kernel_svm_3.png)

# **3. Kernel trick** <a class="anchor" id="3"></a>

[Table of Contents](#0.1)


In practice, SVM algorithm is implemented using a `kernel`. It uses a technique called the `kernel trick`. In simple words, a `kernel` is just a function that maps the data to a higher dimension where data is separable. A kernel transforms a low-dimensional input data space into a higher dimensional space. So, it converts non-linear separable problems to linear separable problems by adding more dimensions to it. Thus, the kernel trick helps us to build a more accurate classifier. Hence, it is useful in non-linear separation problems.

We can define a kernel function as follows-


### Kernel function

![Kernel function](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTodZptqcRor0LGo8Qn7_kJB9n9BACMt6jgIPZ4C3g_rgh_uSRZLQ&s)

In the context of SVMs, there are 4 popular kernels – `Linear kernel`,`Polynomial kernel`,`Radial Basis Function (RBF) kernel` (also called Gaussian kernel) and `Sigmoid kernel`. These are described below -

## **3.1 Linear kernel**

In linear kernel, the kernel function takes the form of a linear function as follows-

linear kernel : K( $x_{i}$ , $x_{j}$ ) = $x_{i}^{T}$ . $x_{j}$

Linear kernel is used when the data is linearly separable. It means that data can be separated using a single line. It is one of the most common kernels to be used. It is mostly used when there are large number of features in a dataset. Linear kernel is often used for **text classification** purposes.

Training with a linear kernel is usually faster, because we only need to optimize the C regularization parameter. When training with other kernels, we also need to optimize the γ parameter. So, performing a grid search will usually take more time.

Linear kernel can be visualized with the following figure.

### Linear Kernel

![Linear Kernel](https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_kernels_thumb.png)

## **3.2 Polynomial Kernel**

Polynomial kernel represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables. The polynomial kernel looks not only at the given features of input samples to determine their similarity, but also combinations of the input samples.

For degree-d polynomials, the polynomial kernel is defined as follows –

**Polynomial kernel : K($x_{i}$ , $x_{j}$ ) = ($γx_{i}^{T}$ $x_{j}$ + r)d , γ > 0**

Polynomial kernel is very popular in Natural Language Processing. The most common degree is d = 2 (quadratic), since larger degrees tend to overfit on NLP problems. It can be visualized with the following diagram.

### Polynomial Kernel

![Polynomial Kernel](https://www.researchgate.net/profile/Cheng_Soon_Ong/publication/23442384/figure/fig12/AS:341444054274063@1458418014823/The-effect-of-the-degree-of-a-polynomial-kernel-The-polynomial-kernel-of-degree-1-leads.png)

## **3.3 Radial Basis Function Kernel**

Radial basis function kernel is a general purpose kernel. It is used when we have no prior knowledge about the data. The RBF kernel on two samples x and y is defined by the following equation –


### Radial Basis Function kernel

![RBK Kernel](data:image/png;base64,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)

The following diagram demonstrates the SVM classification with rbf kernel.

### SVM Classification with rbf kernel

![SVM Classification with rbf kernel](https://www.researchgate.net/profile/Periklis_Gogas/publication/286180566/figure/fig5/AS:304327777374210@1449568804246/An-example-of-an-SVM-classification-using-the-RBF-kernel-The-two-classes-are-separated.png)

## **3.4 Sigmoid kernel**

Sigmoid kernel has its origin in neural networks. We can use it as the proxy for neural networks. Sigmoid kernel is given by the following equation –

**sigmoid kernel : k (x, y) = tanh(αxTy + c)**

Sigmoid kernel can be visualized with the following diagram-

### Sigmoid kernel

![Sigmoid kernel](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKeXbOIlniBXYwMYlEYLKPwZZg8vFU1wVm3RWMACjVcT4iBVDy&s)

# **4. SVM Scikit-Learn libraries** <a class="anchor" id="4"></a>

[Table of Contents](#0.1)


Scikit-Learn provides useful libraries to implement Support Vector Machine algorithm on a dataset. There are many libraries that can help us to implement SVM smoothly. We just need to call the library with parameters that suit to our needs. In this project, I am dealing with a classification task. So, I will mention the Scikit-Learn libraries for SVM classification purposes.

First, there is a **LinearSVC()** classifier. As the name suggests, this classifier uses only linear kernel. In LinearSVC() classifier, we don’t pass the value of kernel since it is used only for linear classification purposes.

Scikit-Learn provides two other classifiers - **SVC()** and **NuSVC()** which are used for classification purposes. These classifiers are mostly similar with some difference in parameters. **NuSVC()** is similar to **SVC()** but uses a parameter to control the number of support vectors. We pass the values of kernel, gamma and C along with other parameters. By default kernel parameter uses rbf as its value but we can pass values like poly, linear, sigmoid or callable function.

# **5. Dataset description** <a class="anchor" id="5"></a>

[Table of Contents](#0.1)


I have used the **Predicting a Pulsar Star** dataset for this project.

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matter. Classification algorithms in particular are being adopted, which treat the data sets as binary classification problems. Here the legitimate pulsar examples form  minority positive class and spurious examples form the majority negative class.

The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive).


### Attribute Information:


Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile. The remaining four variables are similarly obtained from the DM-SNR curve . These are summarised below:

1. Mean of the integrated profile.

2. Standard deviation of the integrated profile.

3. Excess kurtosis of the integrated profile.

4. Skewness of the integrated profile.

5. Mean of the DM-SNR curve.

6. Standard deviation of the DM-SNR curve.

7. Excess kurtosis of the DM-SNR curve.

8. Skewness of the DM-SNR curve.

9. Class

# **6. Import libraries** <a class="anchor" id="6"></a>

[Table of Contents](#0.1)

I will start off by importing the required Python libraries.

# **7.Import dataset** <a class="anchor" id="7"></a>

[Table of Contents](#0.1)

Use pd.read_csv() to read the csv file

# **8. Exploratory data analysis** <a class="anchor" id="8"></a>

[Table of Contents](#0.1)

Now, I will explore the data to gain insights about the data. We can see that there are 9 variables in the dataset. 8 are continuous variables and 1 is discrete variable. The discrete variable is `target_class` variable. It is also the target variable. 

# **9. Declare feature vector and target variable** <a class="anchor" id="9"></a>

[Table of Contents](#0.1)

X = df.drop(['target_class'], axis=1)

y = df['target_class']

# **10. Split data into separate training and test set** <a class="anchor" id="10"></a>

[Table of Contents](#0.1)

Use train_test_split from sklearn.model_selection

# **11. Feature Scaling** <a class="anchor" id="11"></a>

[Table of Contents](#0.1)

Use StandardScaler from sklearn.preprocessing

# **12. Run SVM with default hyperparameters** <a class="anchor" id="12"></a>

[Table of Contents](#0.1)

Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters. Also trained with C=100.0 and C=1000.0 with kernal=rbf 

# **13. Run SVM with linear kernel** <a class="anchor" id="13"></a>

[Table of Contents](#0.1)

linear_svc=SVC(kernel='linear', C=1.0). Also trained with C=100.0 and C=1000.0 with kernal=rbf 

# **14. Run SVM with polynomial kernel** <a class="anchor" id="14"></a>

[Table of Contents](#0.1)

poly_svc=SVC(kernel='poly', C=1.0). Also trained with C=100.0 and C=1000.0 with kernal=rbf 

# **15. Run SVM with sigmoid kernel** <a class="anchor" id="15"></a>

[Table of Contents](#0.1)

sigmoid_svc=SVC(kernel='sigmoid', C=1.0). Also trained with C=100.0 and C=1000.0 with kernal=rbf 

### Comments

We can see that sigmoid kernel is also performing poorly just like with polynomial kernel. We get maximum accuracy with rbf and linear kernel with C=100.0. and the accuracy is 0.9832. Based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.

# **16. Confusion Matrix** <a class="anchor" id="16"></a>

[Table of Contents](#0.1)

A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

- **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

- **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

- **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**

- **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

# **17. Classification Matrices** <a class="anchor" id="17"></a>

[Table of Contents](#0.1)

**Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later. We can import classification_report from sklearn.metrics

# **18. ROC-AUC Curve** <a class="anchor" id="18"></a>

[Table of Contents](#0.1)

Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various classification threshold levels. 

The **ROC Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.

**True Positive Rate (TPR)** is also called **Recall**. It is defined as the ratio of `TP to (TP + FN)`.

**False Positive Rate (FPR)** is defined as the ratio of `FP to (FP + TN)`.

![image](https://user-images.githubusercontent.com/35486320/190153769-e862ecd1-c0f0-4b5a-9a12-eb75ca71a3cf.png)

**ROC AUC** stands for **Receiver Operating Characteristic - Area Under Curve**. It is a technique to compare classifier performance. In this technique, we measure the `area under the curve (AUC)`. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. 

So, **ROC AUC** is the percentage of the ROC plot that is underneath the curve.

### Comments

- ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.

- ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in classifying the pulsar star.

- You can calculate ROC AUC value using cross_val_score using sklearn.model_selection

# **19. Stratified k-fold Cross Validation with shuffle split** <a class="anchor" id="19"></a>

[Table of Contents](#0.1)

k-fold cross-validation is a very useful technique to evaluate model performance. But, it fails here because we have a imbalnced dataset. So, in the case of imbalanced dataset, I will use another technique to evaluate model performance. It is called `stratified k-fold cross-validation`.

In `stratified k-fold cross-validation`, we split the data such that the proportions between classes are the same in each fold as they are in the whole dataset. Moreover, I will shuffle the data before splitting because shuffling yields much better result.

\begin{code}
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5, shuffle=True, random_state=0)
linear_svc=SVC(kernel='linear')
linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)
\end{code}
