# Machine Learning Tasks


## Task 1
### Exercise 1 - k-NN Classification
The file: ``task1main/Microchips.py``  
This code implements a k-Nearest Neighbors (k-NN) classifier for a microchip quality classification problem. It predicts whether a microchip passes or fails based on test results using Euclidean distance to determine the k-nearest neighbors. The implementation includes a custom k-NN class, decision boundary visualization, and predictions for unknown microchips. 

### Exercise 2 - k-NN Regression
The file: ``task1main/Polynomial.py``  
This code implements k-Nearest Neighbors (k-NN) regression to model a non-linear polynomial relationship between input and output variables. The dataset is split into training and test sets, and the model predicts values based on the k nearest neighbors' average. The implementation evaluates multiple k values, computes the Mean Squared Error (MSE) for both training and test data, and selects the optimal k based on performance.


## Task 2
This Jupyter Notebook contains a comprehensive Machine Learning task with multiple exercises covering various aspects of data preprocessing, model training, and evaluation. The notebook includes exploratory data analysis (EDA), feature selection, performance comparison of different models, and hands-on implementations of key ML concepts.  
[📘Go to Task 2 Notebook](/task2main/Task2.ipynb)


## Task 3
### Exercise 1 - Fashion MNIST
The file: ``task3main/Fashion_MNIST.py``  
This code implements a Neural Network classifier for the Fashion MNIST dataset, a benchmark dataset for image classification containing 10 categories of clothing items. The model is built using TensorFlow and Keras, consisting of a Flatten layer, a fully connected hidden layer with ReLU activation, and a softmax output layer for multi-class classification. The dataset is normalized for better training performance, and early stopping is used to prevent overfitting. The model is trained on a subset of Fashion MNIST and evaluated on a test set, with accuracy metrics displayed.

### Exercise 2 - Ensemble of Batman Trees
The file: ``task3main/batman_trees.py``  
This code implements an ensemble of decision trees trained on bootstrap samples to improve classification performance. Using the bagging technique, multiple DecisionTreeClassifier models are trained on different subsets of the dataset, and their predictions are aggregated through majority voting to form a robust final prediction. The implementation evaluates both individual tree accuracy and ensemble accuracy, highlighting the advantages of ensemble learning in reducing variance and improving generalization.

### Exercise 3 - Various kernels
The file: ``task3main/various_kernels.py``  
This code implements Support Vector Machine (SVM) classification using different kernel functions: Linear, Radial Basis Function (RBF), and Polynomial kernels. The model is trained on a dataset and optimized using GridSearchCV to find the best hyperparameters for each kernel type. The best-performing models are evaluated on a validation set using accuracy metrics. This exercise demonstrates the impact of kernel selection on SVM performance and highlights the role of hyperparameter tuning in classification tasks. 

### Exercise 4 - One versus all MNIST
This Jupyter Notebook contains Exercise 4, focusing on Machine Learning techniques applied to a dataset. The notebook includes data preprocessing, model training, hyperparameter tuning, and performance evaluation. It explores different approaches to optimize the model and visualize results.   
[📘Go to Exercise 4 Notebook](/task3main/exercise4.ipynb)