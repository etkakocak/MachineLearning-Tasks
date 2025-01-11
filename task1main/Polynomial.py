# Exercise 2: k-NN Regression

import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt("datasets/polynomial200.csv", delimiter=",")

# Divide the dataset into training set and test set (both of size 100)
X_train, X_test = data[:100, 0], data[100:, 0]
y_train, y_test = data[:100, 1], data[100:, 1]

# Plot the training and the test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data')
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test Data')
plt.tight_layout()
plt.show()

# Calculate mean squared error for training data and test data
def knn_regression(k, X_train, y_train, X_test=None, y_test=None):
    y_pred_train = []
    y_pred_test = []
    for x in X_train:
        # k nearest neighbors
        idx = np.argsort(np.abs(X_train - x))[:k]
        y_pred = np.mean(y_train[idx])
        y_pred_train.append(y_pred)
    mse_train = np.mean((y_train - y_pred_train) ** 2)
    if X_test is not None and y_test is not None:
        for x in X_test:
            idx = np.argsort(np.abs(X_train - x))[:k]
            y_pred = np.mean(y_train[idx])
            y_pred_test.append(y_pred)
        mse_test = np.mean((y_test - y_pred_test) ** 2)
        return mse_train, mse_test
    else:
        return mse_train

# k-NN regression for each k value
k_values = [1, 3, 5, 7, 9, 11]
plt.figure(figsize=(12, 8))
best_k = None
min_mse_test = float('inf')
for i, k in enumerate(k_values):
    mse_train, mse_test = knn_regression(k, X_train, y_train, X_test, y_test)
    plt.subplot(2, 3, i+1)
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    X_plot = np.linspace(min(X_train), max(X_train), 100)
    y_plot = []
    for x in X_plot:
        idx = np.argsort(np.abs(X_train - x))[:k]
        y_pred = np.mean(y_train[idx])
        y_plot.append(y_pred)
    plt.plot(X_plot, y_plot, color='green', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'k = {k}, MSE(train) = {mse_train:.2f}')
    plt.legend()
    if mse_test < min_mse_test:
        min_mse_test = mse_test
        best_k = k
plt.tight_layout()
plt.show()

# Calculate MSE training and test errors for each k
print("MSE Errors:")
for k in k_values:
    mse_train, mse_test = knn_regression(k, X_train, y_train, X_test, y_test)
    print(f"k = {k}, MSE(train) = {mse_train:.2f}, MSE(test) = {mse_test:.2f}")

# Choose the best k value based on the MSE test
print(f"\nBest k for regression: {best_k} with a value of {min_mse_test:.2f}")
