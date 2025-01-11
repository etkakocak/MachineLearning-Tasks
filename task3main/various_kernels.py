# Exercise 3: Various kernels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load the datasets
train_data = pd.read_csv('datasets/dist.csv', sep=';', header=None)
validation_data = pd.read_csv('datasets/dist_val.csv', sep=';', header=None)

X_train = train_data.iloc[:, :2].values
y_train = train_data.iloc[:, 2].values

X_val = validation_data.iloc[:, :2].values
y_val = validation_data.iloc[:, 2].values

# Define the parameters
param_grid_linear = {'C': [0.1, 1, 10]}
param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
param_grid_poly = {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': [0.001, 0.01, 0.1]}

# Grid search for linear kernel
grid_linear = GridSearchCV(SVC(kernel='linear'), param_grid_linear, cv=3)
grid_linear.fit(X_train, y_train)
best_linear = grid_linear.best_estimator_

# Grid search for RBF kernel
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid_rbf, cv=3)
grid_rbf.fit(X_train, y_train)
best_rbf = grid_rbf.best_estimator_

# Grid search for polynomial kernel
grid_poly = GridSearchCV(SVC(kernel='poly'), param_grid_poly, cv=3)
grid_poly.fit(X_train, y_train)
best_poly = grid_poly.best_estimator_

# Evaluate the best models
accuracy_linear = accuracy_score(y_val, best_linear.predict(X_val))
accuracy_rbf = accuracy_score(y_val, best_rbf.predict(X_val))
accuracy_poly = accuracy_score(y_val, best_poly.predict(X_val))

print(f"Best Linear SVM accuracy: {accuracy_linear:.2f}")
print(f"Best RBF SVM accuracy: {accuracy_rbf:.2f}")
print(f"Best Polynomial SVM accuracy: {accuracy_poly:.2f}")

# Plot decision boundaries and true decision boundary
def plot_decision_boundary(model, X, y, title):
    h = .02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='.')
    x_range = np.linspace(x_min, x_max, 1000)
    y_true_boundary = np.piecewise(x_range,
                                   [x_range > 3.94, x_range <= 3.94],
                                   [lambda x: 0.5 * (18 - 2 * x - np.sqrt(724 + 256 * x - 16 * x ** 2)),
                                    lambda x: 0.071 * (174 - 22 * x - np.sqrt(23123 - 6144 * x + 288 * x ** 2))])
    plt.plot(x_range, y_true_boundary, 'r-', label='True decision boundary')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

plot_decision_boundary(best_linear, X_train, y_train, 'Best Linear SVM')
plot_decision_boundary(best_rbf, X_train, y_train, 'Best RBF SVM')
plot_decision_boundary(best_poly, X_train, y_train, 'Best Polynomial SVM')
