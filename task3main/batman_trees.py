# Exercise 2: Ensemble of Batman Trees

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import pandas as pd

# Load the dataset
df_bm = pd.read_csv('datasets/bm.csv')

X = df_bm.iloc[:, :2].values
y = df_bm.iloc[:, 2].values

# Split the data
train_size = 9000
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Create bootstrap samples
n_samples = 5000
n_trees = 100
rng = np.random.default_rng()

bootstrap_samples_X = np.zeros((n_trees, n_samples, X_train.shape[1]))
bootstrap_samples_y = np.zeros((n_trees, n_samples), dtype=int)

for i in range(n_trees):
    indices = rng.choice(X_train.shape[0], n_samples, replace=True)
    bootstrap_samples_X[i] = X_train[indices]
    bootstrap_samples_y[i] = y_train[indices]

# Train decision trees on bootstrap samples
trees = []
for i in range(n_trees):
    tree = DecisionTreeClassifier()
    tree.fit(bootstrap_samples_X[i], bootstrap_samples_y[i])
    trees.append(tree)

# Evaluate the ensemble
predictions = np.zeros((n_trees, X_test.shape[0]), dtype=int)
for i, tree in enumerate(trees):
    predictions[i] = tree.predict(X_test)

# Majority vote
ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

# Individual tree accuracy
individual_accuracies = [accuracy_score(y_test, pred) for pred in predictions]
average_individual_accuracy = np.mean(individual_accuracies)

print(f"Ensemble Test Accuracy: {ensemble_accuracy:.2f}")
print(f"Average Individual Test Accuracy: {average_individual_accuracy:.2f}")

# Plotting decision boundaries
def plot_decision_boundary(clf, X, y, ax=None, title='Decision Boundary'):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='.')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)

# Plot individual trees
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i, tree in enumerate(trees[:99]):
    ax = axes[i // 10, i % 10]
    plot_decision_boundary(tree, X_train, y_train, ax=ax, title=f'Tree {i+1}')
axes[9, 9].axis('off')  

# Plot ensemble data
ensemble_clf = VotingClassifier(estimators=[(f'tree_{i}', tree) for i, tree in enumerate(trees)], voting='hard')
ensemble_clf.fit(X_train, y_train)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_decision_boundary(ensemble_clf, X_train, y_train, ax=ax, title='Ensemble')
plt.tight_layout()
plt.show()

# Short comment on the results
comment = """
The ensemble of decision trees achieved a higher accuracy compared to the average individual decision tree. 
I think this was to be expected due to the greater accuracy created by the ensemble method. 
The ensemble method gives the most accurate result by helping correct errors in individual trees. 
We can count this as an advantage in using ensemble methods. 
However, a disadvantage of ensemble methods will be the increased computational time and complexity of the model.
"""
print(comment)
