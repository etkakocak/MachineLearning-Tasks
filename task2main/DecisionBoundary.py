import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(X1, X2, y, model):
    # Generate a mesh grid to plot the decision boundary
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Flatten the grid so the values match the expected input for the model
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the labels for the whole grid
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 3))
    plt.scatter(X1, X2, c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
