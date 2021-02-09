import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from sklearn.svm import SVC

# Save relative path to file, to find the different files easier
relativeDir = os.path.dirname(__file__)

def drawFunctionPoints(X, y):
    pos1 = np.where(y == 0)
    plt.scatter(X[pos1, 0], X[pos1, 1], marker='.', c='red')
    pos2 = np.where(y == 1)
    plt.scatter(X[pos2, 0], X[pos2, 1], marker='+', c='blue')

def main():
    # Load data
    data = loadmat(os.path.join(relativeDir, "ex6data1.mat"))

    y = data['y']
    X = data['X']

    C = 100
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)

    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:,1].min(), X[:,1].max(), 100)

    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    # Draw
    plt.contour(x1, x2, yp)
    drawFunctionPoints(X, y)
    plt.show()

main()
