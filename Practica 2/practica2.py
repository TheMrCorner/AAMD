import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
import time as t 
import math
import scipy.optimize as opt
from pandas.io.parsers import read_csv
import os

# Save relative path to file, to find the different files easier
relativeDir = os.path.dirname(__file__)

def carga_csv(file_name):
    v = read_csv(file_name, header=None).values
    return v.astype(float)

def H(X, theta):
    return sigmoid(np.matmul(X, theta))

def cost(theta, X, Y):
    m = X.shape[0]
    h = sigmoid(np.matmul(X, theta))
    return (-1/m) * np.dot(Y, np.log(h)) + np.dot((1-Y), np.log(1 - h))

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def gradient(theta, X, Y):
    m = X.shape[0]
    h = sigmoid(np.matmul(X, theta))
    return (1/m) * np.matmul(X.T, (np.ravel(h) - Y))

def check(theta, X, Y):
    xs = sigmoid(np.matmul(X, theta))

    xsPos = np.where(xs >= 0.5)
    xsNeg = np.where(xs < 0.5)

    xsExp = np.where(Y == 1)
    xsExn = np.where(Y == 0)

    perPos = np.intersect1d(xsPos, xsExp).shape[0]/xs.shape[0]
    perNeg = np.intersect1d(xsNeg, xsExn).shape[0]/xs.shape[0]

    total = perNeg + perPos

    print("Suma de porcentajes: ", total)
    return total

def add_ones(val):
    ones = np.ones((val.shape[0], 1), dtype=val.dtype)
    return np.hstack((ones, val))

def pinta_frontera_recta(X, Y, theta):
    plt.figure()

    p = np.where(Y == 1)
    p2 = np.where(Y == 0)

    plt.scatter(X[p, 0], X[p, 1], marker='x', c='k')
    plt.scatter(X[p2, 0], X[p2, 1], marker='.', c='r')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

def main():
    # Cargamos los datos de los archivos
    vals = carga_csv(os.path.join(relativeDir, "ex2data1.csv"))

    N = vals.shape[0]

    acc = 0.0
    nTests = 1000
    for i in range(nTests):
        np.random.shuffle(vals)

        # Separate 80% for tests and 20% for evaluate
        X = vals[:,:-1]
        X = add_ones(X)
        Y = vals[:,-1]
        x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size=0.7, shuffle=True)
        theta = np.zeros((X.shape[1], 1))

        result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x_tr, y_tr))

        theta = result[0]

        grad = (gradient(theta, x_tr, y_tr))
        pinta_frontera_recta(vals[:,:-1], Y, theta)

        acc += check(theta, x_ts, y_ts)

    plt.show()

main()