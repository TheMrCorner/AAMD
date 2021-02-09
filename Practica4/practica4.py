import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import checkNNGradients
import os

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.optimize import minimize

# Save relative path to file, to find the different files easier
relativeDir = os.path.dirname(__file__)

def carga_csv(file_name):
    v = loadmat(file_name)
    return v

def add_ones(val):
    ones = np.ones((val.shape[0], 1), dtype=val.dtype)
    return np.hstack((ones, val))

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    z2 = np.dot(X, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return X, z2, a2, z3, h

def random_thetas(l_in, l_out, range=0.12):
    W = np.zeros((l_out, 1 + l_in))
    W = np.random.rand(l_out, 1 + l_in) * (2 * range) - range

    return W

def unroll_thetas(params, n_entries, n_hidden, n_et):
    theta1 = np.reshape(params[:n_hidden * (n_entries + 1)], (n_hidden, (n_entries + 1)))
    theta2 = np.reshape(params[n_hidden * (n_entries + 1):], (n_et, (n_hidden + 1)))

    return theta1, theta2

def CostNN(params, n_entries, n_hidden, n_et, X, Y, reg):
    theta1, theta2 = unroll_thetas(params, n_entries, n_hidden, n_et)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    j = 0
    for i in range(len(X)):
        j += (-1 / (len(X))) * (np.dot(Y[i], np.log(h[i])) + np.dot((1 - Y[i]), np.log(1 - h[i])))

    j += (reg / (2*len(X))) + ((np.sum(np.square(theta1[:,1:]))) + (np.sum(np.square(theta2[:,1:]))))

    return j

def backprop(params, n_entries, n_hidden, n_et, X, Y, reg):
    m = X.shape[0]
    X = add_ones(X)

    theta1, theta2 = unroll_thetas(params, n_entries, n_hidden, n_et)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    cost = CostNN(params, n_entries, n_hidden, n_et, X, Y, reg)

    delta1, delta2 = 0, 0

    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]
        ht = h[t, :]
        yt = Y[t] 

        d3t = ht -yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = delta1 / m

    delta2 = delta2 / m

    # Delta's gradients
    delta1[:,1:] = delta1[:,1:] + (reg * theta1[:,1:]) / m
    delta2[:,1:] = delta2[:,1:] + (reg * theta2[:,1:]) / m

    # Add gradients
    total_g = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return cost, total_g



def main():
    # Test case number:
    input_layer_size = 400

    # Hidden layer units
    hidden_layer_size = 25

    # Etiquetes
    n_et = 10

    # Load data
    vals = carga_csv(os.path.join(relativeDir, "ex4data1.mat"))
    thetas = carga_csv(os.path.join(relativeDir, "ex4weights.mat"))
    Y = vals['y']
    X = vals['X']

    Y.reshape(Y.shape[0], 1)
    m = X.shape[0]

    Y = (Y - 1)
    y_onepoint = np.zeros((m, n_et))

    for i in range(m):
        y_onepoint[i][Y[i]] = 1
    
    theta1 = random_thetas(input_layer_size, hidden_layer_size)
    theta2 = random_thetas(hidden_layer_size, n_et)

    Thetas = [theta1, theta2]

    unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
    nn_params = np.concatenate(unrolled_Thetas)

    # Search using scipy
    result = minimize(fun=backprop, x0=nn_params, args=(input_layer_size, hidden_layer_size, n_et, X, y_onepoint, 1), method='CG', jac=True, options={'maxiter': 70})
    theta1, theta2 = unroll_thetas(result.x, input_layer_size, hidden_layer_size, n_et)

    # Create NN with optimized Thetas
    X = add_ones(X)
    h = forward_propagate(X, theta1, theta2)[4]
    correct = 0
    wrong = 0

    # Check answer with real one
    for i in range(len(X)):
        maxIndex = np.argmax(h[i])
        if(maxIndex == Y[i]):
            correct += 1
        else:
            wrong += 1
    
    print("Hit: ", correct)
    print("Miss: ", wrong)
    print("Accuracy: ", format((correct / (correct + wrong)) * 100, '.2f'), "%")

main()
    