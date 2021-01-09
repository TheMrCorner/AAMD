import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.optimize as opt

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def coste(theta, X, Y, reg):
    h = sigmoid(np.matmul(X, theta))
    m = X.shape[0]
    return (-1/m) * (np.dot(Y,np.log(h)) + np.dot((1-Y), np.log(1-h)))+ ((reg/2*m) * np.sum(np.power(theta[1:],2)))

def gradiente(theta, X, Y, reg):
    m = X.shape[0]
    h = sigmoid(np.matmul(X, theta))
    Taux = theta
    Taux[0] = 0
    aux = (1/m) * np.matmul(X.T, (np.ravel(h) - Y))
    return aux + ((reg/m) * Taux)

def h(X, thetas1, thetas2):
    a1 = X
    z2 = np.matmul(thetas1, np.insert(a1,0,1))
    a2 = sigmoid(z2)
    z3 = np.matmul(thetas2, np.insert(a2,0,1))
    a3 = sigmoid(z3)
    return a3

def getEtiqueta(Y, etiqueta):
    y_etiqueta = np.ravel(Y)== etiqueta
    y_etiqueta = y_etiqueta *1
    return y_etiqueta

def evalua(theta, X, Y):
    xs = sigmoid(np.matmul(X, theta))
    xspositivas = np.where(xs >= 0.5)
    xsnegativas = np.where(xs < 0.5)
    xspositivasexample = np.where (Y == 1)
    xsnegativasexample = np.where (Y == 0)

    porcentajePositivas = np.intersect1d(xspositivas, xspositivasexample).shape[0]/xs.shape[0]
    porcentajeNegativas = np.intersect1d(xsnegativas, xsnegativasexample).shape[0]/xs.shape[0]
    print("Total:", porcentajeNegativas + porcentajePositivas)
    return porcentajeNegativas + porcentajePositivas

def oneVsAll(X, Y, num_etiquetas, reg):
    m = X.shape[1]
    theta = np.zeros((num_etiquetas, m))
    y_etiquetas = np.zeros((y.shape[0], num_etiquetas))

    for i in range(num_etiquetas):
        y_etiquetas[:,i] = getEtiqueta(y, i)
    y_etiquetas[:,0] = getEtiqueta(y, 10)

    for i in range(num_etiquetas):
        print("I: ", i)
        result = opt.fmin_tnc(func=coste, x0=theta[i,:], fprime=gradiente, args=(X, y_etiquetas[:,i], reg))
        theta[i, :] = result[0]

    evaluacion = np.zeros(num_etiquetas)
    for i in range(num_etiquetas):
        evaluacion[i] = evalua(theta[i,:], X, y_etiquetas[:,i])
    print("Evaluacion: ", evaluacion)
    print("Evaluacion media: ", evaluacion.mean())
    return 0

data = loadmat("ex3data1.mat")
y = data ['y']
X = data ['X']
thetas = loadmat("ex3weights.mat")
thetas1,thetas2 = thetas["Theta1"], thetas["Theta2"]

#PARTE 1
oneVsAll(X, y, 10, 0.1)

sample = np.random.choice(X.shape[0],10)
plt.imshow(X[sample, :].reshape(-1,20).T)
plt.axis('off')

#PARTE 2
aux = np.zeros(10)
print ("Sample: ", sample)
for i in range(10):
    aux[i] = np.argmax(h(X[sample[i],:],thetas1,thetas2))
print("My guess are: ", (aux)+1)
numAciertos = 0
for i in range(X.shape[0]):
    aux2 = np.argmax(h(X[i,:],thetas1, thetas2))
    if(aux2+1) == y[i]:
        numAciertos +=1
print("Porcentaje de aciertos: ", numAciertos / X.shape[0])
plt.show()
