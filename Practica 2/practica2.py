import numpy as np
import matplotlib.pyplot as plt
import time as t 
import math
import scipy.optimize as opt
from pandas.io.parsers import read_csv

#funcion para cagar ficheros csv
def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def coste_sigmoide(z):
    return 1/(1+math.exp(-z))

#función de coste
def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    print(cost)
    return cost

def gradient(theta, XX, Y):
    H = sigmoid(np.matmul(XX, theta))
    grad = (1 / len(Y)) * np.matmul(XX.T, H - Y)
    return grad


def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    #plt.savefig("frontera.pdf")
    #plt.close()


#Esto genera la version de coste_sigmoide que admite vectores y matrices como entrada
sigmoid = np.vectorize(coste_sigmoide)
#la prueba
#A = [[1, 4, 5], 
#    [-5, 8, 9]]
#print(vec_costeSig(A))

datos = carga_csv("C:\hlocal\AAMD\Practica 2\ex2data1.csv")
#datos = carga_csv(r"C:\Users\Daniel Alvarez\Desktop\UniShit\Repos\AAMD\Practica 2\ex2data1.csv")
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)   
#Prueba para las funciones
#print(cost(np.zeros(2), X, Y))
#print(gradient(np.zeros(2), X, Y))
pos = np.where (Y == 1)
plt.figure()
plt.scatter(X[pos ,0] ,X[pos ,1], marker='+', c='k')
pos = np.where (Y != 1)
plt.scatter(X[pos ,0] ,X[pos ,1], c='y')
theta = [0,0]
#a partir de aqui los resultados no son correctos, creo que es porque theta no tienen que ser [0,0]
result = opt.fmin_tnc(func=cost , x0=theta , fprime=gradient, args =(X, Y))
theta_opt = result[0]
pinta_frontera_recta(X, Y, theta_opt)
plt.show()
