import numpy as np
import matplotlib.pyplot as plt
import time as t 
from pandas.io.parsers import read_csv

#funcion para cagar ficheros csv
def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)
def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

#inicializar theta0 a 0, theta1 a 0, bucle 1500 iteraciones-> aplicar la formula que calcula el nuevo theta0, theta1 a partir de los anteriores
def descenso_gradiente(X, Y, alpha):
    theta0 = 0
    theta1 = 1
    for n in range(1500):
        sumatorioT0 = 0
        sumatorioT1 = 0
        for i in range(len(X)):
            H = theta0 + theta1* X[i,1]
            sumatorioT0 += (H - Y[i])
            sumatorioT1 += (H - Y[i])*X[i,1]
        theta0 = theta0 - alpha*(sumatorioT0/len(X))
        theta1 = theta1 - alpha*(sumatorioT1/len(X))
    theta = [theta0, theta1]
    print(theta)
    costes = coste(X, Y, theta)
    return theta, coste

datos = carga_csv("C:\hlocal\AAMD\Practica 1\datos.csv")
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)         # (97,)
plt.figure()
plt.scatter(X ,Y, marker='x', c='r')
m = np.shape(X)[0]
# añadimos una columna de 1's a la X
# esto es para que se pueda multiplicar x por el vector theta
X = np.hstack([np.ones([m, 1]), X])
alpha = 0.01
Thetas, costes = descenso_gradiente(X, Y, alpha)
min_x = np.min([n[1] for n in X])
max_x = np.max(X)
min_y = Thetas[0] + Thetas[1] * min_x
max_y = Thetas[0] + Thetas[1] * max_x
plt.plot([min_x, max_x], [min_y, max_y])
plt.show()