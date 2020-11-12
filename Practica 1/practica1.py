import numpy as np
import matplotlib.pyplot as plt
import time as t 
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
    return theta, costes

def make_data(t0_range, t1_range, X, Y):
    Theta0 = []
    Theta1 = []
    Coste = []
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])
    return [Theta0, Theta1, Coste]

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
n = [n[1] for n in X]
min_x = np.min(n)
max_x = np.max(X)
min_y = Thetas[0] + Thetas[1] * min_x
max_y = Thetas[0] + Thetas[1] * max_x
plt.plot([min_x, max_x], [min_y, max_y])

t0_range = [-10,10]
t1_range = [-1,4]
X, Y, Z = make_data(t0_range, t1_range,X, Y)
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(X, Y, Z, cmap = cm.jet, linewidth =0, antialiased = False)
plt.show()