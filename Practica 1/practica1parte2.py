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

def normalize(X):
    mean = np.mean(X, 0)
    deviation = np.std(X, 0)
    normalized = (X-mean)/deviation
    return (normalized, mean, deviation)

def draw_3D(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap = cm.jet, linewidth =0, antialiased = False)


def draw_2D(X, Y, a, b):
    plt.figure()
    plt.scatter(X ,Y, marker='x', c='r')
    plt.plot(a, b)

def draw_cont(Theta0, Theta1, Coste, min):
    fig = plt.figure()
    plt.contour(Theta0, Theta1, Coste, np.logspace(-4, 6, 40), cmap = cm.jet)
    plt.scatter(min[0], min[1], marker='x', c='r')

#inicializar theta0 a 0, theta1 a 0, bucle 1500 iteraciones-> aplicar la formula que calcula el nuevo theta0, 
# theta1 a partir de los anteriores
def descenso_gradiente(X, Y,b, alpha):
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

datos = carga_csv("datos2.csv")
# Data loading
N = datos.shape[0]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
col = datos.shape[1]-1
X = datos[:,:col]
Y = datos[:,col:]
normX, mu, sigma = normalize(X)

ones = np.ones((normX.shape[0], 1), dtype=normX.dtype)
normX = np.hstack((ones, normX))
ones = np.ones((X.shape[0], 1), dtype=X.dtype)
X = np.hstack((ones, X))

alpha = 0.01
Thetas, costes = descenso_gradiente(normX, X, Y, alpha)
n = [n[1] for n in X]
min_x = np.min(n)
max_x = np.max(X)
min_y = Thetas[0] + Thetas[1] * min_x
max_y = Thetas[0] + Thetas[1] * max_x
draw_2D(X, Y, [min_x, max_x], [min_y, max_y])

t0_range = [-10,10]
t1_range = [-1,4]
X, Y, Z = make_data(t0_range, t1_range,X, Y)
draw_3D(X, Y, Z)
draw_cont(X, Y, Z, [min_x, min_y])

#Ecuacion normal
xt = X.transpose()
aux = np.dot(xt, X)
aux = np.linalg.inv(aux)
aux = np.dot(aux, xt)
theta = np.dot(aux, Y)
