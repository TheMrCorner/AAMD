# Cálculo de la integral por el método de Monte Carlo
import numpy as np
import matplotlib.pyplot as plt
import time as t 

# FUncion de prueba
def funcion(x):
    return -(x * x) + 50

# Comprueba que esta debajo de la curva
def esta_debajo(fun, x, y):
    nY = fun(x)
    return y < nY 

def gen_puntos(min_n, max_n, num_puntos):
    a = np.zeros(num_puntos)
    for i in range(num_puntos):
        a[i] = np.random.uniform(min_n, max_n, 1)

    return a

def draw_graphic(x, y, ry):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(x, y, color='blue')
    plt.scatter(x[np.where(ry > y)], ry[np.where(ry > y)], marker='x', color='red')
    plt.scatter(x[(ry < y)], ry[(ry < y)], marker='o', color='green')
    plt.show()

def guillotina(fun, a, b, num_puntos=1000):
    # Puntos x de la curva principal
    x = np.linspace(a, b, num_puntos)
    y = fun(x)

    # Puntos random, coordenada X
    ry = gen_puntos(min(y), max(y), num_puntos)

    draw_graphic(x, y, ry)

    return y, ry

def integraL(fun, a, b, y, ry):

    count = 0
    #comienza el cronometro
    inicio = t.process_time()
    for i in range(len(ry)):
        if ry[i] < y[i]:
            count += 1

    #print del resultado de la integral
    print("Resultado integral: " + str(count/len(ry) * (b - a) * max(y)))

    #detiene el cronometro
    fin = t.process_time()

    print("Total de trues (L): " + str(count))

    #tiempo que ha tardado
    tiempo = 1000 * (fin- inicio)
    print("Tiempo que ha tomado la funcion lenta: " + str(tiempo))
    return tiempo

def integraR(fun, a, b, y, ry):

    #comienza el cronometro
    inicio = t.process_time()
    # Método rápido
    res = np.sum(ry < y)

    print("Resultado integral: " + str(res/len(ry) * (b - a) * max(y)))

    print("Total de trues (R): " + str(res))

    #detiene el cronometro
    fin = t.process_time()
    
    #tiempo que ha tardado
    tiempo = 1000 * (fin- inicio)
    print("Tiempo que ha tomado la funcion rápida: " + str(tiempo))
    return tiempo

#crea una tuple de cantidades de puntos con los que va a probar
sizes = np.linspace(1000, 1000000, 5)
tiempo_bucle = []
tiempo_numpy = []
#calcula el tiempo de los algoritmos para todas las cantidades de puntos
for size in sizes:
    y, ry = guillotina(funcion, 2, 3000, int(size))
    tiempo_bucle += [integraL(funcion, 3000, 8, y, ry)]
    tiempo_numpy += [integraR(funcion, 3000, 8, y, ry)]

#dibuja el resultado
plt.figure()
plt.scatter(sizes, tiempo_bucle, c='red', label='bucle')
plt.scatter(sizes, tiempo_numpy, c='blue', label='vector')
plt.legend()
plt.show()

#hay que usar sum(y < f(x)) para la versión vectorizada

# Función que va a hacer la integral
