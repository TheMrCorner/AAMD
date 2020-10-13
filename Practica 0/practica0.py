import numpy as np
import matplotlib.pyplot as plt
import time as t 


def funcion(x):
    return x + 1

# Comprueba que esta debajo de la curva
def esta_debajo(fun, x, y):
    nY = fun(x)
    return y < nY 

def gen_puntos(min_n, max_n, num_puntos):
    a = np.zeros(num_puntos)
    for i in range(num_puntos):
        a[i] = np.random.uniform(min_n, max_n, 1)

    return a

def integraL(fun, a, b, num_puntos=1000):
    # Puntos x de la curva principal
    x = np.linspace(a, b, num_puntos)
    y = funcion(x)
    m = max(y)

    # Puntos random, coordenada X
    #rx = gen_puntos(a, b, num_puntos)
    ry = gen_puntos(0, max(y), num_puntos)

    a = 0
    #comienza el cronometro
    inicio = t.process_time()
    for i in range(num_puntos):
        if ry[i] < y[i]:
            a += 1
    #detiene el cronometro
    fin = t.process_time()
    print(a)
    #tiempo que ha tardado
    tiempo = 1000 * (fin- inicio)
    print(tiempo)
    return tiempo
    #a = gen_puntos(num_puntos)

def integraR(fun, a, b, num_puntos=1000):
    # Puntos x de la curva principal
    x = np.linspace(a, b, num_puntos)
    y = funcion(x)
    m = max(y)

    # Puntos random, coordenada X
    #rx = gen_puntos(a, b, num_puntos)
    ry = gen_puntos(0, max(y), num_puntos)
    #comienza el cronometro
    inicio = t.process_time()
    # Método rápido
    res = np.sum(ry < y)
    #detiene el cronometro
    fin = t.process_time()
    print(res)
    #tiempo que ha tardado
    tiempo = 1000 * (fin- inicio)
    print(tiempo)
    return tiempo
    #a = gen_puntos(num_puntos)

#crea una tuple de cantidades de puntos con los que va a probar
sizes = np.linspace(1000, 1000000, 5)
tiempo_bucle = []
tiempo_numpy = []
#calcula el tiempo de los algoritmos para todas las cantidades de puntos
for size in sizes:
    tiempo_bucle += [integraL(funcion, 2, 8, int(size))]
    tiempo_numpy += [integraR(funcion, 2, 8, int(size))]

#dibuja el resultado
plt.figure()
plt.scatter(sizes, tiempo_bucle, c='red', label='bucle')
plt.scatter(sizes, tiempo_numpy, c='blue', label='vector')
plt.legend()
plt.show()

#hay que usar sum(y < f(x)) para la versión vectorizada



# Función que va a hacer la integral