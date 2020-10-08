import numpy as np
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
    for i in range(num_puntos):
        if ry[i] < y[i]:
            a += 1

    print(a)
    print(t.process_time())
    #a = gen_puntos(num_puntos)

def integraR(fun, a, b, num_puntos=1000):
    # Puntos x de la curva principal
    x = np.linspace(a, b, num_puntos)
    y = funcion(x)
    m = max(y)

    # Puntos random, coordenada X
    #rx = gen_puntos(a, b, num_puntos)
    ry = gen_puntos(0, max(y), num_puntos)

    # Método rápido
    res = np.sum(ry < y)
    print(res)
    print(t.process_time())

    #a = gen_puntos(num_puntos)


integraL(funcion, 2, 8, 10000000)
t.perf_counter()
integraR(funcion, 2, 8, 10000000)
t.perf_counter()


#hay que usar sum(y < f(x)) para la versión vectorizada



# Función que va a hacer la integral