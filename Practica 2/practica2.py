import numpy as np
import matplotlib.pyplot as plt
import time as t 
import math
from pandas.io.parsers import read_csv

#funcion para cagar ficheros csv
def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def coste_sigmoide(z):
    return 1/(1+math.exp(-z))
#Esto genera la version de coste_sigmoide que adfmite vectores y matrices como entrada
vec_costeSig = np.vectorize(coste_sigmoide)
#la prueba
#A = [[1, 4, 5], 
#    [-5, 8, 9]]
#print(vec_costeSig(A))

datos = carga_csv("C:\hlocal\AAMD\Practica 2\ex2data1.csv")
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)   
pos = np.where (Y == 1)
plt.figure()
plt.scatter(X[pos ,0] ,X[pos ,1], marker='+', c='k')
pos = np.where (Y != 1)
plt.scatter(X[pos ,0] ,X[pos ,1], c='y')
plt.show()
