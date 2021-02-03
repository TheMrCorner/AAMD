import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
import math
import scipy.optimize as opt

def main():
    data = pd.read_csv(r"F:\UniShit\Repos\AAMD\Proyecto\data.csv")
    data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    valores = data.values
    Y = data.diagnosis.values
    data.drop(['diagnosis'], axis=1)
    X = data.values
    tags = valores[0,:]
    valores = np.delete(valores,0,0).astype(float)
    print(tags)
    print(valores[:2,:])
    accuracy = 0.0
    NUM_PRUEBAS = 1
    for i in range(NUM_PRUEBAS):
        np.random.shuffle(valores)
        #Vamos a separar los ejemplos en 80% para entrenar y un 20% para evaluar
        X = add_ones(X)
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.9,shuffle = True)
        theta = np.zeros((X.shape[1],1))
        result = opt.fmin_tnc(func=coste,x0=theta ,fprime=gradiente,args =(x_train, y_train))
        theta = result[0]
        
        accuracy += evalua(theta,x_test,y_test)
    print("Avg. Accuracy: ",format((accuracy / NUM_PRUEBAS)* 100, '.2f' ),"%")
    #print(X)
def H(X,thetha):
    return sigmoid(np.matmul(X,thetha))

def coste(thetha,X,Y):
    h = sigmoid(np.matmul(X,thetha))
    m = X.shape[0]
    return (-1/m) * (np.dot(Y, np.log(h)) + np.dot((1-Y), np.log(1 - h)))

def sigmoid(Z):
    return 1.0/( 1.0 + np.exp(-Z))

def gradiente(thetha,X,Y):
    m = X.shape[0]
    h = sigmoid(np.matmul(X,thetha))
    return (1/m) * np.matmul( X.T, (np.ravel(h) - Y)) # Las dimensiones de H son (m,1) y las de Y (m,),

def evalua(theta,X,Y):
    xs = sigmoid(np.matmul(X,theta))
    xspositivas = np.where(xs >= 0.5)
    xsnegativas = np.where(xs < 0.5)
    xspositivasejemplo = np.where (Y == 1 )
    xsnegativasejemplo = np.where (Y == 0 )
    #Printea los casos en los que la funcion sigmoide con las thetas de ejemplo indica que va a tener un ataque
    #print("Acertadas en el sigmoide: ", xspositivas)
    #print("Positivas en los ejemplos: ", xspositivasejemplo)
    porcentajepos = np.intersect1d(xspositivas,xspositivasejemplo).shape[0]/xs.shape[0]
    porcentajeneg = np.intersect1d(xsnegativas,xsnegativasejemplo).shape[0]/xs.shape[0]
    print("Total:", porcentajeneg + porcentajepos)
    return porcentajepos + porcentajeneg

def add_ones(valores):
    unos = np.ones((valores.shape[0], 1), dtype=valores.dtype)
    return np.hstack((unos , valores))

main()
