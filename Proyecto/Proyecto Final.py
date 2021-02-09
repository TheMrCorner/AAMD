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
from scipy.optimize import minimize
import sklearn
from sklearn.svm import SVC

#VARIABLES CONSTANTES
#Regresion Logistica
numPruebas = 5
testSizeRL = 0.8
#Red Neuronal
input_layer_size = 31
hidden_layer_size = 25
num_etiquetas = 2
lamb = 1000
#Support Vector Machine
numIteraciones = 1
_C = 100
Sigma = 100
testSizeSVM = 0.2

#Leemos y preparamos el dataset para ser utilizado
def dataSetup():
    data = pd.read_csv(r"F:\UniShit\Repos\AAMD\Proyecto\data.csv")
    data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    Y = data.diagnosis.values
    data.drop(['diagnosis'], axis=1)
    X = data.values
    return X, Y

def main():
    #Preparamos los datos
    X , Y = dataSetup()

    #Usamos las 3 tecnicas de ML
    precisionRL = RegresionLogistica(X, Y)
    aciertos, fallos, falsosPositivos, falsosNegativos = RedNeuronal(X, Y)
    precisionSVM = SupportVectorMachine(X, Y)

    #Imprimimos los resultados
    print("//////////////////////////////RESULTADOS//////////////////////////////////")
    print("Regresion Logistica")
    print("Numero de pruebas: ", numPruebas)
    print("Precision: ",format((precisionRL / numPruebas)* 100, '.2f' ),"%")
    print()
    print("Red Neuronal")
    print("Aciertos: ", aciertos)
    print("Fallos: ", fallos)
    print("Falsos Positivos: ", falsosPositivos)
    print("Falsos Negativos: ", falsosNegativos)
    print("Precision: ",format((aciertos / (aciertos+fallos))*100, '.2f' ),"%")
    print()
    print("Support Vector Machine")
    print("Numero de iteraciones: ", numIteraciones)
    print("Precision: ",format((precisionSVM / numIteraciones)* 100, '.2f' ),"%")
    print("//////////////////////////////////////////////////////////////////////////")


def RegresionLogistica(X, Y):
    accuracy = 0.0
    for i in range(numPruebas):
        X = add_ones(X)
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = testSizeRL,shuffle = True)
        theta = np.zeros((X.shape[1],1))
        result = opt.fmin_tnc(func=coste,x0=theta ,fprime=gradienteRL,args =(x_train, y_train))
        theta = result[0]
        
        accuracy += evalua(theta,x_test,y_test)                            
    print("Avg. Accuracy: ",format((accuracy / numPruebas)* 100, '.2f' ),"%")
    return accuracy

def RedNeuronal(X, Y):
    m = X.shape[0]
    y_onehot = np.zeros((m,num_etiquetas))
    for i in range(m):
        y_onehot[i][int(Y[i])]= 1

    thetas1 = randomizeThetas(input_layer_size,hidden_layer_size)
    thetas2 = randomizeThetas(hidden_layer_size,num_etiquetas)

    result = minimize(fun=backprop, x0=np.append(thetas1,thetas2), args=(input_layer_size, hidden_layer_size,
    num_etiquetas, X, y_onehot,lamb ), method = 'TNC', jac = True, options = {'maxiter': 500, 'disp':True})
    theta1,theta2 = unroll_thetas(result.x,input_layer_size,hidden_layer_size,num_etiquetas)
    X = add_ones(X)
    h = forward_propagate(X, theta1, theta2)[4]
    correct = 0
    wrong = 0
    falsePositive = 0
    falseNegative = 0
    for i in range(len(X)):
        maxIndex = np.argmax(h[i])
        if(maxIndex == Y[i]):
            correct += 1
        else:
            wrong += 1
            if(Y[i] == 1):
                falseNegative +=1
            else:
                falsePositive += 1

    return correct, wrong, falsePositive, falseNegative


def SupportVectorMachine(X, Y):
    precision = 0
    for x in range(0, numIteraciones):
        X_new, Xval, y_new, Yval = train_test_split(X, Y, test_size = 0.2,shuffle = True)
        #USAR ESTO PARA KERNEL LINEAL
        #svm = SVC(kernel = 'linear', C = valorC)
        #USAR ESTO PARA KERNEL GAUSSIANO
        svm = SVC(kernel = 'rbf', C = _C, gamma = 1 / (2 * Sigma **2))
        svm.fit(X_new, y_new)
        precision += test(svm, Xval, Yval)
    return precision


#Funciones compartidas
def sigmoid(Z):
    return 1.0/( 1.0 + np.exp(-Z))

def add_ones(valores):
    unos = np.ones((valores.shape[0], 1), dtype=valores.dtype)
    return np.hstack((unos , valores))
############################################################

#Funciones para regresion logística
def coste(thetha,X,Y):
    h = sigmoid(np.matmul(X,thetha))
    m = X.shape[0]
    return (-1/m) * (np.dot(Y, np.log(h)) + np.dot((1-Y), np.log(1 - h)))


def gradienteRL(thetha,X,Y):
    m = X.shape[0]
    h = sigmoid(np.matmul(X,thetha))
    return (1/m) * np.matmul( X.T, (np.ravel(h) - Y))

def evalua(theta,X,Y):
    xs = sigmoid(np.matmul(X,theta))
    xspositivas = np.where(xs >= 0.5)
    xsnegativas = np.where(xs < 0.5)
    xspositivasejemplo = np.where (Y == 1 )
    xsnegativasejemplo = np.where (Y == 0 )
    porcentajepos = np.intersect1d(xspositivas,xspositivasejemplo).shape[0]/xs.shape[0]
    porcentajeneg = np.intersect1d(xsnegativas,xsnegativasejemplo).shape[0]/xs.shape[0]
    print("Total:", porcentajeneg + porcentajepos)
    return porcentajepos + porcentajeneg
#################################################################################

#Funciones para Red Neuronal
def backprop(params_rn,num_entradas,num_ocultas,num_etiquetas,X,y, reg):
    #Luego "deserializamos" los parametros
    m = X.shape[0]
    X = add_ones(X)
    theta1,theta2 = unroll_thetas(params_rn, num_entradas, num_ocultas, num_etiquetas)
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
    cost = CostNN(params_rn,num_entradas, num_ocultas, num_etiquetas, X, y, reg)
    #Claculo de deltas
    delta1 = 0
    delta2 = 0
    for t in range(m):
        a1t = a1[t, :] # (1, 401)
        a2t = a2[t, :] # (1, 26)
        ht = h[t, :] # (1, 10)
        yt = y[t] # (1, 10)
        d3t = ht - yt # (1, 10)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t)) # (1, 26)
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])
    delta1 = delta1/m
    delta2 = delta2/m
    #Gradiente de cada delta
    delta1[:,1:] = delta1[:,1:] + (reg *theta1[:,1:]) / m
    delta2[:,1:] = delta2[:,1:] + (reg *theta2[:,1:]) / m
    #Juntamos los gradientes
    gradiente = np.concatenate((np.ravel(delta1),np.ravel(delta2)))
    return cost, gradiente

def CostNN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    theta1,theta2 = unroll_thetas(params_rn, num_entradas, num_ocultas, num_etiquetas)
    a1,z2,a2,z3,h = forward_propagate(X,theta1,theta2)
    J = 0
    for i in range(len(X)):
        J += (-1/(len(X)))*(np.dot(Y[i],np.log(h[i]))+np.dot((1-Y[i]),np.log(1-h[i])))
    J += (reg/ (2*len(X))) * ( ( np.sum(np.square(theta1[:,1:]))) + (np.sum(np.square(theta2[:,1:]))) )
    return J

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    z2 = np.dot(X, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return X, z2, a2, z3, h

def unroll_thetas(params_rn,num_entradas,num_ocultas,num_etiquetas):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas +1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas +1)))
    return theta1,theta2

# Inicializa un vector de thetas para una capa con L_in entradas y L_out salidas entre un rango range
def randomizeThetas(L_in, L_out, range = 0.12):
    # Inicializamos los vectores de theta
    W = np.zeros((L_out, 1 + L_in))
    # Randomizamos el vector entre -rango y rango
    W = np.random.rand(L_out, 1 + L_in) * (2 * range) - range
    return W

########################################################################################

#Funciones para SVM
def test(svm, X, Y):
    prediction = svm.predict(X)
    accuracy = np.mean((prediction == Y).astype(int))
    return accuracy
###############################################################################
main()