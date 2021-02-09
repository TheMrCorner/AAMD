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
    input_layer_size = 31
    #Las unidades de la capa oculta
    hidden_layer_size = 25
    #Las etiquetas de la salida
    num_etiquetas = 2
    #Y.reshape(Y.shape[0],1)
    m = X.shape[0]
    y_onehot = np.zeros((m,num_etiquetas))
    for i in range(m):
        y_onehot[i][int(Y[i])]= 1

    #Sacamos las matrices de thetas del archivo
    #thetas1,thetas2 = thetas["Theta1"], thetas["Theta2"]
    thetas1 = randomizeThetas(input_layer_size,hidden_layer_size)
    thetas2 = randomizeThetas(hidden_layer_size,num_etiquetas)
    Thetas = [thetas1,thetas2]
    #Unrolleamos los parametros y los juntamos en el mismo
    unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
    nn_params = np.concatenate(unrolled_Thetas)
    print("Shape of Theta1: ", thetas1.shape)
    print("Shape of Theta2: ", thetas2.shape)
    print("Shape of nn_params: ", nn_params.shape)

    #EVALUACION
    #Buscamos mediante scipy las thetas optimas
    result = minimize(fun=backprop, x0=np.append(thetas1,thetas2), args=(input_layer_size, hidden_layer_size,
    num_etiquetas, X, y_onehot,1000 ), method = 'TNC', jac = True, options = {'maxiter': 500, 'disp':True})
    theta1,theta2 = unroll_thetas(result.x,input_layer_size,hidden_layer_size,num_etiquetas)
    #Creamos la NN con las thetas optimizadas
    X = add_ones(X)
    h = forward_propagate(X, theta1, theta2)[4]
    correct = 0
    wrong = 0
    falsePositive = 0
    falseNegative = 0
    #Y comparamos la respuesta de la NN con la real
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
    print()
    print("Hit: ", correct)
    print("Miss: ", wrong)
    print("False Positives: ", falsePositive)
    print("False Negatives: ", falseNegative)
    print("Accuracy: ",format((correct / (correct+wrong))*100, '.2f' ),"%")

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

def sigmoid(Z):
    return 1.0/( 1.0 + np.exp(-Z))

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
    #a1 = np.hstack([np.ones([m, 1]), X]) No añadimos porque ya lo hicimos antes
    z2 = np.dot(X, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return X, z2, a2, z3, h

def add_ones(valores):
    unos = np.ones((valores.shape[0], 1), dtype=valores.dtype)
    return np.hstack((unos , valores))

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


main()
