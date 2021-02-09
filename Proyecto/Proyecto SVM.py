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
    precision = 0
    numIter = 1
    for x in range(0, numIter):
        #Vamos a separar los ejemplos en 80% para entrenar y un 20% para evaluar
        X_new, Xval, y_new, Yval = train_test_split(X, Y, test_size = 0.2,shuffle = True)
        #Parametro de regulalizacion
        valorC = 100
        valorSigma = 100
        print ("aaaa")
        #Calculo del SVM
        #svm = SVC(kernel = 'linear', C = valorC)
        svm = SVC(kernel = 'rbf', C = valorC, gamma = 1 / (2 * valorSigma **2))
        #Modelos el svm con X, y en nuestro caso son el 80% de la muestra
        svm.fit(X_new, y_new)
        #Calculamos la precision del SVM para un conjunto de prueba, el 20% de la muestra en nuestro caso
        precision += test(svm, Xval, Yval)
    print(precision/ numIter)
#Calcula pa precisión de un SVM
def test(svm, X, Y):
    prediction = svm.predict(X)
    accuracy = np.mean((prediction == Y).astype(int))
    return accuracy
main()
