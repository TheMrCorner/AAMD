import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.optimize as opt

def CosteGrandiente(X, y, theta, coeficienteLambda):
    m = len(X)
    theta = theta.reshape(-1, y.shape[1])
    costeReguralizado = (coeficienteLambda / (2*m)) * np.sum(np.square(theta[1:len(theta)]))
    costeNoReguralizado = (1/(2*m)) * np.sum(np.square(np.dot(X,theta)-y))
    costeTotal = costeReguralizado + costeNoReguralizado

    gradiente =np.zeros(theta.shape)
    gradiente = (1/m)*np.dot(X.T, np.dot(X, theta)-y)+(coeficienteLambda/m)*theta
    gradienteNoRegularizada = (1/m)*np.dot(X.T,np.dot(X,theta)-y)
    gradiente[0] = gradienteNoRegularizada[0]

    return (costeTotal, gradiente.flatten())


def RegresionLinealRegularizada(X, y, coeficienteLambda):
    thetaIni = np.zeros((X.shape[1], 1))
    def fcoste(theta):
        return CosteGrandiente(X, y, theta, coeficienteLambda)
    results = opt.minimize(fun=fcoste, x0=thetaIni, method='CG', jac=True,options={'maxiter':200})
    theta = results.x
    return theta

def curvaAprendizaje(X, y, Xval, yval, coeficienteLambda):
    m = len(X)
    errorTrain= np.zeros((m,1))
    errorVal = np.zeros((m,1))

    for i in range(1, m+1):
        theta = RegresionLinealRegularizada(X[:i], y[:i], coeficienteLambda)
        errorTrain[i-1] = CosteGrandiente(X[:i], y[:i], theta, 0)[0]
        errorVal[i-1] = CosteGrandiente(Xval, yval, theta, 0)[0]
    
    return errorTrain, errorVal

def creaPoly(X, p):
    Xpoly = X
    for i in range(1,p):
        Xpoly = np.column_stack((Xpoly, np.power(X, i+1)))
    return Xpoly

def normalizaX(X):
    mu = np.mean(X, axis=0)
    normX = X - mu
    sigma = np.std(normX, axis =0)
    normX = normX /sigma
    return normX, mu, sigma

def pinta(X, y, mu, sigma, theta, p):
    plt.figure(figsize=(8,6))
    plt.title('Regresion polinomial')
    plt.plot(X, y,'rx')
    minX = min(X)
    maxX = max(X)
    x = np.array(np.arange(minX - 15, maxX + 25, 0.05))
    Xpoly = creaPoly(x, p)
    Xpoly = Xpoly - mu
    Xpoly = Xpoly / sigma
    Xpoly = np.insert(Xpoly, 0, 1 ,axis=1)
    plt.plot(x, np.dot(Xpoly, theta), '--')
    
def pinta2(m, errorTrain, errorVal):
    plt.figure(figsize=(8,6))
    plt.title('Regresion polinomial')
    plt.xlabel('Ejemplos de entrenamientos')
    plt.ylabel('Errores')
    plt.plot(range(1,m+1), errorTrain, 'b', label='Train')
    plt.plot(range(1, m+1), errorVal, 'g', label='Cross Validation')
    plt.legend()    

data = loadmat("ex5data1.mat")

y = data['y']
X = data['X']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
m = len(X)
p=8
Xpoly = creaPoly(X,p)
Xpoly, mu, sigma = normalizaX(Xpoly)
Xpoly = np.insert(Xpoly, 0, 1, axis=1)
XpolyTest = creaPoly(Xtest, p)
XpolyTest = XpolyTest - mu
XpolyTest = XpolyTest / sigma
XpolyTest = np.insert(XpolyTest, 0, 1, axis=1)

theta = RegresionLinealRegularizada(Xpoly, y, 0)
pinta(X, y, mu, sigma, theta, p)

XpolyVal = creaPoly(Xval, p)
XpolyVal = XpolyVal - mu
XpolyVal = XpolyVal / sigma
XpolyVal = np.insert(XpolyVal, 0,1, axis=1)
errorTrain, errorVal = curvaAprendizaje(Xpoly, y, XpolyVal, yval, 0)
pinta2(m , errorTrain, errorVal)
plt.show()


