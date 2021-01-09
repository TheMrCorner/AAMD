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

    
def pinta2(m, errorTrain, errorVal):
    plt.figure(figsize=(8,6))
    plt.title('Selección del parámetro λ')
    plt.xlabel('Lambda')
    plt.ylabel('Errores')
    plt.plot(m, errorTrain, 'b', label='Train')
    plt.plot(m, errorVal, 'g', label='Cross Validation')
    plt.legend()    

data = loadmat(r"F:\UniShit\Repos\AAMD\Practica 5\ex5data1.mat")

y = data['y']
X = data['X']
Xval = data['Xval']
yval = data['yval']
p=8
Xpoly = creaPoly(X,p)
Xpoly, mu, sigma = normalizaX(Xpoly)
Xpoly = np.insert(Xpoly, 0, 1, axis=1)

XpolyVal = creaPoly(Xval, p)
XpolyVal = XpolyVal - mu
XpolyVal = XpolyVal / sigma
XpolyVal = np.insert(XpolyVal, 0,1, axis=1)

vectorLambda = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
errorTrain = np.zeros((len(vectorLambda),1))
errorVal = np.zeros((len(vectorLambda), 1))
for i in range(len(vectorLambda)):
    coeficienteLambda = vectorLambda[i]
    theta = RegresionLinealRegularizada(Xpoly, y, coeficienteLambda)
    errorTrain[i] = CosteGrandiente(Xpoly, y, theta, 0)[0]
    errorVal[i] = CosteGrandiente(XpolyVal, yval, theta, 0)[0]

pinta2(vectorLambda, errorTrain, errorVal)
plt.show()


