import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.optimize as opt


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def cost(theta, X, Y,landa):
    theta = theta.reshape(-1, y.shape[1])
    #theta=theta[1:]
    costeReguralizado =(landa / len(X)*2) * np.sum(np.square(theta))
    costeNoReguralizado =(1/(len(X)*2)) * np.sum( np.square( np.matmul(X , theta )- Y)) 
    return costeNoReguralizado +costeReguralizado 
    

def gradiente(theta, X, Y,landa):
    m=len(X)
    gradienteNoRegularizada = (1/m)*np.dot(X.T,np.dot(X,theta)-y)
    gradiente = gradienteNoRegularizada +(landa/m)*theta
    gradiente[0] = gradienteNoRegularizada[0]

    return gradiente.flatten() 




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




landa=0
data = loadmat('ex5data1.mat')
y = data ['y']
X = data ['X']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']

#insertamos una fila de 1s en la primera fila de la matriz
Xfila1 = np.insert(X, 0, 1, axis=1)
XvalFila1 = np.insert(Xval, 0, 1, axis=1)


theta = np.array([[1], [1]])
newX = np.insert(X, 0,1, axis=1)
#print(Xfila1)
print("coste")
print(cost(theta,newX,y,landa))
print("gradiente")
print(gradiente(theta,newX,y,landa))


theta = np.zeros((X.shape[1], 1))
theta = RegresionLinealRegularizada(newX, y, 0)
m = len(X)
errorTrain, errorVal = curvaAprendizaje(Xfila1, y, XvalFila1, yval, 0)



plt.figure(figsize=(8,6))
title='Learning curve for linear regression'
plt.title(title)  
plt.plot(range(1,m+1), errorTrain, label='Train')
plt.plot(range(1, m+1), errorVal, label='Cross Validation')
plt.legend()
plt.savefig(title+'.png')
plt.show()
