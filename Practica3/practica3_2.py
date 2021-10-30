import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.polynomial import poly
from pandas.io.parsers import read_csv
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def cost(theta, X, Y,landa):
    H = sigmoid(np.matmul(X, theta))
    thetaAux=theta[1:]
    return (-1/(len(X))) * ( np.dot(Y, np.log(H)) + np.dot(1-Y, np.log(1-H))) + (landa/2*len(X))* np.sum(thetaAux*thetaAux)
    

def gradiente(theta, X, Y,landa):
    H=sigmoid(np.matmul(X, theta))
    Theta2 = theta
    Theta2[0] = 0
    aux =(1/len(Y)) * np.matmul((X.T), H-Y) 
    return aux + ((landa/len(X) * Theta2) )


def h(X, thetas1, thetas2):
    a1 = X
    #m = np.shape(X)[0]
    #a1 = np.hstack([np.ones([m, 1]), X])
    # z2 = np.matmul(thetas1, a1)
    z2 = np.matmul(thetas1, np.insert(a1,0,1))
    a2 = sigmoid(z2)
    z3 = np.matmul(thetas2, np.insert(a2,0,1))
    a3 = sigmoid(z3)
    return a3


def getEtiqueta(Y, etiqueta):
    y_etiqueta = np.ravel(Y)== etiqueta
    y_etiqueta = y_etiqueta *1
    return y_etiqueta


def porcentaje (theta, X, Y):
    aux = sigmoid(np.matmul(X, theta))
    auxpos = np.where(aux >= 0.5)
    auxneg = np.where(aux< 0.5)
    auxposexample = np.where (Y == 1)
    auxnegexample = np.where (Y == 0)

    porcentajePos = np.intersect1d(auxpos, auxposexample).shape[0]/aux.shape[0]
    porcentajeNeg = np.intersect1d(auxneg, auxnegexample).shape[0]/aux.shape[0]
    print("Total:", porcentajeNeg + porcentajePos)
    return porcentajeNeg + porcentajePos




#carga de los datos
data = loadmat('ex3data1.mat')
y = data ['y']
X = data ['X']
#almacena los datos leidos en X,Y
thetas = loadmat("ex3weights.mat")
thetas1,thetas2 = thetas["Theta1"], thetas["Theta2"]

#oneVsAll(X, y, 10, 0.1)

sample = np.random.choice(X.shape[0],10)
plt.imshow(X[sample, :].reshape(-1,20).T)
plt.axis('off')


aux = np.zeros(10)
print ("Sample: ", sample)
for i in range(10):
    aux[i] = np.argmax(h(X[sample[i],:],thetas1,thetas2))
print("My guess are: ", (aux)+1)
numAciertos = 0
for i in range(X.shape[0]):
    aux2 = np.argmax(h(X[i,:],thetas1, thetas2))
    if(aux2+1) == y[i]:
        numAciertos +=1
print("Porcentaje de aciertos: ", numAciertos / X.shape[0])
plt.show()

