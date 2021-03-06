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


def oneVsAll(X, y, n_labels, landa):
    """
    oneVsAll entrena varios clasificadores por regresión logística con término
    de regularización 'reg' y devuelve el resultado en una matriz, donde
    la fila i-ésima corresponde al clasificador de la etiqueta i-ésima
    """
    m = X.shape[1]
    theta = np.zeros((n_labels, m))
    y_etiquetas = np.zeros((y.shape[0], n_labels))

    for i in range(n_labels):
        y_etiquetas[:,i] = getEtiqueta(y, i)
    y_etiquetas[:,0] = getEtiqueta(y, 10)

    for i in range(n_labels):
        print("I: ", i)
        result = opt.fmin_tnc(func=cost, x0=theta[i,:], fprime=gradiente, args=(X, y_etiquetas[:,i], landa))
        theta[i, :] = result[0]

    evaluacion = np.zeros(n_labels)
    for i in range(n_labels):
        evaluacion[i] = porcentaje(theta[i,:], X, y_etiquetas[:,i])
    print("Evaluacion: ", evaluacion)
    print("Evaluacion media: ", evaluacion.mean())
    return 0





#carga de los datos
data = loadmat('ex3data1.mat')
y = data ['y']
X = data ['X']
#almacena los datos leidos en X,Y


oneVsAll(X,y,10,0.1)


#Selecciona aleatoriamente 10 ejemplos y los pinta
#sample = np.random.choice(X.shape[0],10)
#plt.imshow(X[sample , :].reshape(-1,20).T)
#plt.axis('off')
#plt.show()