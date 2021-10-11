import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.optimize as opt

def pinta_frontera_recta(X, Y, theta):
 plt.figure()
 x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
 x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

 xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
 np.linspace(x2_min, x2_max))

 h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
 xx1.ravel(),
 xx2.ravel()].dot(theta))
 h = h.reshape(xx1.shape)

 # el cuarto parámetro es el valor de z cuya frontera se
 # quiere pintar
 plt.contour(xx1, xx2, h, [0,5], linewidths=1, colors='b')
 plt.savefig("frontera.pdf")
 plt.close()

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).to_numpy()
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    return (-1/(len(X))) * ( np.dot(Y, np.log(H)) + np.dot(1-Y, np.log(1-H)) )

def gradiente(theta, X, Y):
    H=sigmoid(np.matmul(X, theta))
    return (1/len(Y)) * np.matmul((X.T), H-Y)

datos=carga_csv("ex2data1.csv")
X=datos[:, :-1]
Y=datos[:, -1]
# Obtiene un vector con los índices de los ejemplos positivos
pos=np.where(Y == 1)
posN=np.where(Y == 0)
# Dibuja los ejemplos positivos
plt.scatter(X[pos, 0], X[pos, 1], marker = '+', c = 'k')
plt.scatter(X[posN, 0], X[posN, 1], marker = 'o', c = 'y')

#Añadir fila de unos
X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
n=np.shape(X)[1]
Theta=np.zeros(n)
#print(cost(Theta, X, Y))
#print(gradiente(Theta, X, Y))


result = opt.fmin_tnc ( func=cost,x0=Theta , fprime=gradiente , args =(X, Y) )
theta_opt = result [0]
#print(result[1])
pinta_frontera_recta(X,Y,theta_opt)
print("hola")
print(theta_opt)

aux = cost(theta_opt,X,Y)
print(aux)
plt.show()
