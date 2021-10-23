import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.polynomial import poly
from pandas.io.parsers import read_csv
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def plot_decisionboundary(X, Y, theta, poly):
 plt.figure()
 x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
 x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
 xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
 np.linspace(x2_min, x2_max))
 h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
 xx2.ravel()]).dot(theta))
 h = h.reshape(xx1.shape)
 plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
 


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


def cost(theta, X, Y,landa):
    H = sigmoid(np.matmul(X, theta))
    thetaAux=theta[1:]
    return (-1/(len(X))) * ( np.dot(Y, np.log(H)) + np.dot(1-Y, np.log(1-H))) + (landa/2*len(X))*sum(thetaAux*thetaAux)

def gradiente(theta, X, Y,landa):
    H=sigmoid(np.matmul(X, theta))
    return (1/len(Y)) * np.matmul((X.T), H-Y) + (landa/len(X))*theta


datos=carga_csv("ex2data2.csv")
X=datos[:, :-1]
Y=datos[:, -1]
landa=1
# Obtiene un vector con los índices de los ejemplos positivos
pos=np.where(Y == 1)
posN=np.where(Y == 0)

poly=PolynomialFeatures(6)
Xfit=poly.fit_transform(X)
print(Xfit)
print(len(Xfit))
print(np.shape(X)[1])
print(np.shape(Xfit)[1])
Theta=np.zeros(np.shape(Xfit)[1])

print(cost(Theta,Xfit,Y,landa))

result = opt.fmin_tnc ( func=cost,x0=Theta , fprime=gradiente , args =(Xfit, Y,landa) )
theta_opt = result [0]

plot_decisionboundary(X,Y,theta_opt,poly)
plt.scatter(X[pos, 0], X[pos, 1], marker = '+', c = 'k')
plt.scatter(X[posN, 0], X[posN, 1], marker = 'o', c = 'y')
plt.savefig("boundary.pdf")
plt.show()
plt.close()

