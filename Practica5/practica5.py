import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.optimize as opt




def cost(theta, X, Y,landa=0):
    theta = theta.reshape(-1, y.shape[1])
    #theta=theta[1:]
    costeReguralizado =(landa / len(X)*2) * np.sum(np.square(theta))
    costeNoReguralizado =(1/(len(X)*2)) * np.sum( np.square( np.matmul(X , theta )- Y)) 
    return costeNoReguralizado +costeReguralizado 
    

def gradiente(theta, X, Y,landa=0):
    m=len(X)
    gradienteNoRegularizada = (1/m)*np.dot(X.T,np.dot(X,theta)-y)
    gradiente = gradienteNoRegularizada +(landa/m)*theta
    gradiente[0] = gradienteNoRegularizada[0]

    return gradiente.flatten() 


def gradiente_min(thetas, matrizX, vectorY, _lambda=0.):
    return gradiente(thetas, matrizX, vectorY, _lambda=0.).flatten()

def optimizarTheta(thetas, matrizX, vectorY, _lambda=0.,_print=True):
    return opt.fmin_cg(cost,x0=thetas,fprime=gradiente_min, args=(matrizX,vectorY,_lambda),disp=_print, epsilon=1.49e-12, maxiter=1000)

landa=0
data = loadmat('ex5data1.mat')
y = data ['y']
X = data ['X']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']

theta = np.array([[1], [1]])
newX = np.insert(X, 0,1, axis=1)

print(cost(theta,newX,y,landa))
print(gradiente(theta,newX,y,landa))

X_unos = np.insert(X,0,1,axis=1)

theta = np.zeros((X.shape[1], 1))
theta = optimizarTheta(theta,X_unos,y,0.)

plt.figure(figsize=(8,6))
plt.title('Regresión lineal regularizada')
plt.plot(X, y, 'rx')
plt.plot(X, np.dot(np.insert(X, 0, 1, axis=1),theta), '--')
plt.show()
plt.close()
