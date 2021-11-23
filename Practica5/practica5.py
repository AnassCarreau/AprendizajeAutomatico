import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.optimize as opt



def hipotesis(theta,X):
    return X.dot(theta)

def coste(thetas, matrizX, vectorY, λ=0):
    m = matrizX.shape[0]
    hipo = hipotesis(thetas,matrizX).reshape((m,1))
    cost = (1/(2*m) * np.dot((hipo-vectorY).T,(hipo-vectorY))) + (λ/(2*m)) * thetas[1:].T.dot(thetas[1:])
    return cost 

def gradiente(thetas, matrizX, vectorY, λ=0):
    m = matrizX.shape[0]    
    thetas = thetas.reshape((thetas.shape[0],1))
    grad = (1/m)*matrizX.T.dot(hipotesis(thetas,matrizX)-vectorY) + (λ/m)*thetas
    return grad

def gradiente_min(thetas, matrizX, vectorY, λ=0):
    return gradiente(thetas, matrizX, vectorY, λ=0.).flatten()

def optimizarTheta(thetas, matrizX, vectorY, λ=0,_print=True):
    return opt.fmin_cg(coste,x0=thetas,fprime=gradiente_min, args=(matrizX,vectorY,λ),disp=_print, epsilon=1.49e-12, maxiter=1000)

data = loadmat('ex5data1.mat')
y = data['y']
X = data['X']

yval = data['yval']
Xval = data['Xval']

X_unos = np.insert(X,0,1,axis=1)

λ=0
plt.figure()
title='Linear regression with regularization'
plt.title(title)
plt.plot(X,y,'rx')
Theta = np.ones((2,1))
theta_opt = optimizarTheta(Theta,X_unos,y,λ)
plt.plot(X,hipotesis(theta_opt,X_unos).flatten())
plt.savefig(title+'.png')
plt.show()
plt.close()
