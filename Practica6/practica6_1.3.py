import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from sklearn.svm import SVC

def RepresentData(pos,neg):
    plt.plot(pos[:, 0], pos [:, 1], '+',color = 'k', label = "Positive Values")
    plt.plot(neg[:, 0], neg [:, 1], "ro",color = 'y' ,label = "Negative Values")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.legend()
    plt.grid(True)

def DrawLimit(svm, pos, neg):
    plt.figure()
    X = np.linspace( -0.6, 0.3,200)
    Y = np.linspace( -0.6, 0.6,200)
    zVals = np.zeros(shape = (len(X), len(Y)))
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            aux = np.array([[X[i],Y[j]]])
            zVals[i][j] = float(svm.predict(aux))
    zVals = zVals.T
    RepresentData(pos,neg)
    plt.contour(X,Y,zVals)

def main():
# Load data
    dataset3 = loadmat ("ex6data3.mat")
    X3 = dataset3['X']
    Y3 = dataset3['y']
    xTest = dataset3['Xval'] 
    yTest = dataset3['yval'] 
    pos = np.array([X3[i] for i in range(len(X3))if Y3[i]==1])
    neg = np.array([X3[i] for i in range(len(X3))if Y3[i]==0])
    
    plt.figure()
    RepresentData(pos,neg)
    plt.show()

    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    #copiamos sigma
    sigma_vec = np.copy(C_vec)
    #variables para guardar los mejores parametros
    bestC = 0.01
    bestSigma = 0.01
    bestScore = -1
    #recorremos el vector de seleccion de C y sigma quedandonos con los mejores parametros (C,Sigma,Score)
    for c in  C_vec:
        for sigma in sigma_vec:
            auxKernel = SVC(C = c, kernel = 'rbf', gamma =  1/(2*sigma **2))
            auxKernel.fit(X3, Y3.flatten())
            score = auxKernel.score(xTest,yTest)
            if (score > bestScore):
                bestC = c
                bestSigma = sigma   
                bestScore = score
    
    
    gKernel = SVC (C = bestC, kernel = 'rbf', gamma =  1/(2*bestSigma **2))
    #ajustamos el vector machine
    gKernel.fit(X3, Y3.flatten())
    #figura 4b 
    print(bestC)
    DrawLimit(gKernel, pos, neg)
    plt.show()
    
main()