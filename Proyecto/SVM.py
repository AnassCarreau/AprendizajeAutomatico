import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
import math
import scipy.optimize as opt
from scipy.optimize import minimize
import sklearn
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
    #Leemos el archivo csv
    data=pd.read_csv("heart.csv")
    #Parseamos las columnas que no tengan enteros a enteros
    data.Sex=[1 if each == "M" else 0 for each in data.Sex]
    data.ExerciseAngina=[1 if each == "N" else 0 for each in data.ExerciseAngina]
    data.ChestPainType=[0 if each == "ASY"  else 1 if each=="NAP" else 2  for each in data.ChestPainType]
    data.RestingECG=[0 if each == "Normal"  else 1 if each=="LVH" else 2  for each in data.RestingECG]
    data.ST_Slope=[0 if each == "Flat"  else 1 if each=="Up" else 2  for each in data.ST_Slope]
    #guardamos data
    valores=data.values
    #la columna Y tendra los ataques al corazon
    Y=data.HeartDisease.values
    #Borramos la columna para asignar a X el resto de columnas
    data.drop(['HeartDisease'], axis=1,inplace=True)
    X=data.values

    precision=0
    numIter = 1
    #Parametro de regulalizacion
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    #copiamos sigma
    sigma_vec = np.copy(C_vec)

    #for con iteraciones  para probar
    for x in range(0, numIter):
        #variables para guardar los mejores parametros
        bestC = 0.01
        bestSigma = 0.01
        bestScore = -1
        #Vamos a separar los ejemplos en 80% para entrenar y un 20% para evaluar
        X_new, Xval, y_new, Yval = train_test_split(X, Y, test_size = 0.20,shuffle = True)
        #recorremos el vector de seleccion de C y sigma quedandonos con los mejores parametros (C,Sigma,Score)
        for c in  C_vec:
            for sigma in sigma_vec:
                auxKernel = SVC(C = c, kernel = 'rbf', gamma =  1/(2*sigma **2))
                auxKernel.fit(X_new, y_new.flatten())
                score = auxKernel.score(Xval,Yval)
                if (score > bestScore):
                    bestC = c
                    bestSigma = sigma   
                    bestScore = score
        #Calculo del SVM
        svm = SVC(kernel = 'rbf', C = bestC, gamma = 1 / (2 * bestSigma **2))
        #Modelos el svm con X, y en nuestro caso son el 80% de la muestra
        svm.fit(X_new, y_new.flatten())
        #Calculamos la precision del SVM para un conjunto de prueba, el 20% de la muestra en nuestro caso
        precision += test(svm, Xval, Yval)
    #imprimimos la media de precicion del numero de iteraciones
    print(precision/ numIter)
       
    

def test(svm, X, Y):
    prediction = svm.predict(X)
    accuracy = np.mean((prediction == Y).astype(int))
    return accuracy
main()