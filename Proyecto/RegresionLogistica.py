import numpy as np
import pandas as pd
from pandas.io.formats.format import DataFrameFormatter
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
import math
import scipy.optimize as opt

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def coste(theta, X, Y,landa):
    H = sigmoid(np.matmul(X, theta))
    thetaAux=theta[1:]
    return (-1/(len(X))) * ( np.dot(Y, np.log(H)) + np.dot(1-Y, np.log(1-H))) + (landa/2*len(X))*sum(thetaAux*thetaAux)

def gradiente(theta, X, Y,landa):
    H=sigmoid(np.matmul(X, theta))
    return (1/len(Y)) * np.matmul((X.T), H-Y) + (landa/len(X))*theta


def evalua(theta,X,Y):
    xs = sigmoid(np.matmul(X,theta))
    xspositivas = np.where(xs >= 0.5)
    xsnegativas = np.where(xs < 0.5)
    xspositivasejemplo = np.where (Y == 1 )
    xsnegativasejemplo = np.where (Y == 0 )
    #Printea los casos en los que la funcion sigmoide con las thetas de ejemplo indica que va a tener un ataque
  #  print("Acertadas en el sigmoide: ", xspositivas)
  #  print("Positivas en los ejemplos: ", xspositivasejemplo)
    porcentajepos = np.intersect1d(xspositivas,xspositivasejemplo).shape[0]/xs.shape[0]
    porcentajeneg = np.intersect1d(xsnegativas,xsnegativasejemplo).shape[0]/xs.shape[0]
    print("Total:", porcentajeneg + porcentajepos)
    return porcentajepos + porcentajeneg



def calcula_porcentaje(X,Y,theta):
    sig = sigmoid(np.matmul(X,theta))
    ev_correct = np.sum((sig >= 0.5) == Y)
    return ev_correct/len(sig) * 100



def add_ones(valores):
    unos = np.ones((valores.shape[0], 1), dtype=valores.dtype)
    return np.hstack((unos , valores))

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
    
    landa=1.0
    precision = 0.0
    NUM_PRUEBAS = 10
    for i in range(NUM_PRUEBAS):
       np.random.shuffle(X)
       #Vamos a separar los ejemplos en 80% para entrenar y un 20% para evaluar  
      # X = add_ones(X)
       X= np.hstack((  np.ones((valores.shape[0], 1), dtype=valores.dtype), valores))
       print(X)
       x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,shuffle = True)
       theta = np.zeros((X.shape[1],1))
       result = opt.fmin_tnc(func=coste,x0=theta ,fprime=gradiente,args =(x_train, y_train,landa))
       theta = result[0]
       
       precision += evalua(theta,x_test,y_test)
    print("Avg. Accuracy: ",format((precision / NUM_PRUEBAS)* 100, '.2f' ),"%")
    


    
main()