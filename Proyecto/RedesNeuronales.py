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
from scipy.optimize import minimize

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


#A1==A2
#Z2==Z3
#A2==H
def forward_propagate(Theta1, Theta2, X):
    z1 = Theta1.dot(X.T)
    a1 = sigmoid(z1)
    tuple = (np.ones(len(a1[0])), a1)
    a1 = np.vstack(tuple)
    z2 = Theta2.dot(a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def random_thetas(l_in, l_out, range=0.12):
    W = np.zeros((l_out, 1 + l_in))
    W = np.random.rand(l_out, 1 + l_in) * (2 * range) - range
    return W


def unroll_thetas(params, n_entries, n_hidden, n_et):
    theta1 = np.reshape(params[:n_hidden * (n_entries + 1)], (n_hidden, (n_entries + 1)))
    theta2 = np.reshape(params[n_hidden * (n_entries + 1):], (n_et, (n_hidden + 1)))

    return theta1, theta2


def costFun(X, y, theta1, theta2, reg):
    #Cambios para poder operar
    X = np.array(X)
    y = np.array(y)
    muestras = len(y)

    theta1 = np.array(theta1)
    theta2 = np.array(theta2)
    
    #predecimos la salida de los valores para la matriz de pesos de theta y nos quedamos con el valor predicho en la
    #variable hipothesis y calculamos el coste con el valor de dicha variable
    hipothesis  = forward_propagate(theta1, theta2, X)[3]
    cost = np.sum((-y.T)*(np.log(hipothesis)) - (1-y.T)*(np.log(1- hipothesis)))/muestras
    
    #calculo del coste con regularización 
    regcost = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:,1:], 2))
    regcost = regcost * (reg/(2*muestras))

    return cost + regcost


def getYMatrix(Y, nEtiquetas):
    nY = np.zeros((len(Y), nEtiquetas))
    yaux = np.array(Y) -1
    
    for i in range(len(nY)):
        z = yaux[i]
        if(isinstance(z, np.uint8)):
            if(z == 10): z = 0
            nY[i][z] = 1
        else:
            z = yaux[i].all()
            if(z == 10): z = 0
            nY[i][z] = 1
            
    return nY

def sigmoideDerivada(value):
    temp = sigmoid(value)
    return temp * (1 - temp)
    
def backprop(params_rn, num_entradas,num_ocultas, num_etiquetas, X, Y, reg):
    th1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    # theta2 es un array de (num_etiquetas, num_ocultas)
    th2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))
    
    X_unos = np.hstack([np.ones((len(X), 1)), X])
    nMuestras = len(X)
    y = np.zeros((nMuestras, num_etiquetas))
    
    y = y + getYMatrix(Y, num_etiquetas)
    
    coste = costFun(X_unos, y, th1, th2, reg)
    
    #Backpropagation
    
    # Forward propagation para obtener una hipótesis y los valores intermedios
    # de la red neuronal
    z2, a2, z3, a3 = forward_propagate(th1, th2, X_unos)
    
    gradW1 = np.zeros(th1.shape)
    gradW2 = np.zeros(th2.shape)
    
    # Coste por capas
    delta3 = np.array(a3 - y.T)
    delta2 = th2.T[1:, :].dot(delta3)*sigmoideDerivada(z2)
    
    # Acumulacion de gradiente
    gradW1 = gradW1 + (delta2.dot(X_unos))
    gradW2 = gradW2 + (delta3.dot(a2.T))
    
    G1 = gradW1/float(nMuestras)
    G2 = gradW2/float(nMuestras)
    
    # suma definitiva
    G1[:, 1: ] = G1[:, 1:] + (float(reg)/float(nMuestras))*th1[:, 1:]
    G2[:, 1: ] = G2[:, 1:] + (float(reg)/float(nMuestras))*th2[:, 1:]
    
    gradients = np.concatenate((G1, G2), axis = None)
    
    return coste, gradients



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
    ######
    num_entradas = 11
    #Las unidades de la capa oculta
    num_ocultas = 25
    #Las etiquetas de la salida
    num_etiquetas = 2
    #Y.reshape(Y.shape[0],1)
    m = X.shape[0]
    y_onehot = np.zeros((m,num_etiquetas))
    for i in range(m):
        y_onehot[i][int(Y[i])]= 1
       
    #Sacamos las matrices de thetas 
    thetas1 = random_thetas(num_entradas,num_ocultas)
    thetas2 = random_thetas(num_ocultas,num_etiquetas)
    Thetas = [thetas1,thetas2]
    #Unrolleamos los parametros y los juntamos en el mismo
    unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]
    nn_params = np.concatenate(unrolled_Thetas)
    print("Shape of Theta1: ", thetas1.shape)
    print("Shape of Theta2: ", thetas2.shape)
    print("Shape of nn_params: ", nn_params.shape)
    #EVALUACION
    #Buscamos mediante scipy las thetas optimas
    result = minimize(fun=backprop, x0=np.append(thetas1,thetas2), args=(num_entradas, num_ocultas,
    num_etiquetas, X, y_onehot,1000 ), method = 'TNC', jac = True, options = {'maxiter': 500, 'disp':True})
    theta1,theta2 = unroll_thetas(result.x,num_entradas,num_ocultas,num_etiquetas)
    #Creamos la NN con las thetas optimizadas
    X= add_ones(X)
    h = forward_propagate(theta1, theta2,X)[3]
    print(h[0])
    print(len(h))
    correct = 0
    wrong = 0
    falsePositive = 0
    falseNegative = 0
    #Y comparamos la respuesta de la NN con la real
    print("Lenx",len(X))
    for i in range(len(X)):
        maxIndex = np.argmax(h[i])
        if(maxIndex == Y[i]):
            correct += 1
        else:
            wrong += 1
            if(Y[i] == 1):
                falseNegative +=1
            else:
                falsePositive += 1
        print("hola")
    print()
    print("Hit: ", correct)
    print("Miss: ", wrong)
    print("False Positives: ", falsePositive)
    print("False Negatives: ", falseNegative)
    print("Accuracy: ",format((correct / (correct+wrong))*100, '.2f' ),"%")

    


    
main()