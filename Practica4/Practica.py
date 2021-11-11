import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.polynomial import poly
from pandas.io.parsers import read_csv
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from scipy.optimize import minimize
import displayData as displayData
#from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def sigmoideDerivada(value):
    temp = sigmoid(value)
    return temp * (1 - temp)

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

def NeuralNetworkCost(params, n_entries, n_hidden, n_et, X, Y, reg):
    theta1, theta2 = unroll_thetas(params, n_entries, n_hidden, n_et)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    j = 0
    for i in range(len(X)):
        j += (-1 / (len(X))) * (np.dot(Y[i], np.log(h[i])) + np.dot((1 - Y[i]), np.log(1 - h[i])))

    j += (reg / (2*len(X))) + ((np.sum(np.square(theta1[:,1:]))) + (np.sum(np.square(theta2[:,1:]))))

    return j


def InitRandomWeight(L_in, L_out):
    cini = 0.2
    a = np.random.uniform(-cini, cini, size = (L_in, L_out))
    a = np.insert(a, 0, 1, axis = 0)
    return a

def NNTest (num_entradas, num_ocultas, num_etiquetas, reg, X, Y, laps):
    t1 = InitRandomWeight(num_entradas, num_ocultas)
    t2 = InitRandomWeight(num_ocultas, num_etiquetas)

    params = np.hstack((np.ravel(t1), np.ravel(t2)))
    out = opt.minimize(fun = backprop, x0 = params, args = (num_entradas, num_ocultas, num_etiquetas, X, Y, reg), method='TNC', jac = True, options = {'maxiter': laps})

    Thetas1 = out.x[:(num_ocultas*(num_entradas+1))].reshape(num_ocultas,(num_entradas+1))
    Thetas2 = out.x[(num_ocultas*(num_entradas+1)):].reshape(num_etiquetas,(num_ocultas+1))

    input = np.hstack([np.ones((len(X), 1)), X])
    hipo = forward_propagate(Thetas1, Thetas2, input)[3]


    Ghipo = (hipo.argmax(axis = 0))+1
    prec = (Ghipo == Y)*1
    
    precision = sum(prec) / len(X)

    print("Program precision: ", precision *100, "%")

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



def add_ones(val):
    ones = np.ones((val.shape[0], 1), dtype=val.dtype)
    return np.hstack((ones, val))


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



# main
# numero de etiquetas
num_labels = 10

data = loadmat('ex4data1.mat')
Y = data['y'].ravel() # el metodo ravel hace que y pase de un shape(5000,1) a (5000,)
X = data['X']
nMuestras = len(X)


print(Y[2500])
plt.figure()
displayData.displayImage(X[2500])
plt.savefig("Input_sample")
plt.show()


thetas = loadmat("ex4weights.mat")
thetas1 = thetas["Theta1"] # Theta1 es de dimensión 25 x 401
thetas2 = thetas["Theta2"] # Theta2 es de dimensión 10 x 26

X_aux = np.hstack([np.ones((len(X), 1)), X])
print("Valor predicho para el elemento 0 de X segun la hipotesis: ", (forward_propagate(thetas1, thetas2, X_aux)[3]).T[0].argmax()) 

Y_aux = getYMatrix(Y, 10)

print("El coste con thetas entrenados es: ", costFun(X_aux, Y_aux, thetas1, thetas2, 1))

Thetas = [thetas1, thetas2]
unrolled_Thetas = [Thetas[i].ravel() for i,_ in enumerate(Thetas)]

NNTest(400, 25, 10, 1, X, Y, 100)
##Gradiente




