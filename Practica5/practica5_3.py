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

def coste(theta, X, Y,_lambda=0.):
    theta = theta.reshape(-1, y.shape[1])
    costeReguralizado =(_lambda / len(X)*2) * np.sum(np.square(theta))
    costeNoReguralizado =(1/(len(X)*2)) * np.sum( np.square( np.matmul(X , theta )- Y)) 
    return costeNoReguralizado +costeReguralizado 
    

def gradiente(thetas, matrizX, vectorY, _lambda=0.):
    thetas = thetas.reshape((thetas.shape[0],1))
    nMuestras = matrizX.shape[0]    
    grad = (1./float(nMuestras))*matrizX.T.dot(hipotesis(thetas,matrizX)-vectorY) + (float(_lambda)/nMuestras)*thetas
    return grad


def gradiente_min(thetas, matrizX, vectorY, _lambda=0.):
    return gradiente(thetas, matrizX, vectorY, _lambda=0.).flatten()

def optimizarTheta(thetas, matrizX, vectorY, _lambda=0.,_print=True):
    return opt.fmin_cg(coste,x0=thetas,fprime=gradiente_min, args=(matrizX,vectorY,_lambda),disp=_print, epsilon=1.49e-12, maxiter=1000)


# Recibe una matriz X (dimensiones m*l) y devuelve otra con dimensiones m*p
def generar_dimension(matrizX, p):    
    for i in range(p):
        dim = i+2
        matrizX = np.insert(matrizX,matrizX.shape[1],np.power(matrizX[:,1],dim),axis=1)
    return matrizX
#Funcion para evitar grandes diferencias de rango
def normalizar_atributos(matrizX):    
    medias = np.mean(matrizX,axis=0) 
    matrizX[:,1:] = matrizX[:,1:] - medias[1:]
    desviaciones = np.std(matrizX,axis=0,ddof=1)
    matrizX[:,1:] = matrizX[:,1:] / desviaciones[1:]
    return matrizX, medias, desviaciones #MatrizX esta normalizada

def hipotesis_polinomial(thetas, medias, desviaciones):    
    puntos = 50
    xvals = np.linspace(-55,55,puntos)
    xmat = np.ones((puntos,1))
    
    xmat = np.insert(xmat, xmat.shape[1],xvals.T,axis=1)
    xmat = generar_dimension(xmat,len(thetas)-2)
    
    xmat[:,1:] = xmat[:,1:] - medias[1:]
    xmat[:,1:] = xmat[:,1:] / desviaciones[1:]
    
    plt.figure()
    plt.plot(X, y,'rx')
    plt.plot(xvals, xmat.dot(thetas),'b--')
    plt.show()
def curva_aprendizaje():    
    theta = np.ones((2,1))
    muestras, trainVector, valVector = [], [], []
    for x in range(1,13,1):
        train_aux = X_unos[:x,:] #vamos seleccionando primer conjunto entrenamiento, luego segundo...
        y_aux = y[:x]        
        fit_theta = optimizarTheta(theta,train_aux,y_aux,_lambda=0.,_print=False)
        trainVector.append(coste(fit_theta,train_aux,y_aux,_lambda=0.))
        valVector.append(coste(fit_theta, Xval, yval,_lambda=0.))
        muestras.append(y_aux.shape[0])

    return trainVector, valVector, muestras

# Dibujado y cálculo de curvas de aprendizaje
def dibuja_curva_aprendizaje_polinomio(_lamb=0.):
    thetas = np.ones((grado_polinomio+2, 1))
    muestras, vector_train, vector_val = [], [], []
    matrizXval, aux1, aux2 = normalizar_atributos(generar_dimension(Xval, grado_polinomio))

    for x in range(1,13,1):
        train_aux = X_unos[:x,:]
        y_aux = y[:x]
        muestras.append(y_aux.shape[0])
        train_aux = generar_dimension(train_aux, grado_polinomio)   
        train_aux, aux1, aux2 = normalizar_atributos(train_aux)
        theta_opt = optimizarTheta(thetas, train_aux, y_aux,_lambda=_lamb, _print=False)
        vector_train.append(coste(theta_opt, train_aux, y_aux,_lambda=_lamb))
        vector_val.append(coste(theta_opt, matrizXval, yval, _lambda=_lamb))
        
    plt.figure()
    plt.plot(muestras, vector_train, label='Train')
    plt.plot(muestras, vector_val, label='Cross Validation')
    plt.legend()
    plt.title('Polynomial Regression Learning Curve (lambda = ' + str(_lamb) + ')')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.ylim([0,100])
    plt.show()

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

print(coste(theta,newX,y,landa))
print(gradiente(theta,newX,y,landa))

X_unos = np.insert(X,0,1,axis=1)
Xval = np.insert(Xval,0,1,axis=1)
Xtest=np.insert(Xtest,0,1,axis=1)
# Dibujado de hipótesis polinomial
grado_polinomio = 8
X_poli = generar_dimension(X_unos,grado_polinomio)
X_poli_norm, media, desviacion = normalizar_atributos(X_poli)
Theta = np.ones((X_poli_norm.shape[1],1))
theta_opt = optimizarTheta(Theta, X_poli_norm, y, 0.)
hipotesis_polinomial(theta_opt, media, desviacion)
#Representar curvas de aprendizaje
dibuja_curva_aprendizaje_polinomio(0)