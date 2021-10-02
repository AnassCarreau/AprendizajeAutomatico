
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator,FormatStrFormatter
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib import cm

#metodo que se encrga de la carga de los datos desde el fichero
def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).to_numpy()
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)


def normalizacion(X):
   m = np.shape(X)[0]
   n = np.shape(X)[1]
   medias = np.mean(X,0,dtype=np.float64)
   desviacion = np.std(X,0,dtype=np.float64)
   #newMatriz = (X-medias) / desviacion
   newMatriz = np.copy(X)
   for i in np.arange(n-1):
     newMatriz[:,i+1] =  (X[:,i+1]-medias[i+1]) / desviacion[i+1]
  # print(X)
  # print(n)
   print(newMatriz)
   return newMatriz,medias,desviacion



def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def gradiente(X, Y, Theta, alpha):
 NuevaTheta = Theta
 m = np.shape(X)[0]
 n = np.shape(X)[1]
 H = np.dot(X, Theta)
 Aux = (H - Y)
 for i in range(n):
   Aux_i = Aux * X[:, i]
   NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
 return NuevaTheta

def descenso_gradiente2(X, Y, alpha, m, n):
    Thetas = np.zeros(X[1].size)

    for i in range(400):

        Thetas = Thetas - alpha * (1 / m) * ( X.T.dot(np.dot(X, Thetas) - Y) )

        coste(X, Y, Thetas)
        costes = []
        costes = np.append(costes, coste(X, Y, Thetas))

    return Thetas








def descenso_gradiente(X,Y,alpha):
  matriz_norm, medias, desviacion = normalizacion(X)
  ite = 1500
  m = np.shape(X)[0]
  n = np.shape(X)[1]
  Costes = np.zeros(ite)
  #print(matriz_norm)
  #print(medias)
  #print(desviacion)
  Thetas = np.zeros(n)
  for i in range(ite):
    Thetas = gradiente(matriz_norm,Y,Thetas,alpha)
    Costes[i] = coste(X,Y,Thetas)
    print( Costes[i])
  print("hola")
  print(Thetas)
  print (Costes)
  #print(coste(X,Y,Thetas))
  plt.plot(Costes, [0,ite])
  plt.savefig("coste.png")



  

datos = carga_csv('ex1data2.csv')
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)         # (97,)
m = np.shape(X)[0]
n = np.shape(X)[1]
# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])
alpha = 0.01
#Thetas, costes = descenso_gradiente(X, Y, alpha)
descenso_gradiente(X, Y, alpha)

