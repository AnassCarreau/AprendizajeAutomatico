
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
  # print(newMatriz)
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




def descenso_gradiente(X,Y,alpha):
  matriz_norm, medias, desviacion = normalizacion(X)
  ite = 1500
  n = np.shape(X)[1]
  Costes = np.zeros(ite)
  Thetas = np.zeros(n)
  for i in range(ite):
    Thetas = gradiente(matriz_norm,Y,Thetas,alpha)
    Costes[i] = coste(matriz_norm,Y,Thetas)
  plt.figure()
  x = np.linspace(0, 500, ite, endpoint = True)
  x = np.reshape(x, (ite, 1))
  plt.plot(x, Costes)
  plt.savefig("coste.png")
  plt.show()
  return Thetas,Costes,medias,desviacion


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
Thetas,Costes,medias,desviacion = descenso_gradiente(X, Y, alpha)
extension=(1650-medias[1])/desviacion[1]
habitaciones=(3-medias[2])/desviacion[2] 
prediccion=[1,extension,habitaciones]
print(np.matmul(np.transpose(Thetas),prediccion))
