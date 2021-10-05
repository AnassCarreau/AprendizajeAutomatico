
  
import numpy as np
import numpy.linalg as linalg
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

def Ecuacion_normal(X,Y):
  X_tras=np.transpose(X)
  inv_matr_XtX=linalg.pinv(np.matmul(X_tras,X))
  return np.matmul(np.matmul(inv_matr_XtX,X_tras),Y)
  
datos = carga_csv('ex1data2.csv')
X = datos[:, :-1]
np.shape(X)         # (97, 1)
Y = datos[:, -1]
np.shape(Y)         # (97,)
m = np.shape(X)[0]
n = np.shape(X)[1]
# añadimos una columna de 1's a la X
X = np.hstack([np.ones([m, 1]), X])
Thetas=Ecuacion_normal(X, Y)
prediccion=[1,1650,3]
print("Resultado Regresion Lineal Ecuacion  Normal :")
print(np.matmul(np.transpose(Thetas),prediccion))