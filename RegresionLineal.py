
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    """carga el fichero csv especificado y lo
 devuelve en un array de numpy
    """
    valores = read_csv(file_name, header=None).to_numpy()
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

datos=carga_csv("ex1data1.csv")
alpha=0.01
ite=1500
X=datos[:,0]
Y=datos[:,1]
m=len(X)
theta0 = 0
theta1 = 0
for _ in range(ite) :
    sum0=0
    sum1=0
    for i in range(m):
        sum0+=theta0+theta1 * X[i] - Y[i]
        sum1+=(theta0+theta1 * X[i] - Y[i])*X[i]
    theta0=theta0- (alpha/m) *sum0
    theta1=theta1- (alpha/m) *sum1
plt.plot(X, Y, "x")
min_x = min(X)
max_x = max(X)
min_y = theta0 + theta1 * min_x
max_y = theta0 + theta1 * max_x
plt.plot([min_x, max_x], [min_y, max_y])
plt.savefig("resultado.png")

