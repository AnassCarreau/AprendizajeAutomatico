
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

#metodo que se encarga d calcular el coste
def coste(X,Y,theta):
    m=len(X)
    aux = np.square(theta[0]+theta[1]*X - Y)
    sum = np.sum(aux)
    return sum/(2*m)

def make_data ( t0_range , t1_range , X , Y ) :
    """ Genera las matrices Theta0 , Theta1 , Coste para generar un plot en 3D
        del coste para valores de theta_0 en el intervalo t0_range y
        valores de theta_1 en el intervalor t1_range"""

    step = 0.1
    Theta0 = np.arange ( t0_range[0], t0_range[1], step )
    Theta1 = np.arange ( t1_range[0], t1_range[1], step )
    Theta0 , Theta1 = np.meshgrid (Theta0 , Theta1)
    Coste = np.empty_like (Theta0)
    for ix , iy in  np.ndindex(Theta0.shape):
        Coste [ ix , iy ] = coste(X, Y ,[Theta0 [ix ,iy ] , Theta1[ix ,iy]])
    return [ Theta0 , Theta1 , Coste ]




datos=carga_csv("ex1data1.csv")
alpha=0.01
ite=1500
X=datos[:,0]
Y=datos[:,1]
m=len(X)
theta0 = 0
theta1 = 0
 
#for _ in range(ite) :
#     sum0=0
#     sum1=0
#     for i in range(m):
#         sum0+=theta0+theta1 * X[i] - Y[i]
#         sum1+=(theta0+theta1 * X[i] - Y[i])*X[i]
#     theta0=theta0- (alpha/m) *sum0
#     theta1=theta1- (alpha/m) *sum1
#plt.plot(X, Y, "x")
#min_x = min(X)
#max_x = max(X)
#min_y = theta0 + theta1 * min_x
#max_y = theta0 + theta1 * max_x
#plt.plot([min_x, max_x], [min_y, max_y])
#plt.savefig("resultado.png")



data=make_data( [-10,10], [-1,4],X,Y )

plt.contour(data[0], data[1], data[2],
 np.logspace(-2, 3, 20), colors='blue')
plt.savefig("grafica2D.png")

#para pintar una figura 3D
fig = plt.figure()
ax = fig.gca(projection='3d')

surf=ax.plot_surface(data[0],data[1],data[2],cmap=cm.coolwarm,linewidth=0,antialiased=False)


ax.set_zlim(0,700)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("grafica3D.png")
plt.show()

