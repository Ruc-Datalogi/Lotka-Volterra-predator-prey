import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def LotkaStandard():
    t = np.linspace(0,40,3600)
    alpha, beta, gamma, delta = 1.1,0.4,0.4,0.1
    R0, F0 = 10, 10

    # Actually solve the problem
    R, F = LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t)

    # Plot the solution
    plt.plot(t,R,'b.-',t,F,'r-')
    plt.legend(['Baboon','Cheetah'])
    plt.grid(True)
    plt.title("Solution of Lotka-Volterra system using explicit Euler") 
    plt.show()

def LotkaDerivative():
    pass


def LotkaModified(t,state,alpha,beta,iota,delta,gamma,kappa):
    x,y = state
    dx = alpha*x-beta*x*y-x**2*iota
    dy = delta*x*y-gamma*y

    return [dx,dy]

def LotkaGraph():
    #define population constants
    alpha = 1; beta = 1; iota = 1;
    delta = 1; gamma = 1; kappa = 1;

    #evaluate time range
    t_span = (0.0,4000.0)

    #initial values
    y0 = [3,1]

    #params
    param = (alpha,beta,iota,delta,gamma,kappa)


    #solve the differential shitsytem
    sol1 = solve_ivp(LotkaModified, t_span, y0, args=param, dense_output = True)

    fig, ax = plt.subplots()

    t_lin = np.linspace(0,6,50)

    z = sol1.sol(t_lin)

    ax.plot(t_lin,z.T)

    plt.show()



def LotkaVectorField():
    #define population constants
    alpha = 1; beta = 1; iota = 1;
    delta = 1; gamma = 1; kappa = 1;

    # Create xy plane, Vector field
    x = np.arange(0, 5, 0.2)
    y = np.arange(0, 5, 0.2)
    x, y = np.meshgrid(x, y)
    i=alpha*x-beta*x*y-x**2*iota; j=delta*x*y-gamma*y
    fig, ax = plt.subplots()

    i = i/np.sqrt(i**2+j**2)
    j = j/np.sqrt(i**2+j**2)

    ax.quiver(x, y, i, j, pivot='mid')

    
    #Streamlines
    levels  = np.arange(-5,10,0.4) #3rd value indicates how many stream lines
    x = np.arange(0, 5, 0.1) #third value indicates smoothness of x component of streamline
    y = np.arange(0, 5, 0.1) #smoothness of y
    x, y = np.meshgrid(x, y) #mix em

    C=delta*x-gamma*np.log(abs(x))-alpha*np.log(abs(y))+beta*y #The streamline equation for lotka-volterra

    #ax.contour(x, y, C, levels, cmap = 'prism') #plot it 

    levels = np.arange(-50,50,5)

    C = alpha*x - beta * x * y - delta * x * y - gamma *y 

    ax.set_ylabel('Predators over time')
    ax.set_xlabel('Prey over time')

    #ax.contour(x, y, C, levels) #plot it 

    plt.show()


def main():
    LotkaGraph()
    LotkaVectorField()


main() # Call the driver to get the results
