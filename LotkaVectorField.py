import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

def LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t):
 R = np.zeros(len(t)) # Pre-allocate the memory for R
 F = np.zeros(len(t)) # Pre-allocate the memory for F

 R[0] = R0
 F[0] = F0

 for n in range(0,len(t)-1):
  dt = t[n+1] - t[n]
  R[n+1] = R[n]*(1 + alpha*dt - gamma*dt*F[n])
  F[n+1] = F[n]*(1 - beta*dt + delta*dt*R[n])
 return R,F

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

def LotkaVectorField():
    #define population constants
    a = 1; b = 1
    A = 1; B = 1

    # Create xy plane
    x = np.arange(0, 5, 0.5)
    y = np.arange(0, 5, 0.5)
    x, y = np.meshgrid(x, y)
    i=a*x-b*x*y; j=A*x*y-B*y
    fig, ax = plt.subplots()
    ax.quiver(x, y, i, j, pivot='mid')

    levels  = np.arange(-10,10,0.2)

    x = np.arange(0, 5, 0.1)
    y = np.arange(0, 5, 0.1)
    x, y = np.meshgrid(x, y)

    C=B*x-A*np.log(abs(x))-a*np.log(abs(y))+B*y

    ax.contour(x, y, C, levels)
    plt.show()


def main():
    LotkaVectorField()


main() # Call the driver to get the results
