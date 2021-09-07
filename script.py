import numpy as np
import matplotlib.pyplot as plt

# alpha is the natural growing rate of rabbits, when there's no fox
# beta is the natural dying rate of rabbits, due to predation
# gamma is the natural dying rate of fox, when there's no rabbit
# delta is the factor describing how many caught rabbits let create a new fox


def LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t):
    # Solves Lotka-Volterra equations for one prey and one predator species using
    # explicit Euler method.
    #  R0 and F0 are inputs and are the initial populations of each species
    #  alpha, beta, gamma, delta are inputs and problem parameters
    #  t is an input and 1D NumPy array of t values where we approximate y values.
    #  Time step at each iteration is given by t[n+1] - t[n].

    R = np.zeros(len(t))  # Pre-allocate the memory for R
    F = np.zeros(len(t))  # Pre-allocate the memory for F

    R[0] = R0
    F[0] = F0

    for n in range(0, len(t)-1):
        dt = t[n+1] - t[n]
        R[n+1] = R[n]*(1 + alpha*dt - gamma*dt*F[n])
        F[n+1] = F[n]*(1 - beta*dt + delta*dt*R[n])

    return R, F


def main():
    # Main driver to organize the code better
    t = np.linspace(0, 40, 1500)  # interval [0,40] with 3201 equispaced points
    # as you increase the number of points the
    # solution becomes more similar to the
    # reference solution on wikipedia

    # You should set the parameters below as in your problem
    # I am using the Baboon-Cheetah example from wikipedia
    alpha, beta, gamma, delta = 1.5, 0.4, 0.4, 0.1
    R0, F0 = 10, 10

    # Actually solve the problem
    R, F = LotkaVolterra_EEuler(R0, F0, alpha, beta, gamma, delta, t)

    # Plot the solution
    plt.plot(t, R, 'g.-', t, F, 'b-')
    plt.legend(['Baboon', 'Cheetah'])
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.title("Solution of Lotka-Volterra system using explicit Euler")
    plt.show()


if __name__ == "__main__":
    main()
