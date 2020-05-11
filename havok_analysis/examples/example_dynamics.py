'''

      Project: HAVOK analysis
         File: example_dynamics.py
 File Created: 02.03.2020
       Author: Pascal Schulz (pascal.schulz92@icloud.com)
-----
Last Modified: 11.05.2020 21:59:15
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

    This script contains a function that creates a Lorenz attractor
    
'''


import numpy as np

from scipy import integrate



# function that creates our dynamic system

######################
# Lorenz    #########
#####################
# define constant choose caotic numbers defined by lorenz
#sigma = 10.
#beta = 8./3
#rho = 28.
# choose initial condition
#x0 = [0, 1, 20]
# chose the time variable
# Beta = [10, 8/3, 28]
#dt = 0.001
#T = 50

######################
#    Lorenz96        #
######################
# These are our constants
#N = 30 # Number of variables
#F = 8  # Forcing
#T = 30.0
#dt = 0.01
# x0, Beta, dt, T
#x0 = F * np.ones(N)  # Initial state (equilibrium)
#x0[19] += 0.1  # Add small perturbation to 20th variable
#Beta = [F, N]

#####################
#    Van der Pol    #
#####################
#T = 10
#mu = 1
#dt = 0.001
#Beta = mu
#x0 = [1, 0]

################################################
class ExampleDynamics:
    def __init__(self, x0, Beta, dt, T):
        # define variable
        self.x0 = x0
        self.Beta = Beta
        self.dt = dt
        self.T = T
        self.timespan = np.arange(dt, T, dt)
        
     
    def get_lorenz(self):
        
        # check if beta has the right len for the lorenz system
        if len(self.Beta) != 3:
            raise ValueError('Beta needs sigma, beta and rho') 
    
        # save the constants in single variables        
        sigma, beta, rho = self.Beta
        
        t_eval = np.arange(start = 0, stop =  self.T, step = self.dt)

        # help function for the calculation of the dynamic system
        def lorenz(t0, x_y_z, sigma = sigma, beta = beta, rho = rho):
        
            '''
            Creates the lorenz system
    
            '''
            x, y, z = x_y_z
            return([
                sigma* (y - x),
                x * (rho - z) - y,
                x * y - beta * z
                ])
        
        # help function for the calculation of the jacobian matrix
        def jac(t, state):
            """
            jacobian matrix of lorenz system
            """
            x, y, z = state
            jac = np.array([[-sigma, sigma, 0],
                            [rho, -1, -x],
                            [y, x, -beta]])
            return jac
    
        # solve for the trajectories
        #x_t = np.array(integrate.odeint(lorenz, self.x0, self.timespan))
    
        # new solver
        x_t = integrate.solve_ivp(lorenz, [0, self.T], self.x0,
                                  t_eval=t_eval, method='Radau',
                                  rtol=1e-6, atol=1e-12, jac=jac)
        
        ts = x_t.y.T
        time = x_t.t
        return(ts, time)
        
    
    def get_SimpleExample(self):
        
         # check if beta has the right len for our Simple Example
        if len(self.Beta) != 2:
            raise ValueError('Beta need to contain mu, lambda')
        
        # save the constants in single variabels
        mu, lamda = self.Beta
        
        t_eval = np.arange(start = 0, stop =  self.T, step = self.dt)
        
        # help function for the calculation of the non-linear system
        def simpleExample(t0, x_y, mu = mu, lamda = lamda):
            '''
            Create the simple example from the paper
            '''
            x, y = x_y
            return([
                mu * x,
                lamda * (y - x**2)
                ])
        
        # solve for the trajectories
        x_t = integrate.solve_ivp(simpleExample, [0, self.T], self.x0, t_eval = t_eval) 
        
        ts = x_t.y.T
        time = x_t.t
        
        return(ts, time)
    
    
    def get_lorenz96(self):
        
         # check if beta has the right len for the lorenz 96 system
        if len(self.Beta) != 2:
            raise ValueError('Beta needs to contain F and N')
        
        F, N = self.Beta
        
        # help function for the calculation of the dynamic system
        def lorenz96(t, x):
            #def lorenz96(x, t):
            """Lorenz 96 model."""
            # Compute state derivatives
            d = np.zeros(N)
            # First the 3 edge cases: i=1,2,N
            d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
            d[1] = (x[2] - x[N-1]) * x[0] - x[1]
            d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
            # Then the general case
            for i in range(2, N-1):
                d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
            # Add the forcing term
            d = d + F

            # Return the state derivatives
            return d

        t_eval = np.arange(0.0, self.T, self.dt)
        
        #x = odeint(lorenz96, x0, t)
        x_t = integrate.solve_ivp(lorenz96, [0, self.T], self.x0,
                                  t_eval=t_eval, method='RK45', rtol=1e-6,
                                  atol=1e-12)
        
        ts = x_t.y.T
        time = x_t.t
        return(ts, time)    
    
    
    def get_van_der_pol(self):
        
        # check if beta has the right len
        if not isinstance(self.Beta, int):
            raise ValueError('Beta needs sigma, beta and rho') 
        
        mu = self.Beta
        t_eval = np.arange(0.0, self.T, self.dt)
        
        # help function for the calculation of the dynamic system
        def vdp(t, z):
            x, y = z
            return [y, mu*(1 - x**2)*y - x]

        x_t = integrate.solve_ivp(vdp, [0, self.T], self.x0, t_eval = t_eval)
        
        ts = x_t.y.T
        time = x_t.t
        return(ts, time)

    def get_roessler(self):
        
        # check if beta has the right len
        if len(self.Beta) != 3:
            raise ValueError('Beta needs sigma, beta and rho') 
    
        # save the constants in single variables        
        a, b, c = self.Beta
        
        t_eval = np.arange(start = 0, stop =  self.T, step = self.dt)

        # help function for the calculation of the dynamic system
        def roesler(t0, x_y_z, a = a, b = b, c = c):
        
            '''
            Creates the lorenz system
    
            '''
            x, y, z = x_y_z
            return([
                - (y + z),
                x + a * y,
                b + (x - c) * z
                ])
        
        def jac(t, state):
            """
            jacobian matrix of lorenz system
            """
            x, y, z = state# X   Y   Z
            jac = np.array([[0, -1, -1], # ableitung 1 gleichung nach x, y, z
                            [1, a, 0],      # ableitung 2 gleichung nach x, y, z
                            [z, 0, x - c]])     # ableitung 3 gleichung nach x, y, z
            return jac
    

        x_t = integrate.solve_ivp(roesler, [0, self.T], self.x0, t_eval = t_eval,
                                  method='Radau', rtol=1e-6, atol=1e-12)
        
        ts = x_t.y.T
        time = x_t.t
        return(ts, time)

        
    
    
    
        
        
          
    