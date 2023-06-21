import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Dynamics:
    """
    Skeleton class for system dynamics
    Includes methods for returning state derivatives, plots, and animations
    """
    def __init__(self, x0, singleStateDimn, singleInputDimn, f, N = 1):
        """
        Initialize a dynamics object
        Args:
            x0 (N*stateDimn x 1 numpy array): (x01, x02, ..., x0N) Initial condition state vector for all N agents
            stateDimn (int): dimension of state vector for a single agent
            inputDimn (int): dimension of input vector for a single agent
            f (python function): dynamics function in xDot = f(x, u, t) -> This is for a SINGLE instance
            N (int): Number of "agents" in the system, i.e. how many instances we want running in parallel
        """    
        #store the dynamics function
        self._f = f

        #store the number of agents
        self.N = N

        #store the state and input dimensions for the extended system
        self.singleStateDimn = singleStateDimn
        self.singleInputDimn = singleInputDimn
        self.sysStateDimn = singleStateDimn * N
        self.sysInputDimn = singleInputDimn * N

        #store the state and input
        self._x = x0
        self._u = np.zeros((self.sysInputDimn, 1))

    def get_input(self):
        """
        Retrieve the input to the system
        """
        return self._u
    
    def get_state(self):
        """
        Retrieve the state vector
        """
        return self._x
        
    def deriv(self, x, u, t):
        """
        Returns the derivative of the state vector for the extended system
        Args:
            x (sysStateDimn x 1 numpy array): current state vector at time t
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
        Returns:
            xDot: state_dimn x 1 derivative of the state vector
        """
        #assemble the derivative vector for the entire system
        xDot = np.zeros((self.sysStateDimn, 1))
        for i in range(self.N):
            #define slicing indices
            stateSlice0 = self.singleStateDimn * i
            stateSlice1 = self.singleStateDimn * (i + 1)
            inputSlice0 = self.singleInputDimn * i
            inputSlice1 = self.singleInputDimn * (i + 1)

            #extract the state and input of the ith agent
            xi = x[stateSlice0  : stateSlice1, 0].reshape((self.singleStateDimn, 1))
            ui = u[inputSlice0 : inputSlice1, 0].reshape((self.singleInputDimn, 1))
            
            #compute the derivative of the ith agent's state vector
            xDot[stateSlice0 : stateSlice1, 0] = self._f(xi, ui, t).reshape((self.singleStateDimn, ))

        #return the assembled derivative vector
        return xDot
    
    def integrate(self, u, t, dt):
        """
        Integrates system dynamics using Euler integration
        Args:
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (sysStateDimn x 1 numpy array): state vector after integrating
        """
        #integrate starting at x
        self._x = self.get_state() + self.deriv(self.get_state(), u, t)*dt

        #update the input parameter
        self._u = u

        #return integrated state vector
        return self._x
    
    def show_plots(self, xData, uData, tData):
        """
        Function to show plots specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
        """
        #Plot each state variable in time
        fig, axs = plt.subplots(self.singleStateDimn + self.singleInputDimn)
        fig.suptitle('Evolution of States and Inputs in Time')
        xlabel = 'Time (s)'
        stateLabels = ["X" + str(i + 1) for i in range(self.singleStateDimn)]
        inputLabels = ["U" + str(i + 1) for i in range(self.singleInputDimn)]
        #plot the states for each agent
        for j in range(self.N):
            n = 0 #index in the subplot
            for i in range(self.singleStateDimn):
                axs[n].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], xData[self.singleStateDimn*j + i, :].tolist()[0:-1])
                axs[n].set(ylabel=stateLabels[n]) #pull labels from the list above
                axs[n].grid()
                n += 1
            #plot the inputs
            for i in range(self.singleInputDimn):
                axs[i+self.singleStateDimn].plot(tData.reshape((tData.shape[1], )).tolist()[0:-1], uData[self.singleInputDimn*j + i, :].tolist()[0:-1])
                axs[i+self.singleStateDimn].set(ylabel=inputLabels[i])
                axs[i+self.singleStateDimn].grid()
        axs[self.singleStateDimn + self.singleInputDimn - 1].set(xlabel = xlabel)
        legendList = ["Agent " + str(i) for i in range(self.N)]
        plt.legend(legendList)
        plt.show()
    
    def show_animation(self, x, u, t):
        """
        Function to play animations specific to this dynamic system.
        Args:
            x ((sysStateDimn x N) numpy array): history of N states to plot
            u ((sysInputDimn x N) numpy array): history of N inputs to plot
            t ((1 x N) numpy array): history of N times associated with x and u
        """
        pass
    
    
"""
**********************************
PLACE YOUR DYNAMICS FUNCTIONS HERE
**********************************
"""