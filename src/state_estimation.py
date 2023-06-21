import numpy as np

class StateObserver:
    def __init__(self, dynamics, mean = None, sd = None):
        """
        Init function for state observer

        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        self.dynamics = dynamics
        self.sysStateDimn = dynamics.sysStateDimn
        self.sysInputDimn = dynamics.sysInputDimn
        self.mean = mean
        self.sd = sd
        
    def get_state(self):
        """
        Returns a potentially noisy observation of the system state
        """
        if self.mean or self.sd:
            #return an observation of the vector with noise
            return self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.sysStateDimn, 1))
        return self.dynamics.get_state()
    
class EgoObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for a state observer for a single agent within a system of N agents
        Args:
            dynamics (Dynamics): Dynamics object for the entir system
            mean (float): Mean for gaussian noise. Defaults to None.
            sd (float): standard deviation for gaussian noise. Defaults to None.
            index (int): index of the agent in the system
        """
        #initialize the super class
        super().__init__(dynamics, mean, sd)

        #store the index of the agent
        self.index = index

        #store the state dimension of an individual agent
        self.singleStateDimn = dynamics.singleStateDimn
        self.singleInputDimn = dynamics.singleInputDimn
    
    def get_state(self):
        """
        Returns a potentially noisy measurement of the state vector of the ith turtlebot
        Returns:
            (Dynamics.singleStateDimn x 1 NumPy array), observed state vector of the ith turtlebot in the system (zero indexed)
        """
        return super().get_state()[self.singleStateDimn*self.index : self.singleStateDimn*(self.index + 1)].reshape((self.singleStateDimn, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of the derivative of the state vector of the ith agent
        Returns:
            (Dynamics.singleStateDimn x 1 NumPy array): observed derivative of the state vector of the ith turtlebot in the system (zero indexed)
        """
        #first, get the current input to the system
        u = self.dynamics.get_input()

        #now, get the noisy measurement of the entire state vector
        x = self.get_state()
        
        #calculate the derivative of the ith state vector using the noisy state measurement
        return self.dynamics._f(x, u, 0) #pass in zero for the time (placeholder for time invar system)
    
    
class ObserverManager:
    def __init__(self, dynamics, mean = None, sd = None):
        """
        Managerial class to manage the observers for a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object instance
            mean (float, optional): Mean for gaussian noise. Defaults to None.
            sd (float, optional): standard deviation for gaussian noise. Defaults to None.
        """
        #store the input parameters
        self.dynamics = dynamics
        self.mean = mean
        self.sd = sd

        #create an observer dictionary storing N observer instances
        self.observerDict = {}

        #create N observer objects
        for i in range(self.dynamics.N):
            #create an observer with index i
            self.observerDict[i] = EgoObserver(dynamics, mean, sd, i)

    def get_observer_i(self, i):
        """
        Function to retrieve the ith observer object for the turtlebot
        Inputs:
            i (integet): index of the turtlebot whose observer we'd like to retrieve
        """
        return self.observerDict[i]
    
    def get_state(self):
        """
        Returns a potentially noisy observation of the *entire* system state (vector for all N bots)
        """
        #get each individual observer state
        xHatList = []
        for i in range(self.dynamics.N):
            #call get state from the ith observer
            xHatList.append(self.get_observer_i(i).get_state())

        #vstack the individual observer states
        return np.vstack(xHatList)