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
        self.stateDimn = dynamics.stateDimn
        self.inputDimn = dynamics.inputDimn
        self.mean = mean
        self.sd = sd
        
    def get_state(self):
        """
        Returns a potentially noisy observation of the system state
        """
        if self.mean or self.sd:
            #return an observation of the vector with noise
            return self.dynamics.get_state() + np.random.normal(self.mean, self.sd, (self.stateDimn, 1))
        return self.dynamics.get_state()
    
class EgoTurtlebotObserver(StateObserver):
    def __init__(self, dynamics, mean, sd, index):
        """
        Init function for a state observer for a single turtlebot within a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object for the entire turtlebot system
            mean (float): Mean for gaussian noise. Defaults to None.
            sd (float): standard deviation for gaussian noise. Defaults to None.
            index (Integer): index of the turtlebot in the system
        """
        #initialize the super class
        super().__init__(dynamics, mean, sd)

        #store the index of the turtlebot
        self.index = index
    
    def get_state(self):
        """
        Returns a potentially noisy measurement of the state vector of the ith turtlebot
        Returns:
            3x1 numpy array, observed state vector of the ith turtlebot in the system (zero indexed)
        """
        return super().get_state()[3*self.index : 3*self.index + 3].reshape((3, 1))
    
    def get_vel(self):
        """
        Returns a potentially noisy measurement of the derivative of the state vector of the ith turtlebot
        Inputs:
            None
        Returns:
            3x1 numpy array, observed derivative of the state vector of the ith turtlebot in the system (zero indexed)
        """
        #first, get the current input to the system of turtlebots
        u = self.dynamics.get_input()

        #now, get the noisy measurement of the entire state vector
        x = self.get_state()

        #to pass into the deriv function, augment x with zeros elsewhere
        x = np.vstack((np.zeros((self.index*3, 1)), x, np.zeros(((self.dynamics.N - 1 - self.index)*3, 1))))
        
        #calculate the derivative of the ith state vector using the noisy state measurement
        xDot = self.dynamics.deriv(x, u, 0) #pass in zero for the time (placeholder for time invar system)

        #slice out the derivative of the ith turtlebot and reshape
        return xDot[3*self.index : 3*self.index + 3].reshape((3, 1))
    
    def get_z(self):
        """
        Return the augmented input vector, z, of the system from the ith turtlebot.
        Inputs:
            None
        Returns:
            z (2x1 NumPy Array): augmented input vector of the ith turtlebot
        """
        #call the get z function from the system dynamics
        z = self.dynamics.get_z()

        #slice out the ith term
        return z[2*self.index : 2*self.index + 2].reshape((2, 1))
    
    
class ObserverManager:
    def __init__(self, dynamics, mean, sd):
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
            self.observerDict[i] = EgoTurtlebotObserver(dynamics, mean, sd, i)

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