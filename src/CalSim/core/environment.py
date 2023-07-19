import numpy as np
from .controller import ControllerManager, Controller
from .state_estimation import ObserverManager

class Environment:
    def __init__(self, dynamics, controller = None, observer = None, obstacleManager = None, T = 10):
        """
        Initializes a simulation environment
        Args:
            dynamics (Dynamics): system dynamics object
            controller (Controller): system controller object
            observer (Observer): system state estimation object
            obstacleManager (ObstacleManager): obstacle objects
            T (Float): simulation time
        """
        #store system parameters
        self.dynamics = dynamics #store system dynamics
        self.obstacleManager = obstacleManager #store manager for any obstacles present

        #if observer and controller are none, create default objects
        if observer is not None:
            self.observer = observer
        else:
            #create a default noise-free observer manager
            self.observer = ObserverManager(dynamics)
        if controller is not None:
            self.controller = controller
        else:
            #create a default zero input controller manager (using the skeleton Controller class)
            self.controller = ControllerManager(self.observer, Controller, None, None, None)
        
        #define environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds 
        self.done = False
        
        #Store system state
        self.x = self.dynamics.get_state() #Actual state of the system
        self.x0 = self.x #store initial condition for use in reset
        self.xObsv = None #state as read by the observer
        self.ptCloud = None #point cloud state as read by vision
        
        #Define simulation parameters
        self.SIM_FREQ = 1000 #integration frequency in Hz
        self.CONTROL_FREQ = 50 #control frequency in Hz
        self.SIMS_PER_STEP = self.SIM_FREQ//self.CONTROL_FREQ
        self.TOTAL_SIM_TIME = T #total simulation time in s
        self.TOTAL_ITER = self.TOTAL_SIM_TIME*self.CONTROL_FREQ + 1 #total number of iterations
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.sysStateDimn, self.TOTAL_ITER))
        self.uHist = np.zeros((self.dynamics.sysInputDimn, self.TOTAL_ITER))
        self.tHist = np.zeros((1, self.TOTAL_ITER))
        
    def reset(self):
        """
        Reset the gym environment to its inital state.
        """
        #Reset gym environment parameters
        self.iter = 0 #number of iterations
        self.t = 0 #time in seconds
        self.done = False
        
        #Reset system state
        self.x = self.x0 #retrieves initial condiiton
        self.xObsv = None #reset observer state
        
        #Define history arrays
        self.xHist = np.zeros((self.dynamics.sysStateDimn, self.TOTAL_ITER))
        self.uHist = np.zeros((self.dynamics.sysInputDimn, self.TOTAL_ITER))
        self.tHist = np.zeros((1, self.TOTAL_ITER))

    def step(self):
        """
        Step the sim environment by one integration
        """
        #retrieve current state information
        self._get_observation() #updates the observer
        
        #solve for the control input using the observed state
        self.controller.eval_input(self.t)

        #update the deterministic system data, iterations, and history array
        self._update_data()
        
        #Zero order hold over the controller frequency and step dynamics
        for i in range(self.SIMS_PER_STEP):
            self.dynamics.integrate(self.controller.get_input(), self.t, 1/self.SIM_FREQ) #integrate dynamics
            self.t += 1/self.SIM_FREQ #increment the time
    
    def _update_data(self):
        """
        Update history arrays and deterministic state data
        """
        #append the input, time, and state to their history queues
        self.xHist[:, self.iter] = self.x.reshape((self.dynamics.sysStateDimn, ))
        self.uHist[:, self.iter] = (self.controller.get_input()).reshape((self.dynamics.sysInputDimn, ))
        self.tHist[:, self.iter] = self.t
        
        #update the actual state of the system
        self.x = self.dynamics.get_state()
        
        #update the number of iterations of the step function
        self.iter +=1
    
    def _get_observation(self):
        """
        Updates self.xObsv using the observer data
        Useful for debugging state information.
        """
        self.xObsv = self.observer.get_state()
        # print("current orientation: ", self.observer.get_orient())
    
    def _get_reward(self):
        """
        Calculate the total reward for ths system and update the reward parameter.
        Only implement for use in reinforcement learning.
        """
        return 0
    
    def _is_done(self):
        """
        Check if the simulation is complete
        Returns:
            boolean: whether or not the time has exceeded the total simulation time
        """
        #check if we have exceeded the total number of iterations
        if self.iter >= self.TOTAL_ITER:
            return True
        return False
    
    def run(self, N = 1, run_vis = True, verbose = False, obsManager = None):
        """
        Function to run the simulation N times
        Inputs:
            N (int): number of simulation examples to run
            run_vis (bool): run the visualization of the results
        Returns:
            self.xHist, self.uHist, self.tHist
        """
        #loop over an overall simulation N times
        for i in range(N):
            self.reset()
            print("Running Simulation.")
            while not self._is_done():
                if verbose:
                    print("Simulation Time Remaining: ", self.TOTAL_SIM_TIME - self.t)
                self.step() #step the environment while not done
            if run_vis:
                self.visualize() #render the result
        #return the history arrays
        return self.xHist, self.uHist, self.tHist
            
    def visualize(self):
        """
        Provide visualization of the environment
        Inputs:
            obsManager (Obstacle Manager, optional): manager to plot obstacles in the animation
        """
        self.dynamics.show_animation(self.xHist, self.uHist, self.tHist, obsManager = self.obstacleManager)
        self.dynamics.show_plots(self.xHist, self.uHist, self.tHist, obsManager = self.obstacleManager)