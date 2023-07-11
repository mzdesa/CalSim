import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

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

        #set the plotting style with seaborn
        sns.set_theme()
        sns.set_context("paper")

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
    
    def euler_integrate(self, u, t, dt):
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
    
    def rk4_integrate(self, u, t, dt):
        """
        Integrates system dynamics using RK4 integration
        Args:
            u (sysInputDimn x 1 numpy array): current input vector at time t
            t (float): current time with respect to simulation start
            dt (float): time step for integration
        Returns:
            x (sysStateDimn x 1 numpy array): state vector after integrating
        """
        #get current deterministic state
        x = self.get_state()

        #evaluate RK4 constants
        k1 = self.deriv(x, u, t)
        k2 = self.deriv(x + dt*k1/2, u, t + dt/2)
        k3 = self.deriv(x + dt*k2/2, u, t + dt/2)
        k4 = self.deriv(x + dt*k3, u, t + dt)

        #update the state parameter
        self._x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        #update the input parameter
        self._u = u

        #return the integrated state vector
        return self._x
    
    def integrate(self, u, t, dt, integrator = "rk4"):
        """
        Integrate dynamics with either rk4 or euler integration
        Choose either "rk4" or "euler" to select integrator.
        """
        if integrator == "rk4":
            return self.rk4_integrate(u, t, dt)
        else:
            return self.euler_integrate(u, t, dt)
    
    def show_plots(self, xData, uData, tData, stateLabels = None, inputLabels = None):
        """
        Function to show plots specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
            stateLabels ((singleStateDimn) length list of strings): Optional custom labels for the state plots
            inputLabels ((singleInputDimn) length list of strings): Optional custom labels for the input plots
        """
        #Plot each state variable in time
        fig, axs = plt.subplots(self.singleStateDimn + self.singleInputDimn)
        fig.suptitle('Evolution of States and Inputs in Time')
        xlabel = 'Time (s)'

        #set state and input labels
        if stateLabels is None:
            stateLabels = ["X" + str(i + 1) for i in range(self.singleStateDimn)]
        if inputLabels is None:
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
    
    def show_animation(self, xData, uData, tData, axis_lims, labels, anim_point, anim_line = None, animate = True):
        """
        Function to play animations specific to this dynamic system.
        Args:
            xData ((sysStateDimn x N) numpy array): history of N states to plot
            uData ((sysInputDimn x N) numpy array): history of N inputs to plot
            tData ((1 x N) numpy array): history of N times associated with x and u
            axis_lims (Python List, length 4): list of axis limits, ex: [-0.25, 5.25, -0.25, 5.25]
            labels (Python list, length 3): List of strings ["Xlabel", "Ylabel", "Title"] for plot
            anim_point (Python function): function to be called at an index i, returns x, y to be animated
            anim_line (Python function): function to be called at an index i, returns line to be animated
        """
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        
        if animate:
            fig, ax = plt.subplots()
            # set the axes limits
            ax.axis(axis_lims)
            # set equal aspect such that the circle is not shown as ellipse
            ax.set_aspect("equal")

            # create a set of points in the axes
            point, = ax.plot([],[], marker="o", linestyle='None')

            #check for anim_line
            if anim_line is not None:
                #define the line for the quadrotor
                line, = ax.plot([], [], 'o-', lw=2)

            def anim_func(i):
                #Call animate_func to get the x and y to plot
                x, y = anim_point(i)

                #set the data to points and return points
                point.set_data(x, y)

                #check for anim_line
                if anim_line is not None:
                    #call the line animation function
                    thisx, thisy = anim_line(i)
                    #set the data for the line
                    line.set_data(thisx, thisy)

                    #if we use a line, return line and point
                    return line, point
                else:
                    #if no line is present, just return the point
                    return point,

            num_frames = xData.shape[1]-1

            anim = animation.FuncAnimation(fig, anim_func, frames=num_frames, interval=1/FREQ*1000, blit=True)

            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(labels[2])
            plt.show()
    
    
"""
**********************************
PLACE YOUR DYNAMICS FUNCTIONS HERE
**********************************
"""

class DoubleIntegratorDyn(Dynamics):
    """
    Double Integrator System
    May initialize N integrators to run in parallel.
    """
    def __init__(self, x0, N = 1):
        #define the double integrator dynamics
        def double_integrator(x, u, t):
            """
            Double integrator dynamics
                x (2x1 NumPy array): state vector
                u (1x1 NumPy array): input vector
                t (float): current time
            """
            return np.array([[0, 1], [0, 0]]) @ x + np.array([[0, 1]]).T @ u
        super().__init__(x0, 2, 1, double_integrator, N = N)

class MSDRamp(Dynamics):
    """
    Mass-spring-damper system on a ramp.
    """
    def __init__(self, x0, m = 0.5, g = 9.81, k = 15, b = 0.5, theta = np.pi/6, N = 1):
        """
        Inputs:
            x0 (2x1 NumPy Array): Initial condition
            m (float): mass in kg
            g (float): acceleration due to gravity (m/s^2)
            k (float): spring constant (N/m)
            b (float): damping constant (N/(m/s))
            theta (float): angle of ramp
            N (int): number of agents
        """
        self.m = m
        self.g = g
        self.k = k
        self.b = b
        self.theta = theta

        def msd_ramp(x, u, t):
            """
            Mass spring damper ramp dynamics
            Inputs:
                x (2x1 NumPy array): current state vector
                u (1x1 NumPy Array): force applied to mass (typically 0)
                t (float): current time in simulation
            """
            return np.array([[x[1, 0]], [u[0, 0]/m -k/m * x[0, 0] - b/m * x[1, 0] - g*np.sin(theta)]])
        
        #call the init function on the MSD ramp system
        super().__init__(x0, 2, 1, msd_ramp, N = N)

class DoublePendulum(Dynamics):
    """
    Double pendulum dynamics
    """
    def __init__(self, x0, m = 0.5, I = 0.5, L = 1, g = 9.81, N = 1):
        """
        Inputs:
            x0: initial condition [q1, q2, q1Dot, q2Dot]
            m: mass of joints
            I: inertia of joints
            L: length of joints
            g: gravitational acceleration
        """
        #store the system parameters
        self.m = m
        self.I = I
        self.L = L
        self.g = g

        def double_pend(x, u, t):
            """
            Double pendulum dynamics
            Includes a torque input at each joint.
            """
            #extract the states
            q1, q2, q1Dot, q2Dot = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

            #compute the inertia matrix
            M = np.array([[2*I + 1.5*m*L**2 + m*L**2*np.cos(q2), I+0.25*m*L**2+0.5*m*L**2*np.cos(q2)],
                          [I+0.25*m*L**2+0.5*m*L**2*np.cos(q2), I+0.25*m*L**2]])
            #compute C
            C = np.array([[0, -m*L**2*q1Dot*np.sin(q2)-0.5*m*L**2*q2Dot*np.sin(q2)], [0.5*m*L**2*q1Dot*np.sin(q2), 0]])

            #compute N
            N = np.array([[0.5*m*g*L*(np.sin(q1+q2)+3*np.sin(q1)), 0.5*m*g*L*np.sin(q1+q2)]]).T

            #compute qDDot
            qDot = np.array([[q1Dot, q2Dot]]).T
            qDDot = np.linalg.inv(M)@(u - C@qDot - N)

            #return [qDot, qDDot]
            return np.vstack((qDot, qDDot))
        
        #call the init function on the double pendulum system
        super().__init__(x0, 4, 2, double_pend, N = N)


class FlywheelPendulum(Dynamics):
    """
    Pendulum driven by a flywheel
    """

class MarsProbe(Dynamics):
    """
    Dynamics of a particle falling through the atmosphere of mars.
    """
    def __init__(self, x0, m = 572.743, Cd = 1.46, A = 5.5155, ftMax = 9806.65, N = 1):
        """
        Init function for mars probe.
        
        Mission Phases and Probe Parameters:
        - https://ntrs.nasa.gov/api/citations/20080034645/downloads/20080034645.pdf
        - Enter atmosphere at 5600 m/s
        - Chute deployment occurs at Mach 1.65, 12.9 km above surface
        - Touchdown occurs at 0.7 m/s
        
        Inputs:
            x0 (2x1 NumPy Array): position vector of the system
            m (float): mass of the probe
            Cd (float): drag coefficient of probe 
                NOTE: Cd for a parachute ~0.5 -> add this to Cd for just probe. Chute area ~200m^2.
                      https://downloads.spj.sciencemag.org/space/2022/9805457.pdf
            A (float): surface area for drag calculation
            ftMax (float): maximum thrust force of probe
            N (int): number of probes to simulate
        """
        #properties of probe
        self.m = m
        self.Cd = Cd
        self.ftMax = ftMax

        #properties of Mars
        self.rho = 0.02 #density of atmosphere (kg/m^3)
        self.M = 6.41693 * 10**23 #mass of Mars (kg)
        self.R = 3390 * 10**3 #radius of Mars (m)
        self.G = 6.6743 * 10**(-11) #universal gravitational constant

        def probe_dyn(x, u, t):
            """
            Mars probe dynamics.
            Inputs:
                x (2x1 NumPy Array): [y, yDot] for y distance to the center of mars
                u (1x1 NumPy Array): thrust force of the probe
                t (float): time
            """
            y = x[0, 0]
            yDot = x[1, 0]
            yDDot = -self.G * self.M / (y**2) + 0.5*self.rho*self.Cd*yDot**2/self.m + u[0, 0]/m
            return np.array([[yDot, yDDot]]).T
        
        #call the init function on the probe pendulum system
        super().__init__(x0, 2, 1, probe_dyn, N = N)

class TurtlebotSysDyn(Dynamics):
    """
    System of N Turtlebots
    """
    def __init__(self, x0, N = 1, rTurtlebot = 0.15):
        """
        Init function for a system of N turtlebots.
        Args:
            x0 (NumPy Array): (x1, y1, phi1, ..., xN, yN, phiN) initial condition for all N turtlebots
            N (Int, optional): number of turtlebots in the system
            rTurtlebot (float): radius of the turtlebots in the system
        """

        #define the turtlebot dynamics
        def f_turtlebot(x, u, t):
            #extract the orientation angle of the Nth turtlebot
            PHI = x[2, 0]
            return np.array([[np.cos(PHI), 0], [np.sin(PHI), 0], [0, 1]])@u

        #call the super init function to create a turtlebot system
        super().__init__(x0, 3, 2, f_turtlebot, N)

        #store a copy of the augmented input vector for feedback linearization
        self._z = np.zeros((self.sysInputDimn, 1))

        #store the turtlebot radius
        self.rTurtlebot = rTurtlebot 

    def set_z(self, z, i):
        """
        Function to set the value of z, the augmented input vctor.
        Inputs:
            z ((2N x 1) NumPy Array): Augmented input vector
            i (int): index we wish to place the updated z at
        """
        #store in class attribute
        self._z[2*i : 2*i + 2, 0] = z.reshape((2, ))
    
    def get_z(self):
        """
        Function to return the augmented input vector, z, at any point.
        """
        #retrieve and return augmented input vector
        return self._z
    
    def show_plots(self, xData, uData, tData):
        #Plot the spatial trajectory of the turtlebots
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        #iterate over each turtlebot state vector
        for j in range(self.N):
            xCoords = xData[3*j, :].tolist() #extract all of the velocity data to plot on the y axis
            yCoords = xData[3*j+1, :].tolist() #remove the last point, get artefacting for some reason
            ax.plot(xCoords[0:-1], yCoords[0:-1])
        
        legendList = ["Agent " + str(i) for i in range(self.N)]
        plt.legend(legendList)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Positions of Turtlebots in Space")
        plt.show()

        #call the super plots with custom labels to show the individual states
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Phi (rad)']
        inputLabels = ["V (m/s)", "Omega (rad/s)"]
        super().show_plots(xData, uData, tData, stateLabels, inputLabels)
    
    def show_animation(self, xData, uData, tData, animate = True):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
        """
        #define animation function to return the x, y points to be animated
        def anim_point(i):
            x = []
            y = []
            #get the x, y data associated with each turtlebot
            for j in range(self.N):
                x.append(xData[3*j, i])
                y.append(xData[3*j+1, i])
            return x, y
    
        #define axis limits
        axis_lims = [-0.25, 5.25, -0.25, 5.25]

        #define labels
        axis_labels = ["X Position (m)", "Y Position (m)", "Positions of Turtlebots in Space"]

        #call the super() animation function
        super().show_animation(xData, uData, tData, axis_lims, axis_labels, anim_point, anim_line = None, animate = animate)



class PlanarQrotor(Dynamics):
    def __init__(self, x0 = np.zeros((8, 1)), m = 0.92, Ixx = 0.0023, l = 0.15, N = 1):
        """
        Init function for a Planar quadrotor system.
        State Vector: X = [x, y, z, theta, x_dot, y_dot, z_dot, theta_dot]
        Input Vector: U = [F, M]
        
        Args:
            x0 ((8 x 1) NumPy Array): initial state (x, y, z, theta, x_dot, y_dot, z_dot, theta_dot)
            m (float): mass of quadrotor in kg
            Ixx (float): moment of inertia about x axis of quadrotor
            l (float): length of one arm of quadrotor
            N (int): number of quadrotors
        """
        #store physical parameters
        self._m = m
        self._Ixx = Ixx
        self._g = 9.81
        self._l = l

        #define quadrotor dynamics
        def quadrotor_dyn(X, U, t):
            """
            Returns the derivative of the state vector
            Args:
                X (8 x 1 numpy array): current state vector at time t
                U (2 x 1 numpy array): current input vector at time t
                t (float): current time with respect to simulation start
            Returns:
                xDot: state_dimn x 1 derivative of the state vector
            """
            #unpack the input vector
            F, M = U[0, 0], U[1, 0]
            F = max(0, F) #Cut off force at zero to prevent negative thrust
            
            #unpack the state vector
            x_dot, y_dot, z_dot = X[4, 0], X[5, 0], X[6, 0] #velocities
            theta, theta_dot = X[3, 0], X[7, 0] #orientations
            
            #calculate the second time derivatives of each
            x_ddot = 0
            y_ddot = (-np.sin(theta)*F)/self._m
            z_ddot = (np.cos(theta)*F - self._m*self._g)/self._m
            theta_ddot = M/self._Ixx
            
            #construct and return state vector        
            return np.array([[x_dot, y_dot, z_dot, theta_dot, x_ddot, y_ddot, z_ddot, theta_ddot]]).T
        
        #call the super init function
        super().__init__(x0, 8, 2, quadrotor_dyn, N = N)

    def show_plots(self, xData, uData, tData):
        #Plot each state variable in time
        stateLabels = ['X Pos (m)', 'Y Pos (m)', 'Z Pos (m)', 'Theta (rad)', 'X Vel (m/s)', 'Y Vel (m/s)', 'Z Vel (m/s)', 'Angular Vel (rad/s)']
        inputLabels = ['Force (N)', 'Moment (N*m)']
        super().show_plots(xData, uData, tData, stateLabels, inputLabels)
        
    def show_animation(self, xData, uData, tData, animate = True):
        """
        Shows the animation and visualization of data for this system.
        Args:
            xData (stateDimn x N Numpy array): state vector history array
            u (inputDimn x N numpy array): input vector history array
            t (1 x N numpy array): time history
            animate (bool, optional): Whether to generate animation or not. Defaults to True.
        """
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        L = self._l #quadrotor arm length

        #define point animation function
        def anim_point(i):
            y = xData[1, i]
            z = xData[2, i]
            return y, z
        
        #define line animation function
        def anim_line(i):
            y = xData[1, i]
            z = xData[2, i]

            #draw the quadrotor line body
            theta = xData[3, i]
            x1 = y + L*np.cos(theta)
            x2 = y - L*np.cos(theta)
            y1 = z + L*np.sin(theta)
            y2 = z - L*np.sin(theta)
            thisx = [x1, x2]
            thisy = [y1, y2]
            return thisx, thisy
        
        #define axis limits
        axis_lims = [-1, 2.5, 0, 2.5]

        #define plot labels
        labels = ["Y Position (m)", "Z Position (m)", "Position of Drone in Space"]

        #call the super animation function
        super().show_animation(xData, uData, tData, axis_lims, labels, anim_point, anim_line, animate = animate)