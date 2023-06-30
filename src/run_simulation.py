#Our dependencies
from environment import *
from dynamics import *
from controller import *
from state_estimation import *
from sim_utils import *


#define an initial condition
q0 = np.array([[0, 1, 1, -1]]).T

#create a dynamics object for a single double integrator
# dynamics = DoubleIntegratorDyn(q0, N = 2)
dynamics = MSDRamp(q0, N = 2)

#create a simulation environment
T = 10 #10 second simulation
env = Environment(dynamics, None, None, T = T)

#run the simulation
env.run()