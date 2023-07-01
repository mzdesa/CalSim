#Import CalSim
import CalSim as cs
import numpy as np

#define an initial condition
q0 = np.array([[0, 1, 1, -1]]).T

#create two MSDs
dynamics = cs.MSDRamp(q0, N = 2)

#create a simulation environment
T = 10 #10 second simulation
env = cs.Environment(dynamics, None, None, T = T)

#run the simulation
env.run()