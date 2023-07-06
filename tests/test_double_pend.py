#Import CalSim
import CalSim as cs
import numpy as np

#define an initial condition
q0 = np.array([[0.1, 0.1, 0, 0]]).T

#create a double pendulum
dynamics = cs.DoublePendulum(q0, N = 1)

#create a simulation environment
T = 10 #10 second simulation
env = cs.Environment(dynamics, None, None, T = T)

#run the simulation
env.run()