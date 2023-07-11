#route directory to core
import sys
sys.path.append("..")

#import core as we would CalSim
import core as cs
import numpy as np
import matplotlib.pyplot as plt

#system initial condition
x0 = np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T #start the quadrotor at 1 M in the air

#create a dynamics object for the double integrator
dynamics = cs.PlanarQrotor(x0)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create a controller manager with a basic FF controller
controllerManager = cs.ControllerManager(observerManager, cs.FFController)

env = cs.Environment(dynamics, controllerManager, observerManager)
env.reset()

#run the simulation
env.run()