#%% Import packages
import sys
import json
import pandas as pd
import numpy as np
import os
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy.matlib
from policy import *
import functools
from platypus.algorithms import NSGAIII
from platypus.core import Problem
from platypus.types import Real
from platypus.types import Binary
from platypus.operators import PCX
import time
sys.path.append("/Users/jenniferskerker/Documents/GradSchool/Courses/Winter 2023/CS 238/Project/Code")
from res_model import ResModel


#%% Try creating instance of class Model
# inputs
Yrs = 30
Months = 12

# demands and set params
DemandYr = np.full((12,), 20)
Res_cap = 100
Res_init = 100
# set policies
marketMax = 10
marketThresh = 0.7
MarketCostH2O = 100
DesalCostsH2O = np.array([1000, 1, 1])
# cost to build each unit of capacity, cost to produce 1 unit water, cost factor for choosing to/not to use water
DesalUnitSize = 2
DesalMaxUnits = 10
DesalTime2Build = 1 # yrs
k = 1
m = 3
n = k + m + 1

# Create fake inflow data
TS = 50
InflowFakeData = np.zeros(((Yrs+1)*Months, 1))
InflowYrData = np.array([10, 20, 30, 100, 0, 0, 0, 0, 0, 0, 10, 10])
# one time series-- need to figure out how to adapt for multiple time series
inflowBaseline = np.reshape(np.matlib.repmat(InflowYrData, Yrs+1, 1), newshape=(Yrs+1)*Months, order='C')
for i in range(TS):
    inflowPerturb = np.random.random(size=(Yrs+1)*Months)+0.5
    newData = np.multiply(inflowBaseline, inflowPerturb).reshape((Yrs+1)*Months, 1)
    InflowFakeData = np.concatenate([InflowFakeData, newData], axis=1)
#InflowFakeData = InflowFakeData[:, 1:Yrs+1] # get rid of column of zeros from initialization

# demands and set params
DemandYrData = np.full((12,), 20)

#%% run simulation for desal discrete version
# create instance of model class
k = 1
m = 5
n = k + m + 1
ModelDesalTest = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)
marketThresh = 0.8
cost, shortage = ModelDesalTest.run_simulation_discreteDesal([marketThresh]) #, print_output=True
ModelDesalTest.plot_simulation(1)

#%% run simulation for desal continuous version
# create instance of model class
k = 1
m = 5
n = k + m + 1
ModelDesalCont = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)
marketMax = 10
cost, shortage = ModelDesalCont.run_simulation_contDesal([marketMax]) #, print_output=True
ModelDesalCont.plot_simulation([0])

#%% run simulation for desal continuous version- 2 DVs
# create instance of model class
k = 2
m = 5
n = k + m + 1
ModelDesalCont2DV = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)
marketMax = 10
marketThresh = 0.7
cost, shortage = ModelDesalCont2DV.run_simulation_contDesal2DVs([marketMax, marketThresh]) #, print_output=True
ModelDesalCont.plot_simulation([0])

#%% Try running optimization- Desal Discrete
# input variables
k = 1
m = 5
n = k + m + 1
# initialize instance of class
modelOptDesal = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)

#marketMin = 1
#marketMax = 10
threshMin = 0
threshMax = 1

# optimization
start = time.perf_counter()
# Problem(number of decisions, number of objectives)
problem = Problem(1, 2)
#problem.types[0] = Binary(1) # feasible range of market water yes/no
problem.types[0] = Real(threshMin, threshMax) # feasible range of marketMax
#problem.types[1] = Real(threshMin, threshMax) # feasible range of marketThresh
problem.directions[1] = Problem.MINIMIZE # default for 0th is minimize
problem.function = functools.partial(modelOptDesal.run_simulation_discreteDesal)

algorithm = NSGAIII(problem, divisions_outer=100)

# optimize the problem using 10000 function evaluations
algorithm.run(1000)
end = time.perf_counter()
seconds = end-start
print('time: ', seconds, ' seconds') # print time it takes to run this section

# Plot results
# convert data to numpy first..
obj_dd = np.array([s.objectives for s in algorithm.result])
x_dd = np.array([s.variables for s in algorithm.result])

plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
# plt.contour(X,Y,costs.T, 50, cmap=plt.cm.cool)
# plt.contour(X,Y,rels.T, 50, cmap=plt.cm.Reds)
plt.scatter(np.arange(0, len(x_dd)), x_dd[:,0], s=30, color='k') #zorder=5
#plt.xlabel('Market water multiplier')
#plt.ylabel('Market water threshold')
plt.title('Optimal Decision Variable Values')

plt.subplot(1,2,2)
plt.scatter(obj_dd[:,0],obj_dd[:,1], s=30, color='k')
plt.xlabel('Cost ($)')
plt.ylabel('Shortage (units of H2O)')
plt.title('Pareto Frontier')

plt.show()

#%% Try running optimization- Desal Continuous
# input variables
k = 1
m = 5
n = k + m + 1
# initialize instance of class
modelOptDesal2 = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)

marketMin = 1
marketMax = 40

# optimization
start = time.perf_counter()
# Problem(number of decisions, number of objectives)
problem = Problem(1, 2)
#problem.types[0] = Binary(1) # feasible range of market water yes/no
problem.types[0] = Real(marketMin, marketMax) # feasible range of marketMax
#problem.types[1] = Real(threshMin, threshMax) # feasible range of marketThresh
problem.directions[1] = Problem.MINIMIZE # default for 0th is minimize
problem.function = functools.partial(modelOptDesal2.run_simulation_contDesal)

algorithm = NSGAIII(problem, divisions_outer=100)

# optimize the problem using 10000 function evaluations
algorithm.run(1000)
end = time.perf_counter()
seconds = end-start
print('time: ', seconds, ' seconds') # print time it takes to run this section

# Plot results
# convert data to numpy first..
obj_dc = np.array([s.objectives for s in algorithm.result])
x_dc = np.array([s.variables for s in algorithm.result])

plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
# plt.contour(X,Y,costs.T, 50, cmap=plt.cm.cool)
# plt.contour(X,Y,rels.T, 50, cmap=plt.cm.Reds)
plt.scatter(np.arange(0, len(x_dc)), x_dc[:,0], s=30, color='k') #zorder=5
#plt.xlabel('Market water multiplier')
#plt.ylabel('Market water threshold')
plt.title('Optimal Decision Variable Values')

plt.subplot(1,2,2)
plt.scatter(obj_dc[:,0],obj_dc[:,1], s=30, color='k')
plt.xlabel('Cost ($)')
plt.ylabel('Shortage (units of H2O)')
plt.title('Pareto Frontier')

plt.show()

#%% Try running optimization- Desal Continuous- 2 DVs
# input variables
k = 2
m = 5
n = k + m + 1
# initialize instance of class
modelOptDesal2DVs = ResModel(Yrs, Months, InflowFakeData, InflowYrData, DemandYrData, Res_cap, Res_init, MarketCostH2O,
                  DesalCostsH2O, DesalUnitSize, DesalMaxUnits, DesalTime2Build, n, m, k)

marketMin = 1
marketMax = 40
threshMin = 0
threshMax = 1

# optimization
start = time.perf_counter()
# Problem(number of decisions, number of objectives)
problem = Problem(2, 2)
#problem.types[0] = Binary(1) # feasible range of market water yes/no
problem.types[0] = Real(marketMin, marketMax) # feasible range of marketMax
problem.types[1] = Real(threshMin, threshMax) # feasible range of marketThresh
problem.directions[1] = Problem.MINIMIZE # default for 0th is minimize
problem.function = functools.partial(modelOptDesal2DVs.run_simulation_contDesal2DVs)

algorithm = NSGAIII(problem, divisions_outer=100)

# optimize the problem using 10000 function evaluations
algorithm.run(4000)
end = time.perf_counter()
seconds = end-start
print('time: ', seconds, ' seconds') # print time it takes to run this section

# Plot results
# convert data to numpy first..
obj_2dv_funcevals = np.array([s.objectives for s in algorithm.result])
x_2dv_funcevals = np.array([s.variables for s in algorithm.result])

plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
# plt.contour(X,Y,costs.T, 50, cmap=plt.cm.cool)
# plt.contour(X,Y,rels.T, 50, cmap=plt.cm.Reds)
plt.scatter(x_2dv_funcevals[:,0], x_2dv_funcevals[:,1], s=30, color='k') #zorder=5
plt.xlabel('Desal max capacity multiplier')
plt.ylabel('Desal water use policy threshold')
plt.title('Optimal Policy Variable Values')

plt.subplot(1,2,2)
plt.scatter(obj_2dv_funcevals[:,0],obj_2dv_funcevals[:,1], s=30, color='k')
plt.xlabel('Cost ($)')
plt.ylabel('Shortage (units of H2O)')
plt.title('Pareto Frontier')

#plt.show()
plt.savefig('Fig3_Formulation3.jpg', format='jpg')

#%% Pareto frontier plot for all 3 framings
plt.figure(figsize=(6, 4.5))
plt.scatter(obj_dd[:,0],obj_dd[:,1], s=30, color='b', label='Desal Discrete, 1 DV')
plt.scatter(obj_dc[:,0],obj_dc[:,1], s=30, color='r', label='Desal Continuous, 1 DV')
plt.scatter(obj_2dv[:,0], obj_2dv[:,1], s=30, color='g', label='Desal Continuous, 2 DVs') # run the 2DV version w/ 1000 function evaluations to get this data
plt.xlabel('Cost ($)')
plt.ylabel('Shortages (units)')
plt.legend()
#plt.title('Pareto Frontier')
#plt.show()
plt.savefig('ParetoFrontier.jpg', format='jpg')

#%% Save data to numpy file just in case
obj_dd.tofile('obj_desaldiscrete.csv', sep = ',')
obj_dc.tofile('obj_desalcont.csv', sep = ',')
obj_2dv.tofile('obj_desalcont2DV.csv', sep = ',')
x_dd.tofile('x_desaldiscrete.csv', sep = ',')
x_dc.tofile('x_desalcont.csv', sep = ',')
x_2dv.tofile('x_desalcont2DV.csv', sep = ',')

#%% Save data to numpy file from 2nd optimization
obj_2dv_funcevals.tofile('obj_desalcont2DV_4000.csv', sep = ',')
x_2dv_funcevals.tofile('x_desalcont2DV_4000.csv', sep = ',')

#%% Re-import data
# formulation 1: desal discrete, 1 DV
dd_import = np.loadtxt("obj_desaldiscrete.csv",delimiter=",")
obj_dd = np.reshape(dd_import, (104,2))

# formulation 2: desal continuous, 1 DV
dc_import = np.loadtxt("obj_desalcont.csv",delimiter=",")
obj_dc = np.reshape(dc_import, (104,2))

# formulation 3: desal continuous, 2 DVs
dc2dv_import = np.loadtxt("obj_desalcont2DV.csv",delimiter=",")
obj_2dv = np.reshape(dc2dv_import, (104,2))


#%% Time series of fake inflow data
plt.figure(figsize=(9, 4.5))
rng = random.randint(100, size=(5))
plt.plot(InflowFakeData[12:371,rng])
plt.xlabel('Time (months)')
plt.ylabel('Streamflow (units H2O)')
#plt.title('Sample of streamflow time series data')
#plt.show()
plt.savefig('fig1_streamflow.jpg', format='jpg')