#%% Import packages
#import sys
#import json
import pandas as pd
import numpy as np
#import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy.matlib
from policy import *
import functools
#from platypus.algorithms import NSGAIII
#from platypus.core import Problem
#from platypus.types import Real
#from platypus.types import Binary
#from platypus.operators import PCX
#import time


# Try creating a class
class ResModel:

    def __init__(self, yrs, months, inflow, inflowYr, demandYr, res_cap, res_init, marketCostH2O, desalCostsH2O, desalUnitSize,
                 desalMaxUnits, desalTime2Build, N, M, K):
        self.yrs = yrs
        self.months = months
        self.inflow = inflow
        self.TS = np.shape(inflow)[1] # number of time series to simulate
        self.inflowYr = inflowYr
        self.demandYr = demandYr
        self.res_cap = res_cap
        self.res_init = res_init
        self.marketCostH2O = marketCostH2O
        self.desalCostsH2O = desalCostsH2O
        self.desalUnitSize = desalUnitSize
        self.desalMaxUnits = desalMaxUnits
        self.desalTime2Build = desalTime2Build # yrs
        #self.desalMultiplier = 10 # maybe this should be something different?
        self.N = N
        self.M = M
        self.K = K

        # add time series initializations
        self.R = np.zeros((self.yrs*self.months, self.TS))
        self.I = np.zeros((self.yrs*self.months, self.TS))
        self.Sh = np.zeros((self.yrs*self.months, self.TS))
        self.Dactual = np.zeros((self.yrs*self.months, self.TS))
        self.AllAddedCap = np.zeros((self.yrs*self.months, self.TS))
        self.u = np.zeros((self.yrs*self.months, self.TS))

    # simulation version for adding market water- 2 DVs: (1) the market water multiplier and
    # (2) the market water threshold (at what u level to add market water)
    def run_simulation_marketH2O(self, x, print_output=False):
        self.costAll = np.zeros(self.TS)
        self.shortAll = np.zeros(self.TS)

        for d in range(self.TS):
            if print_output:
                print('simulation time series: ', d)
            # initialize arrays of zeros
            R = np.zeros(self.yrs*self.months)
            Sh = np.zeros(self.yrs*self.months)
            Sp = np.zeros(self.yrs*self.months)
            Dactual = np.zeros(self.yrs*self.months)
            I = np.zeros(self.yrs*self.months)
            Imavg = np.zeros(self.yrs*self.months)
            self.uAll = np.zeros(self.yrs*self.months)
            self.Pol = np.zeros((self.yrs*self.months,2))

            # params to optimize
            # TODO: UPDATE THIS PART FOR DIFFERENT FORMULATIONS
            marketMax = x[0]
            marketThresh = x[1]

            policies = np.random.uniform(0, 1, self.N * (2 * self.M + self.K) + self.K + 2)  # for initialization
            param, lin_param = set_param(policies, self.N, self.M, self.K)  # P is the policy params only

            for t in range((self.yrs)*self.months):
                m = t % self.months
                yr = t // self.months
                if print_output:
                    print('month of simulation: ', m, ', yr of simulation: ', yr)

                I[t] = self.inflow[t+12,d]
                if t==0:
                    R[t] = np.maximum(self.res_init + I[t] - self.demandYr[m], 0)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (self.res_init + I[t])
                        Dactual[t] = self.res_init + I[t]
                    else:
                        Dactual[t] = self.demandYr[m]
                else:
                    R[t] = np.minimum(np.maximum(R[t-1] + I[t] - self.demandYr[m], 0), self.res_cap)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (R[t-1] + I[t])
                        Dactual[t] = R[t-1] + I[t]
                        if print_output:
                            print('water shortage')
                    elif R[t] == self.res_cap:
                        Sp[t] = R[t-1] + I[t] - self.demandYr[m] - self.res_cap
                        Dactual[t] = self.demandYr[m]
                        if print_output:
                            print('spill: ', Sp[t])
                    else:
                        Dactual[t] = self.demandYr[m]

                # inputs for policy search
                # get moving avg of inflows
                Imavg[t] = np.mean(self.inflow[t:t+12,d])
                inputs = np.array([R[t]/self.res_cap, m/self.months, Imavg[t]/np.mean(self.inflowYr)])
                u = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                self.uAll[t] = u[0]

                # if u>0.7:
                # add u*10 units of water to reservoir-- market water
                R[t] = np.minimum(R[t]+marketMax*np.maximum(u[0]-marketThresh, 0), self.res_cap)
                self.Pol[t,0] = 1*(np.maximum(u[0]-marketThresh, 0) > 0) # add a 1 to policy if market water added
                self.Pol[t,1] = np.minimum(10*np.maximum(u[0]-marketThresh, 0), self.res_cap-R[t]) # add amount of market water added

            # get one time series for plotting
            self.R[:,d] = R
            self.I[:,d] = I
            self.Sh[:,d] = Sh
            self.Dactual[:,d] = Dactual
            self.u[:,d] = self.uAll
            self.AllAddedCap[:,d] = self.Pol[:,1]

            # Compute cost of market water to return
            self.costAll[d] = self.marketCostH2O * np.sum(self.Pol[:,1])

            # Compute sum of shortages
            # could do also square the shortages, but not doing this for now
            self.shortAll[d] = np.sum(Sh)
        # , uAll, R, I, Sh, Sp, Dactual, Pol
        cost = np.sum(self.costAll)/self.TS
        short = np.sum(self.shortAll)/self.TS
        return cost, short

    # simulation version for adding discrete unit of desal
    def run_simulation_discreteDesal(self, x, print_output=False):
        self.costAll = np.zeros(self.TS)
        self.shortAll = np.zeros(self.TS)

        for d in range(self.TS):
            if print_output:
                print('simulation time series: ', d)
            # initialize arrays of zeros
            R = np.zeros(self.yrs*self.months)
            Sh = np.zeros(self.yrs*self.months)
            Sp = np.zeros(self.yrs*self.months)
            Dactual = np.zeros(self.yrs*self.months)
            I = np.zeros(self.yrs*self.months)
            Imavg = np.zeros(self.yrs*self.months)
            self.uAll = np.zeros(self.yrs*self.months)
            self.Pol = np.zeros((self.yrs*self.months,2))
            self.PlannedCap = np.zeros(self.yrs*self.months)
            self.AddedCap = np.zeros(self.yrs*self.months)

            # params to optimize
            # TODO: UPDATE THIS PART FOR DIFFERENT FORMULATIONS
            #marketMax = x[0]
            #marketThresh = x[1]
            uThresh = x[0]
            #DesalUnits = x[1]

            policies = np.random.uniform(0, 1, self.N * (2 * self.M + self.K) + self.K + 2)  # for initialization
            param, lin_param = set_param(policies, self.N, self.M, self.K)  # P is the policy params only

            for t in range((self.yrs)*self.months):
                m = t % self.months
                yr = t // self.months
                if print_output:
                    print('month of simulation: ', m, ', yr of simulation: ', yr)

                I[t] = self.inflow[t+12,d]
                if t==0:
                    R[t] = np.maximum(self.res_init + I[t] - self.demandYr[m], 0)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (self.res_init + I[t])
                        Dactual[t] = self.res_init + I[t]
                    else:
                        Dactual[t] = self.demandYr[m]
                else:
                    R[t] = np.minimum(np.maximum(R[t-1] + I[t] - self.demandYr[m], 0), self.res_cap)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (R[t-1] + I[t])
                        Dactual[t] = R[t-1] + I[t]
                        if print_output:
                            print('water shortage')
                    elif R[t] == self.res_cap:
                        Sp[t] = R[t-1] + I[t] - self.demandYr[m] - self.res_cap
                        Dactual[t] = self.demandYr[m]
                        if print_output:
                            print('spill: ', Sp[t])
                    else:
                        Dactual[t] = self.demandYr[m]

                # inputs for policy search
                # get moving avg of inflows
                Imavg[t] = np.mean(self.inflow[t:t+12,d])

                # update added capacity
                if t > 0:
                    n = len(self.AddedCap)
                    if np.abs(self.PlannedCap[t-1]-self.PlannedCap[t]) != 0:
                        self.AddedCap[t:n] = self.AddedCap[t:n] + np.repeat(self.desalUnitSize, n-t)

                inputs = np.array([R[t]/self.res_cap, m/self.months, Imavg[t]/np.mean(self.inflowYr),
                                   self.PlannedCap[t]/(self.desalUnitSize*self.desalMaxUnits),
                                   self.AddedCap[t]/(self.desalUnitSize*self.desalMaxUnits)])
                u = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                self.uAll[t] = u[0]

                # if u > threshold
                if u[0] > uThresh:

                    # 1. check that planned cap + added cap < desalUnits * desalMaxUnits
                    if self.PlannedCap[t]+self.AddedCap[t]+self.desalUnitSize <= (self.desalUnitSize*self.desalMaxUnits):
                        # 2. Update planned capacity
                        if t < (self.yrs*self.months-self.months*self.desalTime2Build-1):
                            self.PlannedCap[t:t+self.months*self.desalTime2Build] = self.PlannedCap[t:t+self.months*self.desalTime2Build]+\
                                                                                    np.repeat(self.desalUnitSize, self.months*self.desalTime2Build)

                            # 2b. Update Capex costs
                            self.costAll[d] = self.costAll[d] + self.desalCostsH2O[0]

                # Update the amount in the reservoir
                R[t] = R[t] + self.AddedCap[t]
                self.Pol[t,0] = 1*(self.AddedCap[t] > 0) # add a 1 to policy if added cap > 0
                self.Pol[t,1] = self.AddedCap[t] # add amount of added capacity
                # update the cost based on operating costs
                self.costAll[d] = self.costAll[d] + self.desalCostsH2O[1]*self.AddedCap[t]

            # get time series for plotting
            self.R[:,d] = R
            self.I[:,d] = I
            self.Sh[:,d] = Sh
            self.Dactual[:,d] = Dactual
            self.u[:,d] = self.uAll
            self.AllAddedCap[:,d] = self.AddedCap

            # Compute sum of shortages
            # could do also square the shortages, but not doing this for now
            self.shortAll[d] = np.sum(Sh)
        # , uAll, R, I, Sh, Sp, Dactual, Pol
        cost = np.sum(self.costAll)/self.TS
        short = np.sum(self.shortAll)/self.TS
        return cost, short

    # simulation version for optimizing the size added of desal
    def run_simulation_contDesal(self, x, print_output=False):
        self.costAll = np.zeros(self.TS)
        self.shortAll = np.zeros(self.TS)

        for d in range(self.TS):
            if print_output:
                print('simulation time series: ', d)
            # initialize arrays of zeros
            R = np.zeros(self.yrs*self.months)
            Sh = np.zeros(self.yrs*self.months)
            Sp = np.zeros(self.yrs*self.months)
            Dactual = np.zeros(self.yrs*self.months)
            I = np.zeros(self.yrs*self.months)
            Imavg = np.zeros(self.yrs*self.months)
            self.uAll = np.zeros(self.yrs*self.months)
            self.Pol = np.zeros((self.yrs*self.months,2))
            self.PlannedCap = np.zeros(self.yrs*self.months)
            self.AddedCap = np.zeros(self.yrs*self.months)

            # params to optimize
            # TODO: UPDATE THIS PART FOR DIFFERENT FORMULATIONS
            #marketMax = x[0]
            #marketThresh = x[1]
            #uThresh = x[0]
            desalMaxAdded = x[0]

            policies = np.random.uniform(0, 1, self.N * (2 * self.M + self.K) + self.K + 2)  # for initialization
            param, lin_param = set_param(policies, self.N, self.M, self.K)  # P is the policy params only

            for t in range((self.yrs)*self.months):
                m = t % self.months
                yr = t // self.months
                if print_output:
                    print('month of simulation: ', m, ', yr of simulation: ', yr)

                I[t] = self.inflow[t+12,d]
                if t==0:
                    R[t] = np.maximum(self.res_init + I[t] - self.demandYr[m], 0)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (self.res_init + I[t])
                        Dactual[t] = self.res_init + I[t]
                    else:
                        Dactual[t] = self.demandYr[m]
                else:
                    R[t] = np.minimum(np.maximum(R[t-1] + I[t] - self.demandYr[m], 0), self.res_cap)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (R[t-1] + I[t])
                        Dactual[t] = R[t-1] + I[t]
                        if print_output:
                            print('water shortage')
                    elif R[t] == self.res_cap:
                        Sp[t] = R[t-1] + I[t] - self.demandYr[m] - self.res_cap
                        Dactual[t] = self.demandYr[m]
                        if print_output:
                            print('spill: ', Sp[t])
                    else:
                        Dactual[t] = self.demandYr[m]

                # inputs for policy search
                # get moving avg of inflows
                Imavg[t] = np.mean(self.inflow[t:t+12,d])

                # update added capacity
                if t > 0:
                    n = len(self.AddedCap)
                    if np.abs(self.PlannedCap[t-1]-self.PlannedCap[t]) != 0:
                        self.AddedCap[t:n] = self.AddedCap[t:n] + np.repeat(np.floor(des_size), n-t)

                inputs = np.array([R[t]/self.res_cap, m/self.months, Imavg[t]/np.mean(self.inflowYr),
                                   self.PlannedCap[t]/(desalMaxAdded),
                                   self.AddedCap[t]/(desalMaxAdded)])
                u = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                self.uAll[t] = u[0]

                # set the desal size
                des_size = u[0]*desalMaxAdded

                # 1. Determine desal size based on policy output
                # we do not limit the amount of desal that can be installed over time
                if des_size > self.PlannedCap[t] + self.AddedCap[t]:
                    # 2. Update planned capacity
                    if t < (self.yrs * self.months - self.months * self.desalTime2Build - 1):
                        addedCap = np.floor(des_size - (self.PlannedCap[t] + self.AddedCap[t]))
                        self.PlannedCap[t:t + self.months * self.desalTime2Build] = self.PlannedCap[
                                                                                    t:t + self.months * self.desalTime2Build] + \
                                                                                    np.repeat(addedCap,
                                                                                              self.months * self.desalTime2Build)

                        # 2b. Update Capex costs
                        self.costAll[d] = self.costAll[d] + self.desalCostsH2O[0]*addedCap

                # Update the amount in the reservoir
                R[t] = R[t] + self.AddedCap[t]
                # this part doesn't quite make sense, so maybe fix this??
                self.Pol[t,0] = 1*(self.AddedCap[t] > 0) # add a 1 to policy if added cap > 0
                self.Pol[t,1] = self.AddedCap[t] # add amount of added capacity
                # update the cost based on operating costs
                self.costAll[d] = self.costAll[d] + self.desalCostsH2O[1]*self.AddedCap[t]

            # get time series for plotting
            self.R[:,d] = R
            self.I[:,d] = I
            self.Sh[:,d] = Sh
            self.Dactual[:,d] = Dactual
            self.u[:,d] = self.uAll
            self.AllAddedCap[:,d] = self.AddedCap

            # Compute sum of shortages
            # could do also square the shortages, but not doing this for now
            self.shortAll[d] = np.sum(Sh)
        # , uAll, R, I, Sh, Sp, Dactual, Pol
        cost = np.sum(self.costAll)/self.TS
        short = np.sum(self.shortAll)/self.TS
        return cost, short

    # simulation version for optimizing the size added of desal- 2DVs:
    # (1) max capacity of desal
    # (2) threshold for when to use desal water
    def run_simulation_contDesal2DVs(self, x, print_output=False):
        self.costAll = np.zeros(self.TS)
        self.shortAll = np.zeros(self.TS)

        for d in range(self.TS):
            if print_output:
                print('simulation time series: ', d)
            # initialize arrays of zeros
            R = np.zeros(self.yrs * self.months)
            Sh = np.zeros(self.yrs * self.months)
            Sp = np.zeros(self.yrs * self.months)
            Dactual = np.zeros(self.yrs * self.months)
            I = np.zeros(self.yrs * self.months)
            Imavg = np.zeros(self.yrs * self.months)
            self.uAll = np.zeros(self.yrs * self.months)
            self.Pol = np.zeros((self.yrs * self.months, 2))
            self.PlannedCap = np.zeros(self.yrs * self.months)
            self.AddedCap = np.zeros(self.yrs * self.months)

            # params to optimize
            # TODO: UPDATE THIS PART FOR DIFFERENT FORMULATIONS
            # marketMax = x[0]
            # marketThresh = x[1]
            # uThresh = x[0]
            desalMaxAdded = x[0]
            desalThresh = x[1]

            policies = np.random.uniform(0, 1, self.N * (2 * self.M + self.K) + self.K + 2)  # for initialization
            param, lin_param = set_param(policies, self.N, self.M, self.K)  # P is the policy params only

            for t in range((self.yrs) * self.months):
                m = t % self.months
                yr = t // self.months
                if print_output:
                    print('month of simulation: ', m, ', yr of simulation: ', yr)

                I[t] = self.inflow[t + 12, d]
                if t == 0:
                    R[t] = np.maximum(self.res_init + I[t] - self.demandYr[m], 0)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (self.res_init + I[t])
                        Dactual[t] = self.res_init + I[t]
                    else:
                        Dactual[t] = self.demandYr[m]
                else:
                    R[t] = np.minimum(np.maximum(R[t - 1] + I[t] - self.demandYr[m], 0), self.res_cap)
                    if R[t] == 0:
                        Sh[t] = self.demandYr[m] - (R[t - 1] + I[t])
                        Dactual[t] = R[t - 1] + I[t]
                        if print_output:
                            print('water shortage')
                    elif R[t] == self.res_cap:
                        Sp[t] = R[t - 1] + I[t] - self.demandYr[m] - self.res_cap
                        Dactual[t] = self.demandYr[m]
                        if print_output:
                            print('spill: ', Sp[t])
                    else:
                        Dactual[t] = self.demandYr[m]

                # inputs for policy search
                # get moving avg of inflows
                Imavg[t] = np.mean(self.inflow[t:t + 12, d])

                # update added capacity
                if t > 0:
                    n = len(self.AddedCap)
                    if np.abs(self.PlannedCap[t - 1] - self.PlannedCap[t]) != 0:
                        self.AddedCap[t:n] = self.AddedCap[t:n] + np.repeat(np.floor(des_size), n - t)

                inputs = np.array([R[t] / self.res_cap, m / self.months, Imavg[t] / np.mean(self.inflowYr),
                                   self.PlannedCap[t] / (desalMaxAdded),
                                   self.AddedCap[t] / (desalMaxAdded)])
                u = get_output(inputs, param, lin_param, self.N, self.M, self.K)
                #self.uAll[t] = u[0]

                # set the desal size
                des_size = u[0] * desalMaxAdded

                # 1. Determine desal size based on policy output
                # we do not limit the amount of desal that can be installed over time
                if des_size > self.PlannedCap[t] + self.AddedCap[t]:
                    # 1b. Update planned capacity
                    if t < (self.yrs * self.months - self.months * self.desalTime2Build - 1):
                        addedCap = np.floor(des_size - (self.PlannedCap[t] + self.AddedCap[t]))
                        self.PlannedCap[t:t + self.months * self.desalTime2Build] = self.PlannedCap[
                                                                                    t:t + self.months * self.desalTime2Build] + \
                                                                                    np.repeat(addedCap,
                                                                                              self.months * self.desalTime2Build)

                        # 1c. Update Capex costs
                        self.costAll[d] = self.costAll[d] + self.desalCostsH2O[0] * addedCap

                # 2. Update the amount in the reservoir based on 2nd DV
                if u[1] > desalThresh:
                    R[t] = R[t] + self.AddedCap[t]
                    # update the cost based on operating costs
                    self.costAll[d] = self.costAll[d] + self.desalCostsH2O[1] * self.desalCostsH2O[2] * self.AddedCap[t]

                # this part doesn't quite make sense, so maybe fix this??
                self.Pol[t, 0] = 1 * (self.AddedCap[t] > 0)  # add a 1 to policy if added cap > 0
                self.Pol[t, 1] = self.AddedCap[t]  # add amount of added capacity

            # get time series for plotting
            self.R[:, d] = R
            self.I[:, d] = I
            self.Sh[:, d] = Sh
            self.Dactual[:, d] = Dactual
            self.u[:, d] = self.uAll
            self.AllAddedCap[:, d] = self.AddedCap

            # Compute sum of shortages
            # could do also square the shortages, but not doing this for now
            self.shortAll[d] = np.sum(Sh)
        # , uAll, R, I, Sh, Sp, Dactual, Pol
        cost = np.sum(self.costAll) / self.TS
        short = np.sum(self.shortAll) / self.TS
        return cost, short

    def plot_simulation(self, ts):
        plt.figure(figsize=(9,9))
        fig, axs = plt.subplots(3)
        # fig.set_size_inches(10, 8, forward=True)
        # fig.set_size_inches(16, 12)
        axs[0].plot(self.R[:,ts], label='Reservoir storage')
        axs[0].plot(self.I[:,ts], label='Inflow')
        axs[0].plot(self.Sh[:,ts], label='Shortage')
        axs[0].plot(self.Dactual[:,ts], label='Demand met')
        axs[0].set_ylabel('Vol/Flow')
        axs[0].legend()

        axs[1].plot(self.u[:,ts], label='policy output values')
        #axs[1].legend()
        axs[1].set_ylabel('Policy val')

        axs[2].plot(self.AllAddedCap[:, ts], label='Added cap')
        axs[2].set_xlabel('Time (months)')
        axs[2].set_ylabel('Desal capacity')
        plt.show()