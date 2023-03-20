#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:24:17 2021

@author: martazaniolo
# updated by: jennyskerker
"""

import numpy as np

class node_param:
    def __init__(self):
        self.c = [] # center
        self.b = [] # radius or other shape parameter?
        self.w = [] # weight

# no difference between param (params of nodes) and lin_param (linearly combine outputs of single nodes)
#class ncRBF(object):
# function for the network- multiple nodes; each node defined in node_param class
def get_output(inp, param, lin_param, N, M, K):
    # get layers charateristics
    # N = self.N # number of nodes in hidden layer
    # M = self.M # number of inputs
    # K = self.K # number of outputs
    
    phi = []
    o = []
    output = []
    
    for j in range(N):
        bf = 0
        
        for i in range(M):
            num = (inp[i] - param[j].c[i])*(inp[i] - param[j].c[i])
            den = (param[j].b[i]*param[j].b[i])
            
            if den < pow(10,-6):
                den = pow(10,-6)
            
            bf = bf + num / den
            #print('j: ', j, ', i: ', i, ', num: ', num, ', denom: ', den, ', bf: ', bf)
        
        phi.append( np.exp(-bf) )
            
    for k in range(K):
        o = lin_param[k]
        for j in range(N):
            o = o + param[j].w[k]*phi[j]
            #print('param: ', param[j].w[k], ', phi: ', phi[j], ', o: ', o)
        #print('o: ', o)
        if o > 1:
            o = 1.0
        if o < 0:
            o = 0.0
        #print('o: ', o)
        output.append(o)
        
    return output

def set_param(policies, N, M, K): # policy includes the string of parameters
    #param_string = np.array([policies[0]])
    param_string = policies
    count = 0
    lin_param = []
    param = []
    planning_param = []
    
    # lin parameters. As many as the outputs
    for k in range(K):
        lin_param.append(param_string[count])
        count += 1
    
    #print(count, ' ', param_string)
    # RBF paramters
    for i in range(N): # nodes
        node = node_param()
        for j in range(M):
            node.c.append(param_string[count]) # center
            count += 1
            node.b.append(param_string[count]) # radius
            count += 1
            #print('i=', i, ', j=', j, ', count=', count)
        
        for k in range(K):
            node.w.append(param_string[count]) # output weight
            count += 1
    
        param.append(node)  

        
    return param,lin_param # parameterization that defines the functional class
    
