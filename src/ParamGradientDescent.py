# -*- coding:utf-8 -*-
"""
Created on Nov 11, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand
#ploting functionality:
from pylab import plot, figure, hold, savefig, show, hist, title, xlabel, ylabel, mpl
from matplotlib.pyplot import legend

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/LikelihoodDescent/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/LikelihoodDescent/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

#STRUCTURAL PARAMATER VALUES - HELD FIXED
tau = 20. #ms;
pbeta = 1/tau;
mu = -60.0; 
sigma = .1;

#SIMULATION PARAMETERS:
dt = 1e-3; Tf = 50.0

def get_ts_dWs(dt, tf):
    #Obtain the increments of a Wiener Process realization
    #returns: ts - the sampled time points of the realization
    #         dWs - the forward incrments of W_t at ts[:-1]
    #the square root of dt:
    sqrt_dt = sqrt(dt)
    #the time points:
    ts = r_[arange(.0, tf, dt), tf];
    #the Gaussian incrments:
    dWs = randn(len(ts)-1)* sqrt_dt
    
    return ts, dWs

def simulateAugmentedState(dt=dt, Tf=Tf, alpha = .0, X_0=None,
                           param_step = .01):
    #Obtain a Wiener realization:
    ts,dWs = get_ts_dWs(dt, Tf)
    #allocate memory for soln:
    Xs = empty_like(ts)
    bs = empty_like(Xs)
    #If no X_0 provided, set X_0 to the long-time mean
    if None == X_0:
        X_0 = mu #+ 2*sqrt(sigma / pbeta);
    
    from numpy.random import gamma
    b_0 = gamma(shape = pbeta/2, scale = 2)
    print 'b_0', b_0
    
    Xs[0]  =  X_0; 
    bs[0] = b_0;
    #evolve in time:
    for idx,t in enumerate(ts[:-1]):
        x = Xs[idx]
        #the drift:
        f = alpha + pbeta*(mu - x)
        #the volatility:
        g = sigma
        #update X:              
        Xs[idx+1] = x + f*dt + g*dWs[idx]
        
#        update the param state:
        b = bs[idx];
        
        #the drift:
        f = alpha + b*(mu - x);
        #the gradient of the drift:
        grad_f = (mu - x);
        
        bs[idx+1] =  b + param_step*(-f*grad_f / (sigma*sigma) * dt +\
                                        grad_f / sigma * dWs[idx] )
          
    return ts, Xs, bs
     
def generateData(alpha = 0.,
                 num_samples = 1):
    #number of fine sampled points for the realization:
    n_datapoints = int(Tf / dt) + 1
    
    #the data structure that will hold all the paths:
    trajectoryBank = empty(( n_datapoints, 1+ 2*num_samples ))

    for k in xrange(num_samples):
        #simulate path:    
        ts, Xs, bs = simulateAugmentedState()
        
        #store the time points (once)
        if 0 == k:
            trajectoryBank[:,0] = ts
        
        #Store solution:
        trajectoryBank[:,1+2*k] = Xs
        trajectoryBank[:,1+2*k+1] = bs
    
#    uniq_tag = ''.join(time.asctime().split(' ')[1:3]) + \
#                ''.join(time.asctime().split(' ')[3].split(':'))
    uniq_tag = '%d'%num_samples
    file_name = os.path.join(RESULTS_DIR,
                             'X_gradL_b_%s'%(uniq_tag));
    print file_name
    save(file_name, trajectoryBank)
    
    
    
def analyzeData(num_samples = 1):
    uniq_tag = '';
    uniq_tag = '%d'%num_samples
    file_name = os.path.join(RESULTS_DIR,
                             'X_gradL_b_%s.npy'%(uniq_tag));
    print file_name
    trajectoryBank = load(file_name)
    ts = trajectoryBank[:,0]
    figure()
    for k in xrange(num_samples):
        Xs = trajectoryBank[:,1+2*k]
        bs = trajectoryBank[:,1+2*k+1];
        
        
        subplot(211)
        plot(ts[::10], Xs[::10])
        
        subplot(212)
        plot(ts[::10], bs[::10])
    subplot(212)
    hlines(pbeta, ts[0], ts[-1])
    

def simulateEnsemble(dt=dt, Tf=Tf, alpha = .0, X_0=None,
                     param_step = .01, ensemble_size = 10):
    #Obtain a Wiener realization:
    ts,dWs = get_ts_dWs(dt, Tf)
    #allocate memory for soln:
    Xs = empty_like(ts)
    bs = empty((len(Xs), ensemble_size));
    #If no X_0 provided, set X_0 to the long-time mean
    if None == X_0:
        X_0 = mu #+ 2*sqrt(sigma / pbeta);
    
    from numpy.random import gamma
    b_0 = gamma(shape = pbeta/2, scale = 2, size = (ensemble_size))
    print 'b_0', b_0
    
    Xs[0]  =  X_0; 
    bs[0,:] = b_0;
    #evolve in time:
    for idx,t in enumerate(ts[:-1]):
        x = Xs[idx]
        #the drift:
        f = alpha + pbeta*(mu - x)
        #the volatility:
        g = sigma
        #update X:              
        Xs[idx+1] = x + f*dt + g*dWs[idx]
        
#        update the param state:
        b = bs[idx,:];
        
        #the drift:
        f = alpha + b*(mu - x);
        #the gradient of the drift:
        grad_f = (mu - x);
        
        '''use the hessian:?'''
#        param_step = (sigma*sigma) /((mu-x) * (mu-x)* dt)
        bs[idx+1] =  b + param_step*(-f*grad_f / (sigma*sigma) * dt +\
                                        grad_f / sigma * dWs[idx] )
          
    trajectoryBank = c_[ts, Xs, bs];
    uniq_tag = '%d'%ensemble_size
    file_name = os.path.join(RESULTS_DIR,
                             'X_gradL_b_ensemble_%s'%(uniq_tag));
    print file_name
    save(file_name, trajectoryBank)
    
    
    
def analyzeEnsemble(ensemble_size = 10):
    uniq_tag = '%d'%ensemble_size
    file_name = os.path.join(RESULTS_DIR,
                             'X_gradL_b_ensemble_%s.npy'%(uniq_tag));
    print file_name
    trajectoryBank = load(file_name)
    ts = trajectoryBank[:,0]
    figure(figsize = (17,8))
    subplot(211)
    Xs = trajectoryBank[:,1]
    plot(ts[::10], Xs[::10])
    for k in xrange(ensemble_size):
        bs = trajectoryBank[:,2+k];
        subplot(212); hold(True)
        plot(ts[::10], bs[::10], 'g')
    
    subplot(212); hold(True)
    plot(ts[::10], mean( trajectoryBank[::10,2:], axis=1), 'k')
    
    subplot(212)
    hlines(pbeta, ts[0], ts[-1])
    hlines(0, ts[0], ts[-1])
    hlines(pbeta*3, ts[0], ts[-1])

#The main function pattern in Python:
if __name__ == '__main__':
    #Generate the paths
    from pylab import *
    from numpy import save,load
    
#    num_samples = 2    
#    generateData(num_samples = num_samples)
#    analyzeData(num_samples = num_samples)
    
    ensemble_size=8;
    simulateEnsemble(ensemble_size= ensemble_size)
    analyzeEnsemble(ensemble_size)

    show()