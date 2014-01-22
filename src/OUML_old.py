# -*- coding:utf-8 -*-
"""
Created on Nov 14, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand
#ploting functionality:
from pylab import plot, figure, hold, savefig, show, hist, title, xlabel, ylabel, mpl
from matplotlib.pyplot import legend

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/OUML/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/OUML/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

#STRUCTURAL PARAMATER VALUES - HELD FIXED
tau = 20. #ms;
beta = 1/tau;
mu = -60.0; 
sigma = .1;

alpha = .0
#SIMULATION PARAMETERS:
dt = 1e-3; Tf = 100.0

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

def simulateX(dt=dt, Tf=Tf, X_0=None):
    #Simulates a path of the OU process using the Millstein scheme:
    #returns: ts - the time points of the simulation
    #         Xs - the simulated values of the process at ts
    
    #Obtain a Wiener realization:
    ts,dWs = get_ts_dWs(dt, Tf)
    #allocate memory for soln:
    Xs = empty_like(ts)
    #If no X_0 provided, set X_0 to the long-time mean
    if None == X_0:
        X_0 = mu #+ 2*sqrt(sigma / beta);
    
    Xs[0]  =  X_0; 
    #evolve in time:
    for idx,t in enumerate(ts[:-1]):
        x = Xs[idx]
        #the drift:
        f = alpha + beta*(mu - x)
        #the volatility:
        g = sigma
                      
        Xs[idx+1] = x + f*dt + g*dWs[idx]  
    
    return ts, Xs

def estimateParams(Xs, delta):
#   Implements the estimator from question 2.6, given a sample of
#   X_i and a fixed delta
    
    Xn = Xs[1:]
    Xm = Xs[:-1] 
    N = len(Xn)

    def root_function(p):
        #rip params:
        mu   = p[0];
        beta = p[1];
#        print 'm, b =', mu, beta

#        m_root = sum(2*(exp(-beta*delta) - 1)*((exp(-beta*delta) - 1)*(mu + alpha/beta) -
#                        Xm*exp(-beta*delta) + Xn)*beta/((exp(-2*beta*delta) - 1)*sigma**2))
#        
#        b_root = N/2 * (2*delta*exp(-2*beta*delta)/(exp(-2*beta*delta) - 1) + 1/beta) +\
#                sum( -2*((exp(-beta*delta) - 1)*(mu + alpha/beta) - Xm*exp(-beta*delta) +\
#                Xn)**2*beta*delta*exp(-2*beta*delta)/((exp(-2*beta*delta) - 1)**2*sigma**2) +\
#                2*((exp(-beta*delta) - 1)*(mu + alpha/beta) - Xm*exp(-beta*delta) +\
#                Xn)*((mu + alpha/beta)*delta*exp(-beta*delta) - Xm*delta*exp(-beta*delta)\
#                + (exp(-beta*delta) - 1)*alpha/beta**2)*beta/((exp(-2*beta*delta) -\
#                1)*sigma**2) - ((exp(-beta*delta) - 1)*(mu + alpha/beta) -\
#                Xm*exp(-beta*delta) + Xn)**2/((exp(-2*beta*delta) - 1)*sigma**2))
        
        m_error = mu - sum(Xn - Xm*exp(-beta*delta)) / \
                         (N * (1.0 - exp(-beta*delta)))
        b_error = beta + log( dot(Xn-mu, Xm-mu) / \
                                    dot(Xm-mu, Xm-mu) ) / delta
        
#        b_error = .0;
        return array([m_error, b_error])
    def root_derivative(p):
        #rip params:
        mu   = p[0];
        beta = p[1];
        
        dmuerror_dbeta = - sum(Xn - Xm*exp(-beta*delta))*(-delta*exp(-beta*delta) ) / \
                           (N * (1.0 - exp(-beta*delta))**2) 
        
        dbetaerror_dmu  =  - sum( (Xn-mu) + (Xm-mu) ) /   \
                               dot(Xn-mu, Xm-mu)     + sum( 2.*(Xm - mu) ) /   \
                                                         dot(Xm-mu, Xm-mu) 
        dbetaerror_dmu /= delta;
        
#        b_error = .0;
#        print c_[ [1., dmuerror_dbeta],
#                  [dbetaerror_dmu, 1.] ];
        return c_[ [1., dmuerror_dbeta],
                   [dbetaerror_dmu, 1.] ];
    #####################################
    mb_guess = [mu * rand(),
                1./(tau) * rand()];
    mb_guess = [-50, 0.02]
#    print 'guess 0 = ', mb_guess
    
    from scipy.optimize import fsolve
#    mb_hat = fsolve(root_function, mb_guess, fprime = None);
#    print 'no fprime gives', mb_hat;
    mb_hat = fsolve(root_function, mb_guess, fprime =root_derivative)
#    print 'while fprime gives', mb_hat;
    mu_hat, beta_hat = mb_hat[:];
    
    
    square_term = (Xn - exp(-beta_hat*delta)* Xm - (alpha/beta_hat + mu) *(1-exp(-beta_hat*delta) ) ) **2
    sigma_hat = sqrt( 2* sum (square_term) * beta_hat /\
                       N / (1-exp(-2*beta_hat*delta))) 
    
    #the estimated params
    return [mu_hat, beta_hat, sigma_hat]


def LSestimateParams(Xs, delta):
#   Implements the estimator from question 2.6, given a sample of
#   X_i and a fixed delta

    Xn = Xs[1:]
    N = len(Xn)
    ts = delta*arange(N);

    def LS_objective(p):
        #Rip params:
        mu   = p[0];
        beta = p[1];
        
        Ys = Xs[0]*exp(-beta*ts) + mu*(1.0- exp(-beta*ts));
        return (Ys-Xn); 
        
    
    #####################################
    mb_guess = [mu * rand(),
                1./(tau) * rand()];
    
    from scipy.optimize import leastsq;
    mb_hat = leastsq(LS_objective, mb_guess)[0]
    mu_hat, beta_hat = mb_hat[:];
    
    return [mu_hat, beta_hat]
    
def estimateHarness( delta = 1e-2 ):
    #Load all the simulated trajectories
    file_name = os.path.join(RESULTS_DIR, 'OU_Xs.N=10.npy')
#    file_name = os.path.join(RESULTS_DIR, 'OU_Xs.a=0.000_N=10.npy')
    trajectoryBank = load(file_name)
    
    #Select an arbitrary trajectory: (here the 2nd)
    figure(); hold(True);
    n_thin = int(delta / dt); print n_thin 
    for idx in xrange(1,10):
#    for idx in xrange(3,4):
        ts, Xs = trajectoryBank[:,0], trajectoryBank[:,idx]
    
        #Select sampling rate:    
        #Generate sampled data, by sub-sampling the fine trajectory:    
        ts_thin = ts[::n_thin]; Xs_thin =  Xs[::n_thin]
        
        #Obtain estimator
        est_params = estimateParams(Xs_thin, delta)
        
        print '%.4f,%.4f, %.4f'%(est_params[0],est_params[1],est_params[2])
        plot(ts_thin, Xs_thin);
         
    print 'true param values:', [mu, beta, sigma]
    
  
def pyData2C( delta = 1e-2 ):
    #Load all the simulated trajectories
    file_name = os.path.join(RESULTS_DIR, 'OU_Xs.N=10.npy')
    trajectoryBank = load(file_name)
    
    #Select an arbitrary trajectory: (here the 2nd)
    figure(); hold(True);
    n_thin = int(delta / dt); print n_thin, delta 
    idx = 3;
    ts, Xs = trajectoryBank[:,0], trajectoryBank[:,idx]
    #Select sampling rate:    
    #Generate sampled data, by sub-sampling the fine trajectory:    
    ts_thin = ts[::n_thin]; Xs_thin =  Xs[::n_thin]
    
    file_name = os.path.join(RESULTS_DIR, 'Xdata.txt',);
#    outfile = open(file_name, 'w')
#    outfile.print(N);
#    outfile.print(delta);
#    for X in Xs_thin:
#        outfile.print(X)
#        
#    outfile.close()
    
    savetxt(file_name, Xs_thin, "%f");
    
    print 'Saved to ', file_name
    print 'N, delta,', len(Xs_thin), delta
    
def LSestimateHarness( delta = 1e-2 ):
    #Load all the simulated trajectories
    file_name = os.path.join(RESULTS_DIR, 'OU_Xs.N=10.npy')
    trajectoryBank = load(file_name)
    
    #Select an arbitrary trajectory: (here the 2nd)
    figure(); hold(True);
    n_thin = int(delta / dt); print n_thin 
    for idx in xrange(1,10):
        ts, Xs = trajectoryBank[:,0], trajectoryBank[:,idx]
    
        #Select sampling rate:    
        #Generate sampled data, by sub-sampling the fine trajectory:    
        ts_thin = ts[::n_thin]; Xs_thin =  Xs[::n_thin]
        
        #Obtain estimator
        est_params = LSestimateParams(Xs_thin, delta)
        
        print '%.4f,%.4f, '%(est_params[0],est_params[1])
        plot(ts_thin, Xs_thin);
         
    print 'true:', [mu, beta]
    
     
def generateData():
#    Generates and stores a lot of paths of X_t for Q2.9-11
    num_samples = 10;
    
    #number of fine sampled points for the realization:
    n_datapoints = int(Tf / dt) + 1
    
    #the data structure that will hold all the paths:
    trajectoryBank = empty(( n_datapoints, 1+ num_samples ))

    for k in xrange(num_samples):
        #simulate path:    
        ts, Xs = simulateX(dt, Tf)
        
        #store the time points (once)
        if 0 == k:
            trajectoryBank[:,0] = ts
        
        #Store solution:
        trajectoryBank[:,1+k] = Xs
    
    file_name = os.path.join(RESULTS_DIR, 'OU_Xs.N=%d'%num_samples)
    save(file_name, trajectoryBank)
#def estimateBatch():
#    #Finds estimators for the various delta, Tf combinations in
#    #Q2.9, 2.10 and 2.11
#    
#    #Load data
#    trajectoryBank = load('Q2.91011.N=500.npy')
#    #for each delta, Tf combination
#    for delta, Tf, Q_id in zip([1e-2,1e-3,1e-3],
#                                 [10.0, 10.0, 1.0],
#                                 ['2.9', '2.10', '2.11']):
#        
#        betas = empty(trajectoryBank.shape[1]-1)
#        
#        #for each X_t realization estimate beta:
#        for k in xrange(len(betas)):
#            Xs =  trajectoryBank[:,k+1]
#               
#            #subsample appropriately:
#            n_thin = int(delta / dt)
#            N_last = int(Tf/dt)+1
#            Xs_thin =  Xs[:N_last:n_thin]              
#            
#            #estimate beta:
#            betas[k], est_variance = estimateBeta(Xs_thin, delta)
#            
#        #visualize and save figures:
#        mpl.rcParams['figure.figsize'] = 17,6
#        figure()
#        m = mean(betas)
#        s = std(betas)
#        hist(betas, bins = 25, normed=True)
#        mean_var = r'$\hat\mu_\beta = %.4f, \hat\sigma^2_\beta = %.4f$'%(m,s**2); print mean_var
#        title(mean_var, fontsize=32)
#        xlabel(r'$\beta$', fontsize=24)
#        
#        filename = os.path.join(FIGS_DIR, 'Q'+Q_id.replace('.', '') + '.pdf')
#        savefig(filename)
          

#The main function pattern in Python:
if __name__ == '__main__':
    #Generate the paths     
#    generateData()
    
    #Estimate params:
#    start = time.clock();
    estimateHarness()
#    print time.clock() - start, 'ms'
#    LSestimateHarness()
#    pyData2C()

    

        
    #Estimate beta for many paths and various delta, Tf, Q2.9-11
#    estimateBatch()

    show()