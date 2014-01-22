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

#SIMULATION PARAMETERS:
dt = 1e-2; Tf = 1000.0

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

def simulateX(dt=dt, Tf=Tf, alpha = .0, X_0=None):
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
    mb_guess = [-55, 0.02]
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



def estimateParamsBeta(Xs, delta, alpha = .0):
#   Implements the estimator from question 2.6, given a sample of
#   X_i and a fixed delta
    
    Xn = Xs[1:]
    Xm = Xs[:-1] 
    N = len(Xn)
    
    def mu_hat_function(beta):
        exp_beta_delta = exp(-beta*delta);
        mu_hat = sum( Xn - exp_beta_delta*Xm - (alpha / beta)* (1 - exp_beta_delta) ) / \
                     (N * (1 - exp_beta_delta))
        return mu_hat;

    def root_function(beta):
        #calc mu:
        mu = mu_hat_function(beta);
        
        #an exponential that occurs often:
        exp_beta_delta = exp(-beta*delta);
        
        #the reduced log-likelihood:
        reduced_logL_arg = (Xn - exp_beta_delta*Xm - (alpha / beta + mu) * (1 - exp_beta_delta));   
        
        #its partial with respect to beta:
        darg_dbeta = (delta*exp_beta_delta * Xm \
                      + (alpha / (beta*beta) ) * (1 -  exp_beta_delta)   \
                      - (alpha / beta   +  mu) * (delta * exp_beta_delta) );
        
        b_error =  dot(reduced_logL_arg, darg_dbeta);
        
        return b_error
#    
#    def alternate_root_function(beta):
#        #calc mu:
#        mu = mu_hat_function(beta);
#        
#        numerator   = dot(Xn-mu, Xm-mu);
#        denominator = dot(Xm-mu, Xm-mu);
#        
#        b_error =  delta*beta + log( numerator /
#                                     denominator);
#        
#        return b_error
                   
    #####################################
    b_a = 1./(tau) * .01;
    b_b = 1./(tau) * 10.
    
#    figure();
#    bs = arange(b_a, b_b, .01)
#    rs = array( [root_function(b) for b in bs])
#    plot (bs, rs);

#    print 'guess brackets a,b = ', b_a, b_b
#    print 'root at a, b = %.3f, %.3f'%(root_function(b_a),
#                                       root_function(b_b)); 
    
    from scipy.optimize import brentq
#    beta_hat = brentq(root_function, b_a, b_b,
#                      xtol = 1e-4);
    beta_hat = brentq(root_function, b_a, b_b,
                      xtol = 2e-5)
#    print 'no fprime gives', mb_hat;
#    b_hat = fsolve(root_function, mb_guess, fprime =root_derivative)
#    print 'while fprime gives', mb_hat;
    
    mu_hat = mu_hat_function(beta_hat)
    save
    square_term = (Xn - \
                   exp(-beta_hat*delta)* Xm - \
                   (alpha/beta_hat + mu)*(1-exp(-beta_hat*delta) ) ) **2
    sigma_hat = sqrt( 2* sum (square_term) * beta_hat /\
                       N / (1-exp(-2*beta_hat*delta))) 
    
    #the estimated params:
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
    
def estimateHarness( delta = 1e-1,
                     alpha = .0,
                     num_samples=10,
                     Tf_sample = Tf ):
    #Load all the simulated trajectories
    file_name = os.path.join(RESULTS_DIR,
                              'OU_Xs.a=%.3f_N=%d.npy'%(alpha,
                                                       num_samples));
    trajectoryBank = load(file_name)
    
    #Select an arbitrary trajectory: (here the 2nd)
    figure(); hold(True);
    n_thin = int(delta / dt); print n_thin
    N_sample = int(Tf_sample / dt) 
#    for idx in xrange(1,10):
#    for idx in [2]: #xrange(3,4):
    for idx in xrange(1,num_samples+1):
        ts, Xs = trajectoryBank[:N_sample,0], trajectoryBank[:N_sample,idx]
    
        #Select sampling rate:    
        #Generate sampled data, by sub-sampling the fine trajectory:    
        ts_thin = ts[::n_thin];
        Xs_thin = Xs[::n_thin];
        
        #Obtain estimator
#        est_params = estimateParams(Xs_thin, delta)
#        print 'est original: %.4f,%.4f, %.4f'%(est_params[0],est_params[1],est_params[2])
        est_params = estimateParamsBeta(Xs_thin, delta, alpha)
        print 'est reduced: %.4f,%.4f, %.4f'%(est_params[0],est_params[1],est_params[2])
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
    
     
def generateData(alpha = 0.,
                 num_samples = 10):

    #number of fine sampled points for the realization:
    n_datapoints = int(Tf / dt) + 1
    
    #the data structure that will hold all the paths:
    trajectoryBank = empty(( n_datapoints, 1+ num_samples ))

    for k in xrange(num_samples):
        #simulate path:    
        ts, Xs = simulateX(dt, Tf, alpha)
        
        #store the time points (once)
        if 0 == k:
            trajectoryBank[:,0] = ts
        
        #Store solution:
        trajectoryBank[:,1+k] = Xs
    
    file_name = os.path.join(RESULTS_DIR,
                             'OU_Xs.a=%.3f_N=%d'%(alpha,
                                                  num_samples));
    save(file_name, trajectoryBank)
    
#The main function pattern in Python:
if __name__ == '__main__':
    alpha = 10. / tau;
    num_samples = 10;
    #Generate the paths     
    generateData(alpha       = alpha,
                 num_samples = num_samples)
    
    #Estimate params:
#    start = time.clock();
    estimateHarness(alpha       = alpha,
                    delta       = 1e-1,
                    num_samples = num_samples,
                    Tf_sample   = 1000.)
#    print time.clock() - start, 'ms'
#    LSestimateHarness()
#    pyData2C()

    #Estimate beta for many paths and various delta, Tf, Q2.9-11
#    estimateBatch()

    show()