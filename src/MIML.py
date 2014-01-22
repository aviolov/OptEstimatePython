# -*- coding:utf-8 -*-
"""
Created on Nov 11, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand
from OUML import RESULTS_DIR as OUML_RESULTS_DIR, simulateX
from OUML import estimateParamsBeta, dt

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/MIML/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/MIML/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

#STRUCTURAL PARAMATER VALUES - HELD FIXED
tau = 20. #ms;
beta = 1/tau;
mu = -60.0; 
sigma = .1;

xlabel_fontsize = 32
######################################################
class MISimulation():
    _FILE_EXTENSION = '.mis'
    def __init__(self, true_params,
                 ts, Xs,
                 ts_f, Xs_f,  
                 alphas,
                 alphas_f):
        self._true_params = true_params

        self._ts  = ts;
        self._Xs  = Xs;
        self._alphas = alphas;
        
        
        self._ts_f = ts_f;
        self._Xs_f = Xs_f;
        self._alphas_f = alphas_f;

    @classmethod
    def _default_file_name(cls, true_params, alpha_mi):
        return 'MISimulation_default'
    
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = self._default_file_name(self._true_params,
                                                self._alphas_f[0]);
        print 'saving path to ', file_name
        
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name=None, true_params=None, alpha_mi=None):
        ''' not both args can be None!!!'''
        if None == file_name:
            file_name = cls._default_file_method(true_params,
                                                 alpha_mi);
        print 'loading ', file_name

        file_name = os.path.join(RESULTS_DIR,
                                 file_name + cls._FILE_EXTENSION) 
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln

########################

def appendData( alpha=.0,
                num_samples=10,
                delta = 1e-1,
                alphas_applied = [-.25, .0, .25] ):
    ''' Calculate the integral of the likelihood of the next transition 
        wrt to the prior of \beta '''
    alpha_initial = .0;
    file_name = os.path.join(OUML_RESULTS_DIR,
                            'OU_Xs.a=%.3f_N=%d.npy'%(alpha_initial,
                                                     num_samples));
    from numpy import load, save
    trajectoryBank = load(file_name)
    
    T_MI = 50.0
    N_f    = int(T_MI/dt)
    idx    = 3 #!!! 
    ts, Xs = trajectoryBank[:N_f,0], trajectoryBank[:N_f,idx]

    Tf = 100*delta;
    
    Xs_alphas = dict(zip(alphas_applied, 
                         [[],[],[]]));
    ts_f = None;
    for idx in xrange(num_samples):
        l_seed = randint(1e9)
        for alpha in alphas_applied:
            seed(l_seed);
            ts_f, Xs_f = simulateX(dt, Tf, alpha, Xs[-1])
            Xs_alphas[alpha].append(Xs_f[1:]);
    
    true_params= [mu, 1./ tau, sigma];
    ts_f = ts[-1] + ts_f[1:];
    for alpha in alphas_applied:
        Xs_f = Xs_alphas[alpha];
        alphas = zeros_like(ts)
        alphas_f = alpha * ones_like(ts_f)
        file_name = 'alpha_%.2f'%(alpha);
        (MISimulation(true_params, ts, Xs,
                      ts_f, Xs_f, alphas, alphas_f)).save(file_name=file_name)
              
                
    
def visualizeForwardSims(alphas_applied = [-.25, .0, .25] ):
    figure(figsize = (17,8));
    
    colors = dict( zip(alphas_applied,
                    ['b', 'g', 'r']) )
    for alpha in alphas_applied:
        file_name = 'alpha_%.2f'%(alpha);
        Sim = MISimulation.load(file_name)
        ts, Xs, ts_f = Sim._ts, Sim._Xs, Sim._ts_f;
        plot(ts, Xs, 'k');
        
        for idx, Xs_f in enumerate(Sim._Xs_f):
            if idx == 0:
                plot(ts_f, Xs_f, colors[alpha], label= r'$\alpha = %.2f$'%alpha)
            else:
                plot(ts_f, Xs_f, colors[alpha])
        
    legend()
    file_name = os.path.join(FIGS_DIR, 'forward_sims.pdf')
    print file_name
    savefig(file_name)


def estimateParamsBeta(Xs, delta, alphas):
    
    Xn = Xs[1:]
    Xm = Xs[:-1] 
    N = len(Xn)
    alphas=alphas[1:]
    
    def mu_hat_function(beta):
        exp_beta_delta = exp(-beta*delta);
        mu_hat = sum( Xn - exp_beta_delta*Xm - (alphas / beta)* (1 - exp_beta_delta) ) / \
                     (N * (1 - exp_beta_delta))
        return mu_hat;

    def root_function(beta):
        #calc mu:
        mu = mu_hat_function(beta);
        
        #an exponential that occurs often:
        exp_beta_delta = exp(-beta*delta);
        
        #the reduced log-likelihood:
        reduced_logL_arg = (Xn - exp_beta_delta*Xm - (alphas / beta + mu) * (1 - exp_beta_delta));   
        
        #its partial with respect to beta:
        darg_dbeta = (delta*exp_beta_delta * Xm \
                      + (alphas / (beta*beta) ) * (1 -  exp_beta_delta)   \
                      - (alphas / beta   +  mu) * (delta * exp_beta_delta) );
        
        b_error =  dot(reduced_logL_arg, darg_dbeta);
        
        return b_error 
                   
    #####################################
    b_a = 1./(tau) * .01;
    b_b = 1./(tau) * 10.
    
    from scipy.optimize import brentq
    beta_hat = brentq(root_function, b_a, b_b,
                      xtol = 2e-5)
    mu_hat = mu_hat_function(beta_hat)
    
    square_term = (Xn - \
                   exp(-beta_hat*delta)* Xm - \
                   (alphas/beta_hat + mu)*(1-exp(-beta_hat*delta) ) ) **2
    sigma_hat = sqrt( 2* sum (square_term) * beta_hat /\
                       N / (1-exp(-2*beta_hat*delta))) 
    
    #the estimated params:
    return [mu_hat, beta_hat, sigma_hat]


def estimatePerturbedBatch(alphas_applied = [-.25, .0, .25] , delta = 1e-1):
    alpha_estimates = dict(zip(alphas_applied,
                                [[],[],[]]))
    for alpha in alphas_applied:
        file_name = 'alpha_%.2f'%(alpha);
        Sim = MISimulation.load(file_name)
        ts_b, Xs_b, ts_f = Sim._ts, Sim._Xs, Sim._ts_f;
        n_thin = int(delta / dt); print n_thin
        ts = r_[ts_b,ts_f][::n_thin]
        alphas = zeros_like(ts);
        alphas[ts> ts_b[-1]] = alpha
        
#        figure()
#        plot(ts, alphas);
        for idx, Xs_f in enumerate(Sim._Xs_f ):
            Xs = r_[Xs_b, Xs_f][::n_thin];
            
            ests = estimateParamsBeta(Xs, delta, alphas);
            alpha_estimates[alpha].append(ests)
            
        print alpha, ':', alpha_estimates[alpha]
#    print alpha_estimates
    
    file_name = os.path.join(RESULTS_DIR, 'MI_estimates.mis')
    import cPickle
    dump_file = open(file_name, 'wb');
    cPickle.dump(alpha_estimates, dump_file, 1)
    dump_file.close()


def analyzePerterbuedEstimates():
    file_name = os.path.join(RESULTS_DIR, 'MI_estimates.mis')
    import cPickle
    load_file = open(file_name, 'r')
    alpha_ests_dict = cPickle.load(load_file)
    figure(figsize=(17,6))
    for alpha in alpha_ests_dict.keys():
        betas = [val[1] for val in alpha_ests_dict[alpha]];
        print amin(betas), amax(betas)
        plot(ones_like(betas)*alpha, betas, '.');
    hlines(1.0/tau, -.3, .3);
    
    ylim([.0, .15])
    xlim([-.3, .3]) 
    xlabel(r'$\alpha$', fontsize = xlabel_fontsize)
    ylabel(r'$\hat \beta$', fontsize = xlabel_fontsize)
    
    filename = os.path.join(FIGS_DIR, 'perturbed_estimates.pdf')
    print filename
    savefig(filename)
        

#The main function pattern in Python:
if __name__ == '__main__':
    from pylab import *
    
    appendData()
#    visualizeForwardSims()
#    estimatePerturbedBatch()
#    analyzePerterbuedEstimates()
    
    show()