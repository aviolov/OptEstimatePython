# -*- coding:utf-8 -*-
"""
Created on Mar 13, 2012

@author: alex
"""
from __future__ import division
from numpy import *

from copy import deepcopy
from numpy.random import randn, rand, randint, seed
from ControlOptimizer import PiecewiseConstSolution
from ControlBasisBox import piecewiseConst

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/SpikeTrains/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/SpikeTrains/'
import os

#import ext_fpc

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

########################
class SpikeTrain():
    FILE_EXTENSION = '.st'
    def __init__(self, spike_ts, params):
        self._spike_ts = spike_ts;
        self._params = params; 
        
    def getTf(self):
        return self._spike_ts[-1]
        
    def save(self, file_name):
        import cPickle
        dump_file = open(file_name + SpikeTrain.FILE_EXTENSION, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name):
        import cPickle
        load_file = open(file_name+ SpikeTrain.FILE_EXTENSION, 'r')
        path = cPickle.load(load_file)        
        return path
########################
#
#def simulateSpikeTrain(self, spikes_requested, dt = None):
#        #Set default dt:        
#        if (None == dt):
#            dt = 1e-4 / self._path_params._theta
#        
#        #Main Sim Routine
##        spike_ts = self._pysimulate(spikes_requested, dt)
#        spike_ts = self._csimulate(spikes_requested, dt)
#        
#        #Return:
#        path_params = deepcopy(self._path_params)
#        simulatedSpikeTrain = SpikeTrain(spike_ts, path_params)
#        
#        return simulatedSpikeTrain;
#        
#    def _csimulate(self, spikes_requested, dt):
##        alpha, beta, gamma, theta = ;
#        abgth = array( self._path_params.getParams() );
#        return ext_fpc.simulateSDE(abgth,spikes_requested,dt);
#        
#    def _pysimulate(self, spikes_requested, dt):
#        #Set the (fixed) integration  times:
#        spike_ts = [];
##        ts = arange(0., spikes_requested, dt);
##        vs = zeros_like(ts);
#        
##        Bs = randn(len(ts));
#        sqrt_dt = sqrt(dt);
#        
#        alpha, beta, gamma, theta = self._path_params.getParams();
#
#        #THE MAIN INTEGRATION LOOP:
#        v = .0
#        t = .0;
#        recorded_spikes =0;
#        while recorded_spikes < spikes_requested:
#            
#            dB = randn()*sqrt_dt
#            dv = (alpha - v + gamma*sin(theta*t))*dt + \
#                          beta*dB
#
#            v += dv;
#            t += dt;
#            if v >= self._v_thresh:
#                spike_ts.append(t)
#                v = .0;
#                recorded_spikes += 1
#
#        #Return:
#        return spike_ts;         


    
def simulateSpikeTrain(N_spikes, 
                             params, alpha_fun,
                             save_path=True, path_tag='', overwrite=True,
                             seeds = None, dt = None):
    mu, theta, beta   = params[:]
    print 'Parameterizing Spike Train with: ', mu, theta, beta
    
    file_save_name = os.path.join(RESULTS_DIR,
                                  'mu=%.2f_th=%.2f_b=%.2f__N=%d_%s'%(mu, theta,
                                                                     beta, N_spikes, path_tag))  
    if save_path and \
         False == overwrite and \
            True  == os.path.exists(file_save_name + SpikeTrain.FILE_EXTENSION):
        print file_save_name, ' exists, returning'
        return

    # The actual work:  
    if (None == dt):
            #default time-step.
            dt = 1e-4 / theta #the bigger the theta the stiffer the problem, the smaller the time-step
    spike_ts = [];
    sqrt_dt = sqrt(dt);
    #THE MAIN INTEGRATION LOOP:
    #TODO: Move to _pysimulate and have a separate _csimulate
    X = .0
    t = .0;
    recorded_spikes = 0;
    seed(seeds[0])
    while recorded_spikes < N_spikes:
        
        dB = randn()*sqrt_dt
        dX = (alpha_fun(t) + mu - theta * X ) * dt + \
                      beta*dB

        X += dX;
        t += dt;
        if X >= 1.0:
            spike_ts.append(t)
            X = .0;
            recorded_spikes += 1
            if (recorded_spikes != N_spikes):
                #reset the seed for the next spike time (unless we are done):
                seed(seeds[recorded_spikes])
    #//END THE MAIN INTEGRATION LOOP
    
    T = SpikeTrain(spike_ts, params);
    print 'Spike count = ', N_spikes
    print 'Simulation time = ' , T.getTf()
    print T._params
        
    if save_path:   
        print 'Saving path to ' , file_save_name    
        T.save(file_save_name)

    figure();
    stem(spike_ts, ones_like(spike_ts)); xlabel('t')
    title('mu=%.2f_th=%.2f_b=%.2f__N=%d_%s'%(mu, theta,beta, N_spikes, path_tag))
 
class OptimalAlpha():
    def __init__(self, file_name):
        OptSoln = PiecewiseConstSolution.load(file_name)  
        #the following is better done with a functor, but complexity is an enemy:
        self._ts = OptSoln.ts
        self._alphas = piecewiseConst(OptSoln.alpha_opt, OptSoln.ts);
    def alpha_opt_fun(self, t):
        if t >= self._ts[-1]:
            return self._alphas[-1];
        else:
            return interp(t, self._ts, self._alphas)

#    def alpha_opt_fun(t):
#        if t >= knots[-2]:
#            return OptSoln.alpha_opt[-1]
#        else:
#            t_idx = floor(len(knots)/2);
#            a = 0; b = len(knots)-1
#            while not (knots[t_idx] < t and knots[t_idx+1]>t):
#                if knots[t_idx] <t:
#                    a = t_idx;
#                    t_idx += floor( ( b-t_idx ) / 2 )                    
#                else:
#                    b = t_idx;
#                    t_idx -= floor((t_idx-a)/2)
#            return OptSoln.alpha_opt[t_idx]
    
         
                
def batchSimulator(N_spikes = 100, #N_trains=100
                    overwrite=False):
    mu_true = .0;
    theta_true  = .5;
    beta_true = 1.;
    
    params = [mu_true, theta_true, beta_true]
    
    
    OptFunctor = OptimalAlpha('BasicNMTest'); 
    

    alpha_opt_fun = OptFunctor.alpha_opt_fun
    alpha_crit_fun = lambda t: theta_true;

#    ts = arange(0,20, 1.)
#    alphas = [alpha_opt_fun(t) for t in ts]
#    figure()
#    plot(ts, alphas)
#    show()
    
    seeds = randint(1000000000, size=N_spikes)
#    for  idx in xrange(1, N_trains+1):
    for alpha_fun, alpha_tag in zip([alpha_opt_fun, alpha_crit_fun],
                                    ['alpha_opt', 'alpha_crit']):
        
        simulateSpikeTrain(N_spikes, 
                           params, alpha_fun,
                           save_path=True, path_tag=alpha_tag, overwrite=True,
                           seeds=seeds)
            

def generateSDF(Is):
    N = len(Is)
    unique_Is = unique(Is)
    
    ts = r_[(.0, unique_Is)]
    SDF = ones_like(ts)
    
    for (Ik, idx) in zip(unique_Is, arange(1, len(SDF))):
        SDF[idx] = sum(Is> Ik) / N;
    return SDF, unique_Is

def visualizeDistributions(file_name, fig_name):
    file_name = os.path.join(RESULTS_DIR, file_name)
    P = SpikeTrain.load(file_name)
    Is = r_[(P._spike_ts[0], diff(array(P._spike_ts)))];
    
    SDF, unique_Is = generateSDF(Is)
    
    mpl.rcParams['figure.subplot.left'] = .15
    mpl.rcParams['figure.subplot.right'] =.95
    mpl.rcParams['figure.subplot.bottom'] = .15
    mpl.rcParams['figure.subplot.hspace'] = .4
    
    figure()
    ax = subplot(211)
    hist(Is, 100)
    title(r'$\alpha,\beta,\gamma = (%.3g,%.3g,%.3g) $' %(P._params._alpha, P._params._beta, P._params._gamma), fontsize = 24)
    xlabel('$I_n$', fontsize = 22);
    ylabel('$g(t)$', fontsize = 22);
    
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
    
    
    ax = subplot(212)
    plot(r_[(.0,unique_Is)], SDF, 'rx', markersize = 10)
    ylim((.0,1.))
    xlabel('$t$', fontsize = 22)
    ylabel('$1 - G(t)$', fontsize = 22)
    for label in ax.xaxis.get_majorticklabels():
                label.set_fontsize(20)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(20)
            
    
    fig_name = os.path.join(FIGS_DIR, fig_name)
    print 'saving to ', fig_name
    savefig(fig_name)
    
    
#def CvsPySimulator():
#    import time
#    N_spikes = 1000;
#    for params, tag in zip([[1.5, .3, 1.0, 2.0], [.5, .3, 1.12, 2.0], [.4, .3, .4, 2.0],[.1, .3, 2.5, 2.0]],
#                           ['superT', 'crit', 'subT','superSin']):
#        alpha,beta,gamma,theta = params[0],params[1], params[2], params[3];
#        print 'Parameterizing with: ', alpha, beta, gamma, theta 
#
#        # The actual work:  
#        S = OUSinusoidalSimulator(alpha, beta, gamma, theta);
#        
#        start = time.clock()
#        pyTs         = S._pysimulate(N_spikes, dt=1e-4)
#        stop = time.clock()
#        print 'Py time = ', stop-start
#        
#        start = time.clock()
#        cTs         = S._csimulate(N_spikes, dt=1e-4)
#        stop = time.clock()
#        print 'C time = ', stop-start
#        
#        figure()
#        subplot(211)
#        hist(diff(pyTs), normed=1)
#        subplot(212)
#        hist(diff(cTs), normed=1) 

def latexParamters():
    theta = 1.0
    print r'Regime Name & $\a$ & $\b$ & $\g$ \\ \hline'
    for params, tag in zip([    [1.5-.1, .3, .1 * sqrt(1. + theta**2),     theta],
                                [.5,     .3, .5* sqrt(1. + theta**2),      theta],
                                [.4,     .3, .4* sqrt(1. + theta**2),      theta],
                                [.1,     .3, (1.5-.1)*sqrt(1. + theta**2), theta]],
                        ['Supra-Threshold', 'Critical', 'Sub Threshold','Super Sinusoidal']):
        print tag + r'&%.2f&%.2f&%.2f \\' %(params[0],params[1],params[2])
        
    print r'''\end{tabular}
              \caption{Example $\abg$ parameters for the different regimes, given $\th = %.1f$}
        '''%theta
        
def rename_stst():
    import subprocess
    p = subprocess.Popen("cd " + RESULTS_DIR +"; ls *.st.st", stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    output = output.split('\n')
    
    import os
    
    for line in output:
        src = os.path.join(RESULTS_DIR, line)
        dst = os.path.join(RESULTS_DIR, line.replace('st.st', 'st'))
#        print ' from ', src , ' to ' , dst 
        os.rename(src, dst)
        
        
def crossCompare(N_spikes = 64):
    mu = .0;
    theta  = .5;
    beta = 1.;
    
    OptFunctor = OptimalAlpha('BasicNMTest'); 

    alpha_opt_fun = OptFunctor.alpha_opt_fun
    alpha_crit_fun = lambda t: theta;

    ts = arange(.0, 10, .5);
    figure(figsize=(17,8))
    for alpha_fun, alpha_tag,\
         y_height, alpha_mrkr in zip([alpha_opt_fun, alpha_crit_fun],
                                      ['alpha_opt', 'alpha_crit'],
                                      [1,-1],
                                      ['b*', 'rx']):
        
        file_load_name = os.path.join(RESULTS_DIR,
                                      'mu=%.2f_th=%.2f_b=%.2f__N=%d_%s'%(mu, theta,
                                                                         beta, N_spikes, alpha_tag))  
    
        T = SpikeTrain.load(file_load_name)
        alphas = [alpha_fun(t) for t in ts]
        
        subplot(211); hold(True)
        stem(T._spike_ts, y_height*ones_like(T._spike_ts),
             linefmt=alpha_mrkr[0]+ '-', markerfmt=alpha_mrkr)
        ylim((-1.5, 1.5))
        xlabel(r'$t$')
#        legend()
        subplot(212); hold(True)
        plot(ts, alphas, alpha_mrkr[0], label=alpha_tag);
        ylabel(r'$\alpha$'); xlabel(r'$t_{local}$')
        legend()
    
    save_file_name = os.path.join(FIGS_DIR, 'CrossCompare_alpha_opt_VS_alpha_crit.pdf')
    print 'saving to ',  save_file_name
    savefig(save_file_name)
    
        
if __name__ == '__main__':
    from pylab import *
    import time

#    latexParamters()
#    CvsPySimulator()

    
#    start = time.clock()
#    batchSimulator(N_spikes=64, overwrite=True)
#    batchSimulator(N_spikes=1000, N_trains = 1, overwrite=True)
#    print time.clock() - start

    crossCompare()
    
    


#    sinusoidal_spike_train(N_spikes=1000, fig_tag = 'SuperT', 
#                           params = [1.5, .3, 1.0, 2.0], dt =1e-4)
#    sinusoidal_spike_train(N_spikes = 200, fig_tag = 'Crit', 
#                           params = [.55, .5, .55, 2.0])
#    sinusoidal_spike_train(N_spikes = 200, fig_tag = 'SubT', 
#                           params = [.4, .3, .4, 2.0])

#    sinusoidal_spike_train(N_spikes=100, fig_tag = 'SuperSin', 
#                           params = [.1, .3, 2.0, 2.0], dt =1e-4)
    
#    visualizeDistributions('sinusoidal_spike_train_N=1000_superSin_12', 'SuperSin_Distributions')
#    visualizeDistributions('sinusoidal_spike_train_T=5000_superT_14.path', 'SuperT_Distributions')
    
    show()
        