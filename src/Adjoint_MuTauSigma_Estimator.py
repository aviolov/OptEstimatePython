# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from numpy.random import randn, seed  
from copy import deepcopy

from scipy.interpolate.interpolate import interp2d, interp1d
from scipy.optimize.optimize import fminbound, fmin

from AdjointSolver import FPAdjointSolver,\
                          label_font_size,     xlabel_font_size,\
    generateDefaultAdjointSolver
                          
from AdjointOptimizer import FBKSolution, FBKDriver, visualizeRegimes 
                          
from HitTime_MI_Beta_Estimator import SimulationParams


RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/MuTauSigma_Batch_estimates/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/MuTauSigma_Batch_estimates'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

       
from Adjoint_TauChar_Estimator import SimulationPaths        
       
##################################################################

'Single series estimation routine:'
from scipy.optimize import brentq
def calcMuTauSigmaEstimate(hts, alphaF,
                           mts_init = [0.5,log(0.5), log(0.5)],
                           visualize_gs=False,
                           ftol = 1e-3,
                           maxiter=1000):
    '''given a set of hitting times hts and known control (alpha) - 
       calculate the structural parameters: (mu,tau, sigma) using Max Likelihood'''
    
    Tf = amax(hts)+0.05

    'Objective (negative log-likelihood) Function:'    
    def nllk(mts):
        'current ests:'
        tau = exp(mts[1:2])
        mu_sigma =  [mts[0], exp(mts[2])];
        
        'parametrize solver'
        lSolver =  generateDefaultAdjointSolver(tau, mu_sigma,  Tf=Tf);
        lSolver.refine(0.01, 0.5);
        
        'interpolate control:'
        alphas_for_f = alphaF(lSolver._ts);
        
        'compute hitting time distn:'
        gs = lSolver.solve_hittime_distn_per_parameter(tau,
                                                       mu_sigma,
                                                       alphas_for_f,
                                                       force_positive=True)
        if visualize_gs:
            figure();    plot(lSolver._ts, gs, 'r') ;
        
        'Form likelihood'
        gs_interp = interp1d( lSolver._ts, gs)
        
        'Form negativee log likelihood'
        nllk = -sum(log( gs_interp(hts) ) )
        
        'diagnose:'
        print 'mts: %.3f,%.3f,%.3f,%.0f '%(mu_sigma[0], tau, mu_sigma[1], nllk);
        
        return  nllk; 
    
    'Main Call:'
    from scipy.optimize import fmin
    mts_est, nllk_val, ierr, numfunc, warn_flag = fmin(nllk,
                                                       mts_init,
                                                        ftol = ftol,
                                                         maxiter=maxiter,
                                                          full_output = True);
    if 1 == ierr:
        print 'WARNING: fmin hit max fevals'
        
    print 'NM output:', nllk_val, numfunc, warn_flag
    
    return r_[mts_est[0], exp(array(mts_est[1:]))];




'troubleshotter for the Estimation Routines:'                         
def estimateWorkbox():
    Nblocks = 1;
    Nhits = 100000;
    simPs = SimulationParams();
    alpha_fill = 2.0;

    mts_init = [0.5,log(.5), log(1.)];
        
    'load simulated paths:'
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    'Target Data:'
    thits_Blocks = simPaths.thits_blocks;            
    alphasMap = simPaths.alphasDict;
    Nalphas = len(alphasMap);
    
    print 'Loaded %d Blocks of %d Hitting times for %d controls:'%(thits_Blocks.shape[1], thits_Blocks.shape[2], thits_Blocks.shape[0])
        
    mtsEstimates = empty((Nalphas, Nblocks,3));
            
    alphasMap = {'opt': alphasMap['opt']}
    'loop through controls:'
    for adx, (alpha_tag, alphas) in enumerate( alphasMap.iteritems() ):                    
        print adx, alpha_tag             
        alphaF = interp1d(simPaths.sim_ts,
                          alphas,
                          bounds_error = False, fill_value = alpha_fill);
        figure(figsize=(17,10));
        subplot(211); plot(simPaths.sim_ts, alphaF(simPaths.sim_ts));
        title('Applied Control')
        
        hts = squeeze(thits_Blocks[adx, 0,:]);
            
        'Main estimate call:'
        mts_est = calcMuTauSigmaEstimate(hts,
                                         alphaF,
                                         mts_init= mts_init,
                                            maxiter=100);
#        mts_est = [0.13837874  ,0.29010642,  1.] #0.64458968];
                                              
        print '\hat(mts) = ', mts_est;
        
        Tf = amax(hts)+0.05
        
        cols = ['g', 'r', 'b'];
        
        mts_init[1] = exp(mts_init[1]);
        mts_init[2] = exp(mts_init[2]);
        
        
        mts_true = [.0, 1., 1.];
        
        for pdx, (param_tag,mts) in enumerate(zip(['init', 'final', 'true'],
                                                  [mts_init, mts_est, mts_true])):
            print param_tag, ':', mts
            
            tau = mts[1]; mu_sigma =  [mts[0], mts[2]];
            lSolver =  generateDefaultAdjointSolver(tau, mu_sigma,  Tf=Tf);
            lSolver.refine(0.01, 0.5);
            alphas_for_f = alphaF(lSolver._ts);
            
            
            'Compute hitting time density:'
            gs = lSolver.solve_hittime_distn_per_parameter(tau,
                                                           mu_sigma,
                                                           alphas_for_f,
                                                           force_positive=True)
            
            g_integral = sum(gs)*lSolver._dt;
            
            gs_interp = interp1d(lSolver._ts, gs);                         
            print 'flow-integral = %.3f,llk=%.3g'%(g_integral,
                                                   (sum(log(gs_interp(hts)) )) ) 
            
            subplot(212); hold(True); 
            plot(lSolver._ts, gs, cols[pdx]+'--', linewidth=2)
            
            if pdx == 0:        
                bins = arange(0, lSolver.getTf(), 0.1);
                hist(hts, bins, normed=True); 
            title('Realized Hitting Times')

            
            del lSolver
            
        
           
'the main harness for the estimations'                         
def estimateHarness(simPs,  Nblocks, Nhits,
                    mts_init = [0.1,log(.9), log(1.1)],
                     alpha_fill = 2.0,
                     fig_name=None, reestimate = True):
        
    #Load:
    if reestimate:
        'load simulated paths:'
        simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
        'Target Data:'
        thits_Blocks = simPaths.thits_blocks;        
        alphasMap = simPaths.alphasDict;
        Nalphas = len(alphasMap);
        print 'Loaded %d Blocks of %d Hitting times for %d controls:'%(thits_Blocks.shape[1], thits_Blocks.shape[2], thits_Blocks.shape[0])
        
        mtsEstimates = empty((Nalphas, Nblocks,3))
        
        'loop through controls:'
        for adx, (alpha_tag, alphas) in enumerate( alphasMap.iteritems() ):            
            print adx, alpha_tag             
            
            alphaF = interp1d(simPaths.sim_ts,
                              alphas,
                              bounds_error = False, fill_value = alpha_fill);
            
            'Blocks (sample-path) loop:'
            for bdx in xrange(Nblocks):    
                hts = squeeze(thits_Blocks[adx, bdx,:]);
                
                'Main estimate call:'
                mtsEstimates[adx, bdx,:] = calcMuTauSigmaEstimate(hts,
                                                                  alphaF,
                                                                  mts_init= mts_init);
                print '\n', mtsEstimates[adx, bdx,:]
        
    
    
        'Append estimates to the paths:' 
        simPaths.betaEsts = mtsEstimates;
            
        're-save:'
        simPaths.save('mutausigma_estimation_%d_%d'%(Nblocks, Nhits));
    
    ''' Post Analysis: visualize and latexify results'''
    postAnalysis(Nblocks = Nblocks, 
                 Nhits = Nhits);    
    

'Post process results from Batch Estimation:'
def postAnalysis(Nblocks, Nhits,
                  
                 fig_name=None):
    
    simPaths = SimulationPaths.load(file_name='mutausigma_estimation_%d_%d'%(Nblocks, Nhits));
    alphasMap = simPaths.alphasDict;

    true_params = simPaths.simParams;
    true_params = [true_params.mu, true_params.tau_char, true_params.sigma];
    
    'VISUALIZE:'
    efig = figure(figsize =(17,12)) ; hold(True)
    subplots_adjust(hspace = 0.6, left=0.15, right=0.975 )
    for pdx, (param_tag, param_val) in enumerate(zip([r'\mu', r'\tau', r'\sigma'],
                                                     true_params)):
        'loop through controls:'
        for adx, (alpha_tag, _) in enumerate( alphasMap.iteritems() ):
            print alpha_tag, ' : ', simPaths.betaEsts[adx];            
        
        ctags = [a for a in simPaths.alphasDict.iterkeys()];
           
        Nalphas = len(simPaths.alphasDict);
        
        'per-parameter subplot:'
        ax = subplot(len(true_params), 1, 1+pdx); hold(True);
        for (adx, c_tag, col) in zip( range(Nalphas),
                                         ctags,
                                        ['b', 'r', 'g', 'y', 'k', 'b', 'r', 'g', 'y', 'k'][0:Nalphas]):
            ps = simPaths.betaEsts[adx,:, pdx];
            scatter(adx*ones_like(ps), ps, c = col );
        
        'annotate:'
        ax.set_xlim([-0.5, Nalphas-0.5]);
        ax.set_xticks(range(Nalphas))
        ax.set_xticklabels(ctags,  fontsize = 24) 
        hlines(param_val, 0, Nalphas-1);
        
        title('$%s$ estimates'%param_tag)
        ylabel('$%s$'%param_tag, fontsize=xlabel_font_size)

        
        yticks  = ax.get_yticks();
        ax.set_yticks([yticks[0], yticks[-1]]);         
        ax.set_yticklabels(['$%.2f$'%t for t in ax.get_yticks()],
                            fontsize = label_font_size) 
    
    xlabel('Control Strategy', fontsize = label_font_size);
    
    'Save fig:'    
    fig_name =  os.path.join(FIGS_DIR, 'all_theta_estimates_scatterplot_Nb%d_Ns%d.pdf'%(Nblocks, Nhits))
    print 'Saving to ', fig_name
    savefig(fig_name);
    
    '''LATEXIFY:'''
    print 'skipping latexifying:'
    return
    
    latex_string = r'control type & mean($\hat\tau$) & std($\hat\tau$) \\ \hline '
    
    for adx,   c_tag,  in zip( range(Nalphas),
                                    ctags):
        bs = betaEsts[adx,:]; 
        taus = 1.0/bs;
#        log_taus = log(taus)
        latex_string+= r'%s & %.3f & ' %(c_tag, mean(taus))
        if len(taus)>1:
            latex_string+='%.2f'% std(taus ) 
        else:
            latex_string+='-'
            
        latex_string+=r' \\'
        
    print latex_string
        
    latex_filename = os.path.join(FIGS_DIR,
                                   'tauchar_hit_time_%d.txt'%(Nhits))    
     
    with open(latex_filename, 'w') as the_file:
        the_file.write(latex_string)
        print 'Latex written to ', latex_filename
         
        
        
if __name__ == '__main__':       
    from pylab import *
    rcParams.update({'legend.fontsize': label_font_size,                     
                     'axes.linewidth' : 2.0,
                     'axes.titlesize' : xlabel_font_size ,  # fontsize of the axes title
                     'axes.labelsize'  : xlabel_font_size,  
                     'xtick.labelsize' : label_font_size,
                     'ytick.labelsize' : label_font_size});
  
    
    simPs   = SimulationParams();    
    N_all_hits = 100000;
    
    
    'Workbox:'
#    estimateWorkbox();
    
    
    '''BATCH Parameter Estimation:''' 
#    for Nh in array( [1e5] ).astype(int) :        
#    for Nh in array( [ 1e2  , 1e3, 1e4, 1e5 ]).astype(int): 
    for Nh in array( [1e4]).astype(int): 
        Nb = N_all_hits // Nh;
        
        'Estimate params:'       
        estimateHarness(simPs,
                        Nblocks = Nb, 
                        Nhits = Nh,
                        reestimate=False)
         
    'Shor Figs:'
    show()
    
    