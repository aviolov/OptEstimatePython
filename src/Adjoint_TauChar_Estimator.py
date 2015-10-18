# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from numpy.random import randn, seed  
from copy import deepcopy

from scipy.interpolate.interpolate import interp2d, interp1d
from scipy.optimize.optimize import fminbound

from AdjointSolver import FPAdjointSolver,\
                          label_font_size,     xlabel_font_size
                          
from AdjointOptimizer import FBKSolution, FBKDriver, visualizeRegimes 
                          
from HitTime_MI_Beta_Estimator import SimulationParams


RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/HitTime_MI_TauChar_Adjoint_Estimate/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/HitTime_MI_TauChar_Adjoint_Estimate'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

       
#class EstimationResults():
#    FILE_EXTENSION = '.er'
#    def __init__(self, tags, ests ):
#        self.ests = ts;
#        if shape(xs_block)[1] != shape(cs_block)[1]:
#            raise RuntimeError('xsBlock, csBLock mismatched sizes')
#        self.xs = xs_block;
#        self.cs = cs_block;
#        self.simParams = simParams;
#        self.control_tag = control_tag;
#        
#    def getT(self):
#        return self.ts[-1];
#    def getN(self):
#        return shape(self.xs)[1]
#        
#        
#    def save(self, file_name=None):
##       path_data = {'path' : self}
#        if None == file_name:
#            P = self.simParams;
#            file_name = self.getDefaultFileName(self.getT(),
#                                           self.getN(),
#                                           P.mu,
#                                           P.tau_char,
#                                           P.sigma,
#                                           self.control_tag);
#                                                                   
#        print 'saving path to ', file_name
#        file_name = os.path.join(RESULTS_DIR, file_name + self.FILE_EXTENSION)
#        import cPickle
#        dump_file = open(file_name, 'wb')
#        cPickle.dump(self, dump_file, 1) # 1: bin storage
#        dump_file.close()
#    @classmethod
#    def getDefaultFileName(cls, T,N,m,tc, sigma, ctag):
#        return 'OUSim_%s_T=%.2f_N=%d_m=%.2f_tc=%.2f_s=%.2f'%(ctag, T,N,m,tc, sigma);
#    
#    @classmethod
#    def load(cls, TNmtcs_ctag=None,file_name=None ):
#        ''' not both args can be None!!!'''
#        if None == file_name:
#            T,N,m,tc, sigma, ctag = [x for x in TNmtcs_ctag]
#            file_name = cls.getDefaultFileName(T, N, m, tc, sigma,ctag);
#
#        file_name = os.path.join(RESULTS_DIR, file_name +  cls.FILE_EXTENSION) 
#        print 'loading ', file_name
#        import cPickle
#        load_file = open(file_name, 'r')
#        soln = cPickle.load(load_file)        
#        return soln 

       
class SimulationPaths():
    FILE_EXTENSION = '.sp'
    def __init__(self, thits_blocks, simParams, alphasDict, sim_ts, betaEsts = []):
        self.thits_blocks = thits_blocks;
 
        self.simParams = simParams;
        
        self.alphasDict = alphasDict; 
        
        self.sim_ts = sim_ts;
        
        self.betaEsts = [];   
        
       
    def save(self, file_name=None):
#       path_data = {'path' : self}
        if None == file_name:
            P = self.simParams;
            Nb = self.thits_blocks.shape[1]
            Nh = self.thits_blocks.shape[2]
            
            file_name = self.getDefaultFileName(Nb,
                                                Nh,
                                                P.mu,
                                                P.tau_char,
                                                P.sigma );
                                                                   
        print 'saving hitting times to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + self.FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def getDefaultFileName(cls, Nb, Nh, m,tc, sigma, ctag=''):
        return 'HTSim_%s_Nb=%d_Nh=%d_m=%.2f_tc=%.2f_s=%.2f'%(ctag, Nb, Nh, m,tc, sigma);
    
    @classmethod
    def load(cls, NbNhmtcs_ctag=None,file_name=None ):
        ''' not both args can be None!!!'''
        if None == file_name:
            Nb, Nh, m,tc, sigma, ctag = [x for x in NbNhmtcs_ctag]
            file_name = cls.getDefaultFileName(Nb, Nh, m,tc, sigma, ctag);

        file_name = os.path.join(RESULTS_DIR, file_name +  cls.FILE_EXTENSION) 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln 

##################################################################
##################################################################
##################################################################
   
def visualizeControls(simPs, fbkSoln,
                      Tf = 5.0,
                      fig_name=None):
    
    alphas = fbkSoln._cs_iterates[-1];
    
    alpha_max = fbkSoln._alpha_bounds[-1];
    
    alpha_crit = 1./simPs.tau_char;  
    ts = fbkSoln._Solver._ts;
    
    figure();
    plot(ts, alphas, label='opt'); 
    
    hlines(alpha_max, ts[0], ts[-1], colors='k')
    hlines(alpha_crit, ts[0], ts[-1], colors='k')
    
    ylabel(r'$\alpha_t$', fontsize = xlabel_font_size)
    xlabel(r'$t$', fontsize = xlabel_font_size)
    ylim(( 1.1*alpha_max*array([-1,1])))
      

    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
   
     
def visualizePaths(simPs,  
                   Nblocks , 
                  Nhits , 
                  fig_name=None):
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    thits_Blocks = simPaths.thits_blocks;
    
    Nbins = floor(Nhits / 5);
    
    Nalphas = len(simPaths.alphasDict)
    
    '''VISUALIZE'''   
    figure( figsize = (17, 10))
    subplots_adjust(  wspace = .25 ); 
    
    for (pdx, alpha_tag) in enumerate(simPaths.alphasDict.iterkeys()):
        thits = thits_Blocks[pdx, 0, :];
        
        subplot(Nalphas,1,pdx+1)
        hist(thits, normed=True, bins = Nbins);
        title(r'$\alpha=%s$'%alpha_tag, fontsize = 36);
        
        ylabel('frequency', fontsize = 24)  
        if pdx == Nalphas-1:
            xlabel('$t$', fontsize = 24)
            
    if fig_name != None:
        lfig_name = os.path.join(FIGS_DIR,
                             fig_name + '_thits_distn.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
    
    
'''Main Harness to generate/sample the hitting times for each considered control:'''
def GeneratePathsHarness(simPs,                    
                         Nblocks = 2,
                         Nhits = 100,
                         Tmax = 25., dt = 0.001, 
                         x_0 = 0.,
                         visualize_paths = True,
                         recompute_optimzation   = False,
                         recompute_hitting_simulation = False,
                         alpha_max = 2.0):
    
    

    'Recompute Optimal Controls?'
    Tf = 15.0
    if recompute_optimzation:
        mu_sigma = [simPs.mu, simPs.sigma]
        
        tau_chars_list = [linspace(0.25, 4, 10),
                          [.25, 4],
                          [.75, 1/.75 ]  ]
   
        priorTags = ['default-prior', 'wide-prior', 'narrow-prior']
         
        alpha_bounds = [-alpha_max,alpha_max];
    
        for prior_tag, tau_chars in zip(priorTags, tau_chars_list):
            soln_name = 'BatchEstimate_%s'%prior_tag
            
            FBKDriver(tau_chars, ones_like(tau_chars)/len(tau_chars),
                          mu_sigma,
                          alpha_bounds=alpha_bounds,
                          Tf=Tf,
                          visualize_decsent = False,
                          soln_save_name=soln_name);
                          
            'Visualize Regimes'
            visualizeRegimes(tau_chars,
                             mu_sigma,
                             Tf,
                             soln_name = soln_name)
        
    
    def interpControl(soln_tag):
        lFbkSoln = FBKSolution.load(soln_tag);
        
        return interp1d(lFbkSoln._Solver._ts, lFbkSoln._cs_iterates[-1],
                          bounds_error = False, fill_value = lFbkSoln._alpha_bounds[-1]);
    
    alpha_crit = 1.0/simPs.tau_char
 
    '''Put in a Dictionary'''    
    alphaFunctionMap = {'opt': interpControl('BatchEstimate_default-prior'),
                        'opt-wide': interpControl('BatchEstimate_wide-prior'),
                        'opt-narrow': interpControl('BatchEstimate_narrow-prior'),
                        'max': lambda ts: alpha_max*ones_like(ts),
                        'crit': lambda ts: alpha_crit*ones_like(ts)}

    
    
    '''Main Simulation Call'''
    if recompute_hitting_simulation:
        GeneratePathsBatch(alphaFunctionMap,
                        simPs,                    
                         Nblocks = Nblocks,
                         Nhits = Nhits,
                         visualize_paths=visualize_paths) 
    
    'Visualize for sanity:'
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    alphaFunctionMap = simPaths.alphasDict;
       
    figure(); hold(True)
    for (key,val) in alphaFunctionMap.iteritems():
        plot(simPaths.sim_ts, val, label=key);
      
    legend();
    

def GeneratePathsBatch(alphaFunctionMap,
                        simPs,                    
                         Nblocks = 2,
                         Nhits = 100,
                         Tmax = 25., dt = 0.001, 
                         x_0 = 0.,
                         visualize_paths = True): 
    
    #ASSIGN STRUCT PARAMS:
    mu, tauchar, sigma = simPs.mu, simPs.tau_char, simPs.sigma;
 
    #The random numbers:
    sqrt_dt = sqrt(dt);  
    
    #the time points:   
    N_ts = Tmax/dt;  
    
    # the number of controls:
    N_alphas = len(alphaFunctionMap)
    
    #the storage vectors:
    hts_block =     empty((N_alphas, Nblocks,Nhits)) 
        
    lts = arange(0, Tmax, dt)
    dWs = empty( (Nhits, N_ts-1) ); print prod(dWs.shape) *8/1e6
    
    
    #the dynamics RHS:    
    def compute_dX(alpha, x, dW): 
        return (alpha + (mu - x)/tauchar)*dt + sigma * dW
    X_THRESH = 1.0
    ##MAin integratino loop
    def computeHittingTime(x, alphaF, ldWs):
        thit = .0; 
        for t, dW, alpha in zip(lts, ldWs, alphaF):
            if x>=X_THRESH:
                break;  
#            print '%.3f:%.3f'%(t,alpha)
            x+=compute_dX(alpha, x, dW);
            thit+=dt
        return thit
    
    '''The BATCH Sample Paths:''' 
    for bdx in xrange(Nblocks):   
        print '%d/%d'%(bdx,Nblocks)     
        #the common Gaussian incrments:
        dWs = randn(Nhits, N_ts-1) * sqrt_dt; 
        
        'Iterate over the various controls:'
        for adx, ( alpha_tag, alphaF) in enumerate( alphaFunctionMap.iteritems() ):            
            print adx, alpha_tag
            for hdx in xrange(Nhits):
                ldWs = dWs[hdx,:];
                'compute hitting time'
                hts_block[adx, bdx, hdx] = computeHittingTime(.0, alphaF(lts), ldWs);
                if hts_block[adx, bdx, hdx] > Tmax-.1:
                    print hts_block[adx, bdx, hdx]

    for key in alphaFunctionMap.iterkeys():
        alphaFunctionMap[key] = alphaFunctionMap[key](lts);
        
    (SimulationPaths(hts_block, simPs, alphaFunctionMap, lts)).save();
    
    

from scipy.optimize import brentq
def calcBetaEstimate(lSolver, hts, alphaF, mu_sigma,
                      bmin= 0.2, bmax = 5):
    '''given a set of hitting times hts and known alpha, sigma - 
    calculate the other parameter beta using Max Likelihood'''
    
    '''NOT USED:'''
#    'F solver:'
#    dx = .05;
#    x_min  = -1.5;
    dt = .025;    
    Tf = amax(hts)+2*dt 
#    S = ForwardSolver(mu_sigma, 
#                      dx, x_min,
#                      dt, Tf)     
#    alphas_for_F = alphaF( S._ts );
    
    '''f Solver:'''
    lSolver.setTf(Tf);
    alphas_for_f = alphaF(lSolver._ts);
    
    def beta_nllk(beta):
        ''' Тhis is the root'''
#        Fs = squeeze( S.solve(alphas_for_F,
#                              [beta]));
#        dt = S._ts[1] - S._ts[0]; 
#        gs_dt = -diff(Fs[:, -1]) / dt;
        
        
        gs_dx = lSolver.solve_hittime_distn_per_parameter(1/beta,
                                                          mu_sigma,
                                                          alphas_for_f)
#        
#        figure(); hold(True)
#        plot(S._ts[1:], gs_dt, 'b')
#        plot(lSolver._ts, gs_dx, 'r')
#        show();
        
#        gs_interp = interp1d( S._ts[1:], gs_dt)
        gs_interp = interp1d( lSolver._ts, gs_dx)
        nllk = -sum(log( gs_interp(hts) ) )
        sys.stdout.write('%.3f,%.0f '%(beta, nllk));
        sys.stdout.flush()
        return  nllk ; 
            
    beta_est, nllk_val, ierr, numfunc= fminbound(beta_nllk, bmin, bmax, xtol=1e-3,
                                                 full_output = True);
    if 1 == ierr :
        print 'WARNING: fminbound hit max fevals'
        
    return beta_est;

def estimatorWorkbench(simPs, fbkSoln):
    Nb = 1
    Nh = 1e5
    
    '''The weird defective NLLK Issue:'''
#    betaSweep_JaggedNLLK_Issue(simPs, fbkSoln);
    
    '''Check if the Empirical Distribution matches the analytic one'''
    for Nh in array( [ 1e5 ]).astype(int) :
#    for Nh in array( [1e3,1e4,1e5]).astype(int): 
        Nb = 100000 // Nh;
        estimatorEmpricialDistnMatch(simPs, fbkSoln,
                                  Nblocks = Nb, Nhits = Nh)
#        estimatorInterpolationBox(simPs, fbkSoln, 
#                                  Nblocks = Nb, Nhits = Nh)
    
    'Check the effect of refining dx, dt:'
#    estimatorDtDxConvergence(simPs, fbkSoln)

    'Profile Bottleneck:'
#    import cProfile
#    cProfile.runctx('estimatorDtDxConvergence(simPs, fbkSoln)',
#                     globals(),
#                     {'simPs':simPs, 'fbkSoln':fbkSoln}, 'cfsolve.prof')
    
#    import pstats
#    pyprof = pstats.Stats('pyfsolve.prof')
#    pyprof.sort_stats('time')
#    pyprof.print_stats()
#    cprof  = pstats.Stats('cfsolve.prof')
#    cprof.sort_stats('time')
#    cprof.print_stats()
                     
    

def estimatorEmpricialDistnMatch(simPs, fbkSoln,
                                    Nblocks ,   Nhits,
                                    refine_factor = 8.):
    'Here we plot the Empirical vs. Analytics Hitting time distribution'
    alpha_max = 2.0
    alpha_crit = 1./simPs.tau_char;  
    alphas_opt = fbkSoln._cs_iterates[-1]
    alphaFopt = interp1d(fbkSoln._Solver._ts, alphas_opt,
                        bounds_error = False, fill_value = alpha_max) 
         
    alphaFs = [alphaFopt, 
               lambda ts: alpha_crit*ones_like(ts), 
               lambda ts: alpha_max*ones_like(ts)]         
    lSolver = deepcopy( fbkSoln._Solver ); 
    lSolver.rediscretize( lSolver._dx/refine_factor, lSolver._dt/refine_factor,
                          lSolver.getTf(),    lSolver._xs[0] )
    
    print 'Solver min, dt, dx', lSolver.getXmin(), lSolver._dt, lSolver._dx    
    
    'Load hitting times:'
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    thits_Blocks = simPaths.thits_blocks;

#    for alpha_idx in arange(0, len(alphaFs)):
#    for alpha_idx in arange(0, 2):
    for alpha_idx in arange(0, 3):
        alphaF = alphaFs[alpha_idx]
        bdx = 0 
        
        thitsSet = thits_Blocks[alpha_idx,bdx,:].flatten()
#        thitsSet = thitsSet[thitsSet<20];
        
        figure(  figsize = (17, 12)  )
        subplot(211)   
        hold(True);
        hist(thitsSet, bins = 500, normed = True, label='empirical')
        title_tag = 'Ns = %d, beta_true = %.2f' %(len(thitsSet),
                                                  1/simPs.tau_char )
        title(title_tag)    
        xlabel(r'$t$',   fontsize=xlabel_font_size);
        ylabel('$g(t)$', fontsize=xlabel_font_size)  
        
        betas_plot = 2**(linspace(0,.5, 2));
        
        maxThit = amax(thitsSet);
        print 'bdx, Nspikes, maxTf ', bdx, len(thitsSet), maxThit;
        lSolver.setTf(maxThit +.01)
        ts = lSolver._ts;
        alphas = alphaF(ts);
        mu_sigma = [simPaths.simParams.mu,  simPaths.simParams.sigma]  ;

        for beta in betas_plot:
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1/beta,
                                                              mu_sigma,
                                                              alphas)
            plot( ts, gs_dx, linewidth=3, label='b=%.2f'%beta)
               
               
#        betas =  2**(linspace(-1,1, 50)) 
#        betas_sweep =  2**(linspace(-.2,.3, 100))  
        betas_sweep =   linspace(0.8, 1.25, 45)        
        def beta_nllk(beta): 
            'main call'
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1./beta,
                                                              mu_sigma,
                                                              alphas)
            
            'interpolate data and log-likelihood:'
            gs_interp = interp1d( lSolver._ts, gs_dx ) 
             
            gs_interpolated = gs_interp(thitsSet);         
            nllk = -sum(log(gs_interpolated ) )
            
            'diagnostics:' 
            print 'negative probs: finite-diffs = %d, interpolated=%d'%(sum(gs_dx <= 0), sum( gs_interpolated <= 0 ))
            print 'sum g, beta, nllk', sum(gs_dx[1:]*diff(ts)), beta, nllk
            
            '''return'''
            return  nllk ;  
        
        
        '''Optimization bit:'''     
        nllks = [beta_nllk(b) for b in betas_sweep]
              
        beta_est = calcBetaEstimate(lSolver, thitsSet, alphaF, mu_sigma)
        legend()
#        
        subplot(212) 
        plot(betas_sweep, nllks, 'x-');   
        xlabel(r'$\beta$', fontsize=xlabel_font_size);
        ylabel('nllk', fontsize=xlabel_font_size) 
        
        max_idx = argmin(nllks);
        yl, yu = ylim();
        vlines(betas_sweep[max_idx], yl, yu, colors='r', label='MLE beta brute sweep');
        vlines(1./simPs.tau_char, yl, yu, colors='g', label='true beta');
        vlines(beta_est, yl, yu, colors='g', label='MLE beta - optimized');  
        legend();
        title(r'beta est = %.3f (alphaidx = %d) '%(betas_sweep[max_idx], alpha_idx ))
        
        lfig_name = os.path.join(FIGS_DIR,
                                'Adjoint_TauChar_Estimator_estimatorWorkbench_b=%dx%d_a%d.pdf'%(bdx, Nhits, alpha_idx))
                                   
        print 'saving to ', lfig_name
        savefig(lfig_name)
        

def betaSweep_JaggedNLLK_Issue(simPs, fbkSoln,
                         Nblocks=100 ,   Nhits=1000  ):
    'Here we plot the Empirical vs. Analytics Hitting time distribution'
    alpha_max = 2.0
    alpha_crit = 1./simPs.tau_char;  
    alphaF =   lambda ts: alpha_crit*ones_like(ts)         
    lSolver = deepcopy( fbkSoln._Solver );
    factor = 4.;
    lSolver.rediscretize( lSolver._dx/factor,   lSolver._dt/factor,
                          lSolver.getTf(), lSolver._xs[0] )
    
    print 'Solver min, dt, dx', lSolver.getXmin(), lSolver._dt, lSolver._dx    
    
    'Load hitting times:'
    alpha_idx = 1;
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    thits_Blocks = simPaths.thits_blocks;
    thitsSet = thits_Blocks[alpha_idx,0,:].flatten() 
    thitsSet = sort( thitsSet )
    maxThit = amax(thitsSet);
    print 'Nspikes, maxTf ',  len(thitsSet), maxThit;
    lSolver.setTf(maxThit + 1.0)
    ts = lSolver._ts;
    alphas = alphaF(ts) 
    mu_sigma = [simPaths.simParams.mu,  simPaths.simParams.sigma]  ;
    
    betas_sweep =   linspace(.85, .95 , 200)        
    def beta_nllk(beta): 
        'main call'
        gs_dx = lSolver.solve_hittime_distn_per_parameter(1./beta,
                                                         mu_sigma,
                                                          alphas)
        while any( isnan(gs_dx) )  :
            print 'WARNING: repeating calculation! due nans'
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1./beta,mu_sigma,  alphas)
           
#            if norm_const<1.:
#                lSolver.setTf(lSolver.getTf()+1 );
#                alphas = alphaF(lSolver._ts)
        norm_const = sum(gs_dx[1:]*diff(ts));
        if abs(norm_const - 1) > 1e-2:
            print 'WARNING: retrying calculation! due to non-normalized g'
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1./beta,mu_sigma,  alphas)
            norm_const = sum(gs_dx[1:]*diff(lSolver._ts));
            
        'interpolate data and log-likelihood:'
        gs_interp = interp1d( lSolver._ts, gs_dx,
                              kind = 'nearest') 
         
        gs_interpolated = gs_interp(thitsSet);  
        
#        if contains(set(betas_sweep[0::4], beta)):
        nllk = -sum(log(gs_interpolated ) )
        
        visualize = isnan(nllk) or nllk > 1095 or nllk < 1000
        #False and in1d([beta], betas_sweep[0::4])[0];
        if visualize:
            ''' visualize'''        
            figure()   
            hold(True);
            hist(thitsSet, bins = 500, normed = True, label='empirical')
            plot(lSolver._ts, gs_dx, label='FD g', linewidth = 3)
            plot(thitsSet, gs_interpolated, label='interpolatedd g', linewidth = 3)
            
            title_tag = ' beta_used = %.2f' %(beta )
            title(title_tag)    
            xlabel(r'$t$',   fontsize=xlabel_font_size);
            ylabel('$g(t)$', fontsize=xlabel_font_size)                
            legend()
        
        'diagnostics:' 
        print 'negative probs: finite-diffs = %d, interpolated=%d'%(sum(gs_dx <= 0), sum( gs_interpolated <= 0 ))
        print 'sum (g), beta, nllk', norm_const, beta, nllk
        
        '''return'''
        return  nllk ;  
    
    
    '''Optimization bit:'''     
    nllks = [beta_nllk(b) for b in betas_sweep]
    
    figure(); hold(True);
    plot(betas_sweep, nllks, 'x-');   
    xlabel(r'$\beta$', fontsize=xlabel_font_size);
    ylabel('nllk', fontsize=xlabel_font_size) 
    
    max_idx = argmin(nllks);
    yl, yu = ylim();
    vlines(betas_sweep[max_idx], yl, yu, colors='r', label='MLE beta');
    vlines(1./simPs.tau_char, yl, yu, colors='g', label='true beta');  
    legend();
    title(r'beta est = %.3f  '%(betas_sweep[max_idx] ))
     
#        lfig_name = os.path.join(FIGS_DIR,
#                                'Adjoint_TauChar_Estimator_estimatorWorkbench_b=%dx%d_a%d.pdf'%(bdx, Nhits, alpha_idx))
#                                   
#        print 'saving to ', lfig_name
#        savefig(lfig_name)
        
def estimatorInterpolationBox(simPs, fbkSoln,
                              Nblocks ,   Nhits  ):
    '''QUestion: Does the interpolation kind in gs make a difference
       Answer: NO!!!''' 
    alpha_max = 2.0
    alpha_crit = 1./simPs.tau_char;  
    alphas_opt = fbkSoln._cs_iterates[-1]
    
    'different interpolations:'    
    alphaFs = [interp1d(fbkSoln._Solver._ts, alphas_opt,
                        bounds_error = False, fill_value = alpha_max,
                        kind = interp_kind) 
                    for interp_kind in ["linear", "nearest", "zero", 
                                        "slinear", "quadratic", "cubic" ] ] 
        
    lSolver = deepcopy( fbkSoln._Solver );
    factor = 16.;
    lSolver.rediscretize( lSolver._dx/factor,   lSolver._dt/factor,
                          lSolver.getTf(), lSolver._xs[0] )
    
    print 'Solver min, dt, dx', lSolver.getXmin(), lSolver._dt, lSolver._dx    
    
    'Load hitting times:'
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    thits_Blocks = simPaths.thits_blocks;
    interp_kinds = ["linear", "nearest", "zero", 
                    "slinear", "quadratic", "cubic" ]
#    for alpha_idx in arange(0, len(alphaFs)):
    for alpha_idx in [0, 1, 4]: 
        alphaF = alphaFs[alpha_idx]
        bdx = 0 
        
        thitsSet = thits_Blocks[0,bdx,:].flatten()
#        thitsSet = thitsSet[thitsSet<20];
        
        figure(  figsize = (17, 12)  )
        subplot(211)   
        hold(True);
        hist(thitsSet, bins = 500, normed = True, label='empirical')
        title_tag = 'Ns = %d, beta_true = %.2f' %(len(thitsSet),
                                                  1/simPs.tau_char )
        title(title_tag)    
        ylabel('$g(t)$', fontsize=xlabel_font_size)  
        
        betas_plot = 2**(linspace(0,.5, 2));
        
        maxThit = amax(thitsSet);
        print 'bdx, Nspikes, maxTf ', bdx, len(thitsSet), maxThit;
        lSolver.setTf(maxThit +.01)
        ts = lSolver._ts;
        alphas = alphaF(ts) 
        mu_sigma = [simPaths.simParams.mu,  simPaths.simParams.sigma]  ;

        for beta in betas_plot:
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1/beta,
                                                              mu_sigma,
                                                              alphas)
            plot( ts, gs_dx, linewidth=3, label='b=%.2f'%beta)
               
  
        betas_sweep =   linspace(.9, 1.4, 25)        
        def beta_nllk(beta): 
            'main call'
            gs_dx = lSolver.solve_hittime_distn_per_parameter(1/beta,
                                                              mu_sigma,
                                                              alphas)
            
            'interpolate data and log-likelihood:'
            gs_interp = interp1d( lSolver._ts, gs_dx,
                                   kind = interp_kinds[alpha_idx])  
            gs_interpolated = gs_interp(thitsSet);         
            nllk = -sum(log(gs_interpolated ) )
            
            'diagnostics:' 
            print 'negative probs: raw finite-difs, interpolated',\
                          sum(gs_dx <= 0), sum( gs_interpolated <= 0 )
            print 'sum g, beta, nllk, interpkind = %.6f, %.3f, %.0f, %s'%(  sum(gs_dx[1:]*diff(ts)), beta, nllk, interp_kinds[alpha_idx])
            
            '''return'''
            return  nllk ;  
        
        
        '''Optimization bit:'''     
        nllks = [beta_nllk(b) for b in betas_sweep]
              
        legend()
#        
        subplot(212) 
        plot(betas_sweep, nllks, 'x-');   
        xlabel(r'$\beta$', fontsize=xlabel_font_size);
        ylabel('nllk', fontsize=xlabel_font_size) 
        
        max_idx = argmin(nllks);
        yl, yu = ylim();
        vlines(betas_sweep[max_idx], yl, yu, colors='r', label='MLE beta');
        vlines(1./simPs.tau_char, yl, yu, colors='g', label='true beta');  
        legend();
        title(r'beta est = %.3f (alphaidx = %d) '%(betas_sweep[max_idx], alpha_idx ))
        
        
        
def estimatorDtDxConvergence(simPs, fbkSoln,
                             Nblocks=1 ,   Nhits =1e5 ):
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    alpha_max = fbkSoln._alpha_bounds[-1];
    
    alpha_crit = 1./simPs.tau_char;  
    
    thits_Blocks = simPaths.thits_blocks; 
    
    
    alphas_opt = fbkSoln._cs_iterates[-1]
    alphaF = interp1d(fbkSoln._Solver._ts, alphas_opt,
                      bounds_error = False, fill_value = alpha_max)
#    alphaF = lambda ts: alpha_crit*ones_like(ts);
    lSolver = fbkSoln._Solver;
#    betaEsts = []
    alpha_idx = 0;
    hts = thits_Blocks[alpha_idx,0,:].flatten()
    
    for rdx in arange(0,4):  
      
        lSolver.rediscretize(lSolver._dx/2,   lSolver._dt/2,
                             lSolver.getTf(), lSolver._xs[0]) 
        
        if rdx>=2                                                                                               :
            beta_est = calcBetaEstimate(lSolver,
                                      hts,
                                      alphaF, 
                                      [simPaths.simParams.mu, simPaths.simParams.sigma],
                                      bmin = .75, bmax = 1.5);
                
            print '\n refine level:%d (%d): dx,dt = %.4f %.4f NxNt = %d,%d: best = %.3f'%(rdx, 2**(1+rdx),
                                             lSolver._dx, lSolver._dt,
                                             lSolver._num_nodes(), lSolver._num_steps(),
                                             beta_est)
         
def NblocksNhitsRearrange(simPs,Nblocks, Nhits):
    'a utility to spearate an existing Hitting TIme Simulation into different Ns x Nb splits'
    N_all_hits = Nblocks*Nhits;
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
        
    thits_Blocks = simPaths.thits_blocks;  
    
    Nalphas = thits_Blocks.shape[0];
    
    for Nblocks in [1, 10, 100, 1000]:
        Nhits = N_all_hits // Nblocks;
#        thits_Blocks = thits_Blocks.reshape((3, Nblocks, Nhits))
        simPaths.thits_blocks = copy(thits_Blocks.reshape((Nalphas, Nblocks, Nhits)))
        
        simPaths.save();
        
        print simPaths.betaEsts
                 
                          
def estimateHarness(simPs, fbkSoln,
                     Nblocks, Nhits, 
                     refine_factor = 2.,
                     fig_name=None, reestimate = True):
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    if reestimate:
        'Target Data:'
        thits_Blocks = simPaths.thits_blocks;
        
        'FP Solver:'
        lSolver = deepcopy(fbkSoln._Solver);
        
        ts = simPaths.sim_ts;
        alphasMap = simPaths.alphasDict;
        Nalphas = len(alphasMap);
        print 'Loaded %d Blocks of %d Hitting times for %d controls:'%(thits_Blocks.shape[1], thits_Blocks.shape[2], thits_Blocks.shape[0])
        
        betaEsts = empty((Nalphas, Nblocks))
            
        alpha_max = fbkSoln._alpha_bounds[-1]; 
        
        
        'refine the solver:'
        lSolver.rediscretize(lSolver._dx/refine_factor,   lSolver._dt/refine_factor,
                             lSolver.getTf(), lSolver._xs[0])
        
        for adx, (alpha_tag, alphas) in enumerate( alphasMap.iteritems() ):
            
            print adx, alpha_tag 
            
            alphaF =  interp1d(ts, alphas , bounds_error = False, fill_value = alpha_max)
            
            for bdx in xrange(Nblocks):
    #        for bdx in [0]:
                hts = squeeze(thits_Blocks[adx, bdx,:]);
                
    #            subplot(3, 1, adx)
    #            hist(hts, bins = 100)
    #            title('%s %d'%(alpha_tag, len(hts) ))
                
                betaEsts[adx, bdx] = calcBetaEstimate(lSolver,
                                                      hts,
                                                      alphaF,
                                                      [simPaths.simParams.mu,
                                                        simPaths.simParams.sigma] );
                print '\n', betaEsts[adx, bdx]
        
        'Append estimates to the paths:' 
        simPaths.betaEsts = betaEsts;
        
        'resave:'
        simPaths.save();
    
    ''' Post Analysis: visualize and latexify results'''
    postAnalysis( Nblocks = Nblocks, 
                  Nhits = Nhits, 
                  simPs = simPs) 
    
    
def estimateHarnessSandbox(Tf, Npaths = 10, Npaths_to_visualize = 4,
                    simParams = SimulationParams(),   ctags = ['det', 'feedback', 'placebo'],
                    fig_name = None):
    
    bs = arange(.1, 3.0, .01 )
    def calcBetaRoot(xs, ts, cs):
        Xn = xs[:-1]
        Xm = xs[1:]
        an = cs[:-1]
        Delta = ts[2]-ts[1]
        N = len(Xn);
                
        
        def f_beta_root(beta):
            outer_term = N/2*(2*Delta*e**(-2*Delta*beta)/(e**(-2*Delta*beta) - 1) + 1/beta);
            
            
            inner_sum = ( -2*(Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                                Xn)**2*Delta*beta*e**(-2*Delta*beta)/(e**(-2*Delta*beta) - 1)**2 +
                                2*(Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                                Xn)*(Delta*Xm*e**(-Delta*beta) - Delta*an*e**(-Delta*beta)/beta -
                                (e**(-Delta*beta) - 1)*an/beta**2)*beta/(e**(-2*Delta*beta) - 1) -
                                (Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                                Xn)**2/(e**(-2*Delta*beta) - 1) )
 
            return outer_term - sum(inner_sum)
            
        
        fbs = array([f_beta_root(b) for b in bs] )
        
        return bs, fbs
    
    
    #Load:
    simPathsList = [];
    
    for rdx, c_tag, in enumerate( ctags):
        
        TNMtcs_tag = [Tf, Npaths,
                       simParams.mu, simParams.tau_char, simParams.sigma,
                        c_tag]
        
        simPathsList.append(  SimulationPaths.load(TNMtcs_tag) );
        
    
    figure(figsize=(17,18))
    
    for tdx in xrange(Npaths_to_visualize):

        subplot(ceil(Npaths_to_visualize/2),2 ,tdx+1)
        for simPaths, c_tag in zip( simPathsList,
                                    ctags):
            
            
            ts = simPaths.ts;
            xs = simPaths.xs[:, tdx];
            cs = simPaths.cs[:, tdx];
            
            
            bs, fbs = calcBetaRoot(xs, ts, cs)
            plot(bs, fbs, label=c_tag);
            xlabel(r'$\beta$', fontsize = 28)
            ylabel(r'$\partial_\beta l(\beta)$', fontsize = 28)
            
        plot(bs, .0*bs, 'k--')    
        title(r'The ML Estimates root function for $\beta $', fontsize = 24 )
    
        legend();
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR,
                                  'BetaRoot_' + fig_name + '.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)


def postAnalysis(Nblocks, Nhits, simPs,
                 fig_name=None):
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    betaEsts = simPaths.betaEsts
    
    ctags = [a for a in simPaths.alphasDict.iterkeys()];
       
    Nalphas = len(simPaths.alphasDict);
    
    'VISUALIZE:'
    figure(figsize =(17,10)) ; hold(True)
    for (adx, c_tag, col) in zip( range(Nalphas),
                                     ctags,
                                    ['b', 'r', 'g', 'y', 'k', 'b', 'r', 'g', 'y', 'k'][0:Nalphas]):
        bs = betaEsts[adx,:]; 
        
        scatter(adx*ones_like(bs), 1/bs, c = col );
        print c_tag, mean(bs), std(bs)
        
#    figure(figsize =(17,10)) ; hold(True)
#    for (adx, c_tag) in zip( range(Nalphas),
#                                     ctags):
#        bs = betaEsts[adx,:]; 
#        subplot(5,2, adx*2 +1)
#        hist(1/bs); title('Raw')
#        subplot(5,2, adx*2 +2); hist(log(1/bs)); title('log Transformed')
#    return

    ylabel(r'$\hat{\tau}$', fontsize=xlabel_font_size)
    hlines(simPaths.simParams.tau_char, 0, Nalphas-1);
#    legend(ctags)  
    axc = gca();
    axc.set_xticks(range(Nalphas))
    axc.set_xticklabels(ctags,  fontsize = 24) 
 
    xlabel('Control Strategy', fontsize = label_font_size);
    ylim((0 ,2))
    ticks = [0,1,2,]
    axc.set_yticks(ticks)
    axc.set_yticklabels(['$%.0f$'%t for t in ticks],
                         fontsize = label_font_size) 
    

    
    fig_name =  os.path.join(FIGS_DIR, 'miestimates_scatterplot_Nb%d_Ns%d.pdf'%(Nblocks, Nhits))
    print 'Saving to ', fig_name
    savefig(fig_name);
    
    '''LATEXIFY:'''
    print 'latexifying:'
    
    latex_string = r'control type & mean($ \hat\tau  $ ) & std($ \hat\tau $) \\ \hline '
    
    for adx,   c_tag,  in zip( range(Nalphas),
                                    ctags):
        bs = betaEsts[adx,:]; 
        taus = 1.0/bs;
#        log_taus = log(taus)
        latex_string+= r'%s & %.3f & %.2f \\' %(c_tag, mean(taus ), std(taus ) )
        
    print latex_string
        
    latex_filename = os.path.join(FIGS_DIR,
                                   'tauchar_hit_time_%d.txt'%(Nhits))    
     
    with open(latex_filename, 'w') as the_file:
        the_file.write(latex_string)
        print 'Latex written to ', latex_filename
       
    
    
#def latexifyResults(Npaths = 1000, Tfs = [8, 16,32] ):
#    bEstimates  = {};
#    ctags = []
#    for Tf in Tfs:
#        bEsts = estimateHarness(Tf, Npaths = Npaths);
#        
#        for ctag, bs in bEsts.iteritems():
#            bEstimates[(Tf, ctag)] = (mean(bs), std(bs))
#            
#        ctags = bEsts.keys();
#        
#        
#    #latexify:
#    print 'latexifying:'
#    print '$T_f$:'
#    for Tf in Tfs:
#        print ' & %d'%Tf
#    print r'\\'    
#    for ctag in ctags:
#        print ctag, ':'
#        for Tf in Tfs:  
#            m,s =  bEstimates[(Tf, ctag)]        
#            print  ' & (%.2f, %.2f)'%(m,s)    
#        print r'\\'
#     
#
# 
#def latexifyBetaSweepResults(Npaths, Tf ,
#                             tau_chars, sigmas):
#    from scipy.stats import nanmean, nanstd
#    
#    bEstimates  = {};
#    ctags = []
#    for tau_char in tau_chars:
#            for sigma in sigmas:
#                simPs = SimulationParams(tau_char = tau_char, sigma = sigma)
#                bEsts = estimateHarness(Tf, Npaths = Npaths, simParams= simPs);
#        
#                for ctag, bs in bEsts.iteritems():
#                    bEstimates[(tau_char, sigma, ctag)] = (nanmean(bs),
#                                                           nanstd(bs),
#                                                           sum(isnan(bs)))
#                ctags = bEsts.keys();  
#                
#                  
#    #LATEXIFY:
#    
#    print 'latexifying:'
#    
#    latex_string = ''
#    latex_string+= r'$(\beta_{true}, \sigma)$:'
#    for tau_char in tau_chars:
#            for sigma in sigmas:
#                latex_string += ' & (%.2f,%.2f)'%(1/tau_char, sigma)
#    latex_string+= r'\\ \hline  '    
#    for ctag in ctags:
#        latex_string+= ctag + ':'
#        for tau_char in tau_chars:
#            for sigma in sigmas:
#                m,s,nnan =  bEstimates[(tau_char, sigma, ctag)]        
##                latex_string+=  ' & (%.2f, %.2f, %d)'%(m,s, nnan)    
#                latex_string+=  ' & (%.2f, %.2f )'%(m,s)    
#        latex_string+= r'\\'
#    
#    print latex_string
#        
#    latex_filename = os.path.join(FIGS_DIR, 'betasigmasweep_tabulate.txt')    
#    
#    with open(latex_filename, 'w') as the_file:
#        the_file.write(latex_string)
#        print 'Latex written to ', latex_filename
        
#def BetaSigmaSweepHarness(regenerate_paths = True,
#                          reestimate = False):  
#    ''' we estimate the true beta for different control perturbations for different values 
#    of the 'true beta and sigma'''
#      
#    start = time.clock();
#    Tf = 16;
#    Npaths = 100; 
#    X0 = 2.;
#    
#    tau_chars = [.2, 1 ,5]; sigmas = [.25, 4];
#    
#    '''generate stochastic paths'''
#    if regenerate_paths:
#        for tau_char in tau_chars:
#            for sigma in sigmas:
#                    simPs = SimulationParams(tau_char = tau_char, sigma = sigma) 
#                    GeneratePathsHarness(Tf, Npaths = Npaths,
#                                         x_0 = X0,
#                                          amax = 1., simPs = simPs) 
#    if reestimate:
#        for tau_char in tau_chars:
#            for sigma in sigmas:
#                    simPs = SimulationParams(tau_char = tau_char, sigma = sigma) 
#                    ''' estimate beta'''    
#                    estimateHarness(Tf, Npaths,simParams = simPs)
# 
#    latexifyBetaSweepResults(Npaths, Tf, tau_chars, sigmas);
#        
#    print 'time_taken', time.clock() - start;   
        
        
if __name__ == '__main__':       
    from pylab import *  
    
    simPs   = SimulationParams(tau_char = 1.)
    Tf = 15.;
    Nblocks = 100; 
    Nhits   = 1000;
#    Nblocks = 10; 
#    Nhits   = 100;
    N_all_hits = Nblocks*Nhits;
    
    'Load OptSoln:'
#    fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = [simPs.mu, simPs.sigma, Tf, 16])
#    fbkSoln = FBKSolution.load('PriorSpread_wide prior')
#    fbkSoln = FBKSolution.load('PriorSpread_narrow prior')
#    'visualize controls (sanity check)'   
#    visualizeControls(simPs,
#                      fbkSoln, 
#                      Tf = fbkSoln._Solver._ts[-1])#, fig_name='opt_vs_crit_vs_max_control_illustrate');

    '''Simulate Nb x Nh passage times'''
#    GeneratePathsHarness(simPs,
#                          Nblocks = Nblocks, 
#                            Nhits = Nhits,
#                            recompute_optimzation=False,
#                            recompute_hitting_simulation=False)
    
    ''' splits the existing set into different blocks x hits compartments'''
#    NblocksNhitsRearrange(simPs, Nblocks,  Nhits )
    
    '''visualize the hitting-time (g(t)) histograms'''
#    visualizePaths(simPs, 
#                   Nblocks = 1, 
#                   Nhits = 1000) #fig_name = 'three_pt_prior')


    '''Inspect the estimator routines for various test cases / convergences:'''
#    estimatorWorkbench(simPs, fbkSoln)  
  

    '''BATCH Parameter Estimation:''' 
#    for Nh in array( [1e5] ).astype(int) :
    for Nh in array( [ 1e2  , 1e3, 1e4, 1e5 ]).astype(int): 
#    for Nh in array( [1e2, 1e3 ]).astype(int): 
        Nb = N_all_hits // Nh;
        'Estimate tau_char:'       
        estimateHarness(simPs, 
                        FBKSolution.load('BatchEstimate_wide-prior'),
                        Nblocks = Nb, 
                        Nhits = Nh,
                        reestimate=False) 
         

    '''The sweep through beta /sigma'''
#    BetaSigmaSweepHarness()
    
    
    
    show()
    
    