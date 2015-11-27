# -*- coding:utf-8 -*-
"""
@author: alex
"""

from __future__ import division



from numpy import *
from numpy.random import randn, seed  
from copy import deepcopy

from scipy.interpolate.interpolate import interp2d, interp1d
from scipy.interpolate.fitpack2 import RectBivariateSpline
from matplotlib.patches import FancyArrowPatch, FancyArrow, ArrowStyle
from matplotlib.font_manager import FontProperties
from OUML import get_ts_dWs

from ForwardSolver import ForwardSolver, ForwardSolution
from ControlBasisBox import piecewiseConst
from scipy.optimize.optimize import fminbound

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/HitTime_MI_BetaEstimate/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/HitTime_MI_BetaEstimate'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 32
xlabel_font_size = 40

 
from collections import deque

#PARAMETERS:
#mu  = .0;
#tau_char = 1.;
#sigma = 1.;

class SimulationParams():
    def __init__(self, mu  = .0, tau_char = 1., sigma= 1.0):
        self.mu = mu;
        self.tau_char = tau_char;
        self.sigma = sigma; 
#        %TODO make self.beta = a computed property 

    def printme(self):
        print 'SimPs: m,t,s = %g, %g %g' %( self.mu, self.tau_char, self.sigma )
    
       
class SimulationPaths():
    FILE_EXTENSION = '.sp'
    def __init__(self, thits_blocks, simParams, alphas, alpha_tags, betaEsts = []):
        self.thits_blocks = thits_blocks;
 
        self.alphas = alphas;
        self.alpha_tags = alpha_tags;
        
        self.simParams = simParams;
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

##################################################################

   
     
def visualizePaths(Nblocks , 
                     Nhits ,
                     simPs ,
                  fig_name=None):
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    thits_Blocks = simPaths.thits_blocks;
    alphas = simPaths.alphas;
    
    Nbins = floor(Nhits / 5);

    '''VISUALIZE'''   
    figure( figsize = (17, 10) ); 
    
    for pdx in xrange(0,len(alphas) ):
        thits = thits_Blocks[pdx, 0, :];
        
        
        subplot(len(alphas),1,pdx+1)
        hist(thits, normed=True, bins = Nbins);
        title(r'$\alpha=%.2f$'%alphas[pdx], fontsize = 36);
        
        ylabel('frequency', fontsize = 24)  
        if pdx == len(alphas)-1:
            xlabel('$t$', fontsize = 24)
            
    if fig_name != None:
        lfig_name = os.path.join(FIGS_DIR,
                             fig_name + '_thits_distn.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
    
    

def GeneratePathsHarness(Nblocks = 2,
                         Nhits = 100,
                         Tmax = 25., dt = .001, 
                          amax = 4.,
                          acrit = 1.,
                          simPs = SimulationParams(),
                          x_0 = .0,
                          visualize_paths = True):
 
    
    
    #ASSIGN STRUCT PARAMS:
    mu, tauchar, sigma = simPs.mu, simPs.tau_char, simPs.sigma;
 
    #The random numbers:
    sqrt_dt = sqrt(dt);  
    
    #the time points:   
    N_ts = Tmax/dt;    
    
    #the storage vetors:
    hts_block = empty((3, Nblocks,Nhits)) 

    #the dynamics RHS:    
    def compute_dX(alpha, x, dW):
        return (alpha + (mu - x)/tauchar)*dt + sigma * dW
    X_THRESH = 1.0
    ##MAin integratino loop
    def computeHittingTime(x, alpha, ldWs):
        thit = .0;
        idx = 0;
        for dW in ldWs:
            if x>=X_THRESH:
                break; 
            x+=compute_dX(alpha, x, dW);
            thit+=dt
        return thit
      
    
    #The batch integration
    alphas = [.0, 1.0, 4.0]
    alpha_tags = ['zero', 'crit', 'max']
    for bdx in xrange(Nblocks):
        
        #the common Gaussian incrments:
        dWs = randn(Nhits, N_ts-1)* sqrt_dt;
    
        for adx, (alpha, alpha_tag) in enumerate( zip(alphas, alpha_tags)):
            print adx, alpha, alpha_tag
            
            for hdx in xrange(Nhits):
                ldWs = dWs[hdx,:];
                'compute hitting time'
                hts_block[adx, bdx, hdx] = computeHittingTime(.0, alpha, ldWs);
                if hts_block[adx, bdx, hdx] > Tmax-.1:
                    print hts_block[adx, bdx, hdx]
     
                
        
#    if visualize_paths: 
#        visualizePaths(Tf, Npaths, simParams = simPs)
    (SimulationPaths(hts_block, simPs, alphas, alpha_tags)).save();
    
    

from scipy.optimize import brentq
def calcBetaEstimate(hts, alpha, sigma):
    '''given a set of hitting times hts and known alpha, sigma - 
    calculate the other parameter beta using Max Likelihood'''
    
    'Max Time for the Solver'
   
    
    
    #Solver details:
    dx = .05;
    x_min  = -1.5;
    dt = .025;    
    Tf = amax(hts)+2*dt
    params = [0, sigma]
    
    print  ' Tf=%.2f, alpha=%.2f'%( Tf,alpha)
    #INit Solver:
    S = ForwardSolver(params, 
                      dx, x_min,
                      dt, Tf) 
    
    alphas = alpha*ones_like(S._ts);
    
    def beta_nllk(beta):
        ''' Тhis is the root'''
        Fs = squeeze( S.solve(alphas,
                              [beta]));
        dt = S._ts[1] - S._ts[0]; 
        gs = -diff(Fs[:, -1]) / dt;

        gs_interp = interp1d( S._ts[1:], gs)
        nllk = -sum(log( gs_interp(hts) ) )
#        print beta, nllk
        return  nllk ; 
        
    
    beta_est, nllk_val, ierr, numfunc= fminbound(beta_nllk, .05, 20, full_output = True);
     
#    print beta_est, nllk_val
    if 1 == ierr :
        print 'fminbound hit max fevals'
        
    return beta_est;


def estimateHarness(Nblocks, Nhits, simPs,
                     fig_name=None, reestimate = True):
    
    #Load:
    simPaths = SimulationPaths.load([Nblocks, Nhits, simPs.mu, simPs.tau_char, simPs.sigma, ''])
    
    thits_Blocks = simPaths.thits_blocks;
    
    
    alphas = simPaths.alphas;
    Nalphas = len(alphas);
    print 'Loaded %d Blocks of %d Hitting times for %d controls:'%(thits_Blocks.shape[1], thits_Blocks.shape[2], thits_Blocks.shape[0])
 
    
    betaEsts = empty((Nalphas, Nblocks))
    for adx, alpha in enumerate(alphas):
        for bdx in xrange(Nblocks):
            hts = squeeze(thits_Blocks[adx, bdx,:]);
            
            betaEsts[adx, bdx] = calcBetaEstimate(hts, alpha, simPaths.simParams.sigma);
            
    
    
    simPaths.betaEsts = betaEsts;
    
    simPaths.save();
    
    print betaEsts
    
    return betaEsts
            
    
    
    
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
    
    ctags = simPaths.alpha_tags
       
#    VISUALIZE:
    figure(figsize =(8,10)) ; hold(True)
    for adx, x, c_tag, col  in zip( range(len(simPaths.alphas)),
                                   [-1,0 , 1],
                                    ctags,
                                    ['b', 'r', 'g']):
        bs = betaEsts[adx,:]; 
        scatter(x*ones_like(bs), bs, c = col );
        print c_tag, mean(bs), std(bs)
    hlines(1./simPaths.simParams.tau_char, -1, 1);
    legend(ctags)   
    
    #LATEXIFY:
    
    print 'latexifying:'
    
    latex_string = r'control type & mean($\hat\beta$) & std($\hat\beta$) \\ \hline '
    
    for adx, x, c_tag, col  in zip( range(len(simPaths.alphas)),
                                   [-1,0 , 1],
                                    ctags,
                                    ['b', 'r', 'g']):
        bs = betaEsts[adx,:]; 
        scatter(x*ones_like(bs), bs, c = col );
        latex_string+= r'%s & %.2f & %.2f \\' %(c_tag, mean(bs), std(bs))
        
    print latex_string
        
    latex_filename = os.path.join(FIGS_DIR,
                                   'beta_hit_time_%d.txt'%(Nhits))    
     
    with open(latex_filename, 'w') as the_file:
        the_file.write(latex_string)
        print 'Latex written to ', latex_filename
       
    
    
  
    
def latexifyResults(Npaths = 1000, Tfs = [8, 16,32] ):
    bEstimates  = {};
    ctags = []
    for Tf in Tfs:
        bEsts = estimateHarness(Tf, Npaths = Npaths);
        
        for ctag, bs in bEsts.iteritems():
            bEstimates[(Tf, ctag)] = (mean(bs), std(bs))
            
        ctags = bEsts.keys();
        
        
    #latexify:
    print 'latexifying:'
    print '$T_f$:'
    for Tf in Tfs:
        print ' & %d'%Tf
    print r'\\'    
    for ctag in ctags:
        print ctag, ':'
        for Tf in Tfs:  
            m,s =  bEstimates[(Tf, ctag)]        
            print  ' & (%.2f, %.2f)'%(m,s)    
        print r'\\'
     

 
def latexifyBetaSweepResults(Npaths, Tf ,
                             tau_chars, sigmas):
    from scipy.stats import nanmean, nanstd
    
    bEstimates  = {};
    ctags = []
    for tau_char in tau_chars:
            for sigma in sigmas:
                simPs = SimulationParams(tau_char = tau_char, sigma = sigma)
                bEsts = estimateHarness(Tf, Npaths = Npaths, simParams= simPs);
        
                for ctag, bs in bEsts.iteritems():
                    bEstimates[(tau_char, sigma, ctag)] = (nanmean(bs),
                                                           nanstd(bs),
                                                           sum(isnan(bs)))
                ctags = bEsts.keys();  
                
                  
    #LATEXIFY:
    
    print 'latexifying:'
    
    latex_string = ''
    latex_string+= r'$(\beta_{true}, \sigma)$:'
    for tau_char in tau_chars:
            for sigma in sigmas:
                latex_string += ' & (%.2f,%.2f)'%(1/tau_char, sigma)
    latex_string+= r'\\ \hline  '    
    for ctag in ctags:
        latex_string+= ctag + ':'
        for tau_char in tau_chars:
            for sigma in sigmas:
                m,s,nnan =  bEstimates[(tau_char, sigma, ctag)]        
#                latex_string+=  ' & (%.2f, %.2f, %d)'%(m,s, nnan)    
                latex_string+=  ' & (%.2f, %.2f )'%(m,s)    
        latex_string+= r'\\'
    
    print latex_string
        
    latex_filename = os.path.join(FIGS_DIR, 'betasigmasweep_tabulate.txt')    
    
    with open(latex_filename, 'w') as the_file:
        the_file.write(latex_string)
        print 'Latex written to ', latex_filename
        
def BetaSigmaSweepHarness(regenerate_paths = True,
                          reestimate = False):  
    ''' we estimate the true beta for different control perturbations for different values 
    of the 'true beta and sigma'''
      
    start = time.clock();
    Tf = 16;
    Npaths = 100; 
    X0 = 2.;
    
    tau_chars = [.2, 1 ,5]; sigmas = [.25, 4];
    
    '''generate stochastic paths'''
    if regenerate_paths:
        for tau_char in tau_chars:
            for sigma in sigmas:
                    simPs = SimulationParams(tau_char = tau_char, sigma = sigma) 
                    GeneratePathsHarness(Tf, Npaths = Npaths,
                                         x_0 = X0,
                                          amax = 1., simPs = simPs) 
    if reestimate:
        for tau_char in tau_chars:
            for sigma in sigmas:
                    simPs = SimulationParams(tau_char = tau_char, sigma = sigma) 
                    ''' estimate beta'''    
                    estimateHarness(Tf, Npaths,simParams = simPs)
 
    latexifyBetaSweepResults(Npaths, Tf, tau_chars, sigmas);
        
    print 'time_taken', time.clock() - start;   
        
if __name__ == '__main__':    
    
    raise Exception('This File analytics are deprecated')   
    from pylab import *  
    
    Nblocks = 100; 
    Nhits   = 1000;
    simPs   = SimulationParams(tau_char = 1.)
    
        
#    visualizeControls(Tf = 16, fig_name='Fb_vs_det_control_illustrate');

    '''Simulate N passage times'''
#    GeneratePathsHarness(Nblocks = Nblocks, 
#                         Nhits = Nhits,
#                         simPs = simPs)
#    
#    visualizePaths(Nblocks = Nblocks, 
#                         Nhits = Nhits,
#                         simPs = simPs)
#
#    '''Do the actual parameter estimation''' 
#    estimateHarness(Nblocks = Nblocks, 
#                         Nhits = Nhits,
#                         simPs = simPs) 
    
#    '''visualize and latexify results'''
#    postAnalysis(Nblocks = Nblocks, 
#                         Nhits = Nhits,
#                         simPs = simPs)
     

    '''The sweep through beta /sigma'''
#    BetaSigmaSweepHarness()
    
    
    show()
    
    