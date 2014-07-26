# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from numpy.random import randn, seed  
from copy import deepcopy

from scipy.interpolate.interpolate import interp2d
from scipy.interpolate.fitpack2 import RectBivariateSpline
from matplotlib.patches import FancyArrowPatch, FancyArrow, ArrowStyle
from matplotlib.font_manager import FontProperties
from OUML import get_ts_dWs

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/OU_MIControlSimulator/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/OU_MIControlSimulator'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 32
xlabel_font_size = 40

from OU_FBSolver import FBSolution
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
    
       
class SimulationPaths():
    FILE_EXTENSION = '.sp'
    def __init__(self, ts, xs_block, cs_block, simParams, control_tag):
        self.ts = ts;
        if shape(xs_block)[1] != shape(cs_block)[1]:
            raise RuntimeError('xsBlock, csBLock mismatched sizes')
        self.xs = xs_block;
        self.cs = cs_block;
        self.simParams = simParams;
        self.control_tag = control_tag;
        
    def getT(self):
        return self.ts[-1];
    def getN(self):
        return shape(self.xs)[1]
        
        
    def save(self, file_name=None):
#       path_data = {'path' : self}
        if None == file_name:
            P = self.simParams;
            file_name = self.getDefaultFileName(self.getT(),
                                           self.getN(),
                                           P.mu,
                                           P.tau_char,
                                           P.sigma,
                                           self.control_tag);
                                                                   
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + self.FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
    @classmethod
    def getDefaultFileName(cls, T,N,m,tc, sigma, ctag):
        return 'OUSim_%s_T=%.2f_N=%d_m=%.2f_tc=%.2f_s=%.2f'%(ctag, T,N,m,tc, sigma);
    
    @classmethod
    def load(cls, TNmtcs_ctag=None,file_name=None ):
        ''' not both args can be None!!!'''
        if None == file_name:
            T,N,m,tc, sigma, ctag = [x for x in TNmtcs_ctag]
            file_name = cls.getDefaultFileName(T, N, m, tc, sigma,ctag);

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
##################################################################
##################################################################
   
def visualizeControls(amax = 1, Tf = 16., xmax = 2.,
                      fig_name=None):
    Tbreak = 2.
    def detControl(t, x):        
        if mod(t, Tbreak) >= Tbreak/2.:
            return amax
        else:
            return -amax
    def fbControl(t,x):
        return amax* sign(x);
    
    
    
    ts = arange(.0, Tf, .1);
    xs = arange(-xmax, xmax, .1)
    
    detCs = array([detControl(t, .0) for t in ts]);
    fbCs = fbControl(ts, xs);
    
    # VISUALIZE:
    label_font_size = 24
    xlabel_font_size = 32
    
    figure();
    subplot(121)
    title('Deterministic-in time Bang-Bang')
    plot(ts, detCs); 
    hlines(0, ts[0], ts[-1], colors='k')
    ylabel(r'$\alpha_t$', fontsize = xlabel_font_size)
    xlabel(r'$t$', fontsize = xlabel_font_size)
    ylim((-1.1*amax*array([-1,1])))
    
    subplot(122) 
    title('Feedback  Bang-Bang')
    plot(xs, fbCs); 
    hlines(0, xs[0], xs[-1], colors='k')
    xlabel(r'$X_t$', fontsize = xlabel_font_size)
    ylim((-1.1*amax*array([-1,1])))
        

    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '.pdf')
        print 'saving to ', lfig_name
        savefig(lfig_name)
   
     
def visualizePaths(Tf, Npaths = 10, N_paths_to_visualize=6,
                    simParams = SimulationParams(), fig_name=None):
    
    #Load:
    simPathsList = [];
    ctags =  ['det', 'feedback', 'placebo']
    for rdx, c_tag, in enumerate( ctags ):
        
        TNMtcs_tag = [Tf, Npaths,
                       simParams.mu, simParams.tau_char, simParams.sigma,
                        c_tag]
        
        simPathsList.append(  SimulationPaths.load(TNMtcs_tag) );
        
    
        
    figure( figsize = (17, 10 ));    
    
    N_paths_to_visualize = min(N_paths_to_visualize, Npaths)
    
    alpha_bounds = [amin(simPathsList[0].cs), amax(simPathsList[0].cs)];
    x_bounds = [ceil(amin(simPathsList[1].xs)), floor(amax(simPathsList[1].xs))];
    
    for tdx in xrange(N_paths_to_visualize):
        
        
        for simPaths, c_tag in zip( simPathsList,
                                    ctags):
    
            ts = simPaths.ts;
            xs = simPaths.xs[:, tdx];
            cs = simPaths.cs[:, tdx];
        
            pdx = 1 + 2*tdx
            ax = subplot(N_paths_to_visualize, 2, pdx); hold(True);
            plot(ts, xs);
           
            ax.set_yticks((x_bounds[0], .0, x_bounds[1]))
            if tdx == 0:
                title('$X_t$')
                legend(ctags    )
        
             
            pdx +=1;
            ax = subplot(N_paths_to_visualize, 2, pdx); hold(True)
            plot(ts, cs);
            
            ylim( 1.25*array(alpha_bounds) )
            ax.set_yticks((alpha_bounds[0], .0, alpha_bounds[1]))
            if tdx == 0:
                title(r'$\alpha(t)$')
                legend(ctags)
            if tdx == N_paths_to_visualize-1:
                
                subplot(N_paths_to_visualize, 2, pdx-1); 
                xlabel('$t$' , fontsize = 24)
                subplot(N_paths_to_visualize, 2, pdx); 
                xlabel('$t$', fontsize = 24)
                
    if fig_name != None:
        lfig_name = os.path.join(FIGS_DIR,
                             fig_name + '_amax%d.pdf'%amax(simPathsList[0].cs))
        print 'saving to ', lfig_name
        savefig(lfig_name)
    
    

def GeneratePathsHarness(Tf, dt = .01, Npaths = 10, amax = 1.):
    Tbreak = 2.
    def detControl(t, x):        
        if mod(t, Tbreak) >= Tbreak/2.:
            return amax
        else:
            return -amax 
        
    def fbControl(t,x):
        return amax* sign(x);
    
    def placeboControl(t,x):
        return .0
    
    
    #ASSIGN STRUCT PARAMS:
    simPs = SimulationParams();
    mu, tauchar, sigma = simPs.mu, simPs.tau_char, simPs.sigma;
 
    #The random numbers:
    sqrt_dt = sqrt(dt);  
    
    #the time points:
    ts = r_[arange(.0, Tf, dt), Tf];    
    N_ts = len(ts);
    
    #the Gaussian incrments:
    dWs = randn(N_ts-1, Npaths)* sqrt_dt
     
    #the storage vetors:
    xs_block_det = zeros( (N_ts, Npaths));
    xs_block_fb  = zeros( (N_ts, Npaths));  
    xs_block_plac = zeros( (N_ts, Npaths));
    
    cs_block_det = zeros( (N_ts, Npaths));
    cs_block_fb  = zeros( (N_ts, Npaths));
    cs_block_placebo = zeros( (N_ts, Npaths));

    #the dynamics RHS:    
    def compute_dX(alpha, x, dW):
        return (alpha + (mu - x)/tauchar)*dt + sigma * dW
    ##MAin integratino loop
    def computeTraj(xs, cs, alpha_func, ldWs):
        for t, idx in zip(ts[1:], arange(N_ts)):
            x_prev = xs[idx]
            cs[idx] = alpha_func(t,x_prev)
            xs[idx+1] = x_prev + compute_dX(cs[idx],  x_prev, ldWs[idx])
                    
    
    #The batch integration
    for cFun, x_block, c_block, c_tag in zip([detControl, fbControl, placeboControl],
                                      [xs_block_det,  xs_block_fb, xs_block_plac],
                                      [cs_block_det, cs_block_fb, cs_block_placebo],
                                      ['det', 'feedback', 'placebo']):
        
        for idx in xrange(Npaths):
            computeTraj(x_block[:,idx], c_block[:, idx], cFun, dWs[:, idx]);
            
        (SimulationPaths(ts, x_block, c_block, simPs, c_tag)).save();
     
    visualizePaths(Tf, Npaths, simParams = simPs)


from scipy.optimize import brentq
def calcBetaEstimate(xs, ts, cs):
    'assume the known sigma, mu = 1,0'
    Xn = xs[:-1]
    Xm = xs[1:]
    an = cs[:-1]
    Delta = ts[2]-ts[1]
    N = len(Xn);
            
    
    def f_beta_root(beta):
        outer_term = N/2*(2*Delta*e**(-2*Delta*beta)/(e**(-2*Delta*beta) - 1) + 1/beta)
        
        inner_sum = ( -2*(Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                            Xn)**2*Delta*beta*e**(-2*Delta*beta)/(e**(-2*Delta*beta) - 1)**2 +
                            2*(Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                            Xn)*(Delta*Xm*e**(-Delta*beta) - Delta*an*e**(-Delta*beta)/beta -
                            (e**(-Delta*beta) - 1)*an/beta**2)*beta/(e**(-2*Delta*beta) - 1) -
                            (Xm*e**(-Delta*beta) - (e**(-Delta*beta) - 1)*an/beta -
                            Xn)**2/(e**(-2*Delta*beta) - 1) )
        return outer_term - sum(inner_sum)
        
    
    beta_root, rootResults = brentq(f_beta_root, .01, 100.000, xtol = 1e-3, rtol = 1e-8, full_output=True, disp=False )
    
    if False == rootResults.converged:
        print 'Brentq failed'
        
    return beta_root;


def estimateHarness(Tf, Npaths = 10, 
                    simParams = SimulationParams(), fig_name=None, 
                    ctags = ['det', 'feedback', 'placebo']):     
    #Load:
    simPathsList = [];
     
    for rdx, c_tag, in enumerate( ctags ):
        
        TNMtcs_tag = [Tf, Npaths,
                       simParams.mu, simParams.tau_char, simParams.sigma,
                        c_tag]
        
        simPathsList.append(  SimulationPaths.load(TNMtcs_tag) );
        
    
    betaEsts = dict(zip( ctags, [ [], [], [] ] ));
    
    for tdx in xrange(Npaths): 
        
        for simPaths, c_tag in zip( simPathsList,
                                    ctags ):
            ts = simPaths.ts;
            xs = simPaths.xs[:, tdx];
            cs = simPaths.cs[:, tdx];
            
            b_est = calcBetaEstimate( xs, ts, cs )
            betaEsts[c_tag].append(b_est)
            'sentinel'
            
    
    
#    VISUALIZE:
    figure(); hold(True)

    for x, c_tag, col  in zip( [-1,0 , 1],
                                    ctags,
                                    ['b', 'r', 'g']):
        bs = betaEsts[c_tag]; 
        scatter(x*ones_like(bs), bs, c= col );
        print c_tag, mean(bs), std(bs)
    hlines(1.0, -1,1);
    legend(ctags)
    
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
    
def latexifyResults(Npaths = 1000, Tfs = [8 , 16,32] ):
     
    
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
     

        
        
if __name__ == '__main__':
    from pylab import *    
    
    Tf = 16;
    Npaths = 100; 
    
#    visualizeControls(Tf = 16, fig_name='Fb_vs_det_control_illustrate');
    
    GeneratePathsHarness(Tf, Npaths = Npaths, amax = 1.)

#    visualizePaths(Tf, Npaths = Npaths, N_paths_to_visualize=4,
#                    fig_name = 'det_vs_fb')
#    visualizePaths(Tf, Npaths = Npaths)
    
    
    estimateHarness(Tf, Npaths)
#    estimateHarnessSandbox(Tf, Npaths, fig_name='Tf=%d'%Tf)


#    for Tf in [8,16,32]:
#        GeneratePathsHarness(Tf, Npaths = Npaths, amax = 1.) 
#    latexifyResults()

#    visualizeBetaRoot()
    
    show()
    
    