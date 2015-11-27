# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
import numpy
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from scipy.optimize.zeros import brentq
from copy import deepcopy
from scipy.optimize.optimize import fminbound
from scipy import interpolate 

from matplotlib.font_manager import FontProperties

import ext_fpc

from HitTime_MI_Beta_Estimator import SimulationParams
#from PathSimulator import ABCD_LABEL_SIZE
from TauParticleEnsemble import TauParticleEnsemble
from scipy.interpolate.interpolate import interp1d

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/FP_Adjoint/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/FP_Adjoint'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
        
import time


label_font_size = 32
xlabel_font_size = 40
ABCD_LABEL_SIZE = 28

def deterministicControlHarness(params,
                                 Tf = 1.5,    
                                   alpha_bounds = (-2., 2.)):
    mu, tau_char = params[0], params[1]
    xth = 1.0
#    \frac{\xth}{\tc(1 - e^{-\T/\tc})} - \m$$
    alpha_constant = xth / (tau_char * (1. - exp(-Tf/tau_char) )) - mu  
    
    return alpha_constant
  
#import ext_fpc
class FPAdjointSolver():    
    TAUCHAR_INDEX = 1;
    BETA_INDEX = 2
    
    def __init__(self, dx, dt, Tf,
                  x_min, x_thresh = 1.0,
                   Tf_optimization=None):
        #DISCRETIZATION:
        self.rediscretize(dx, dt, Tf, x_min, x_thresh)
        
        '''Tf_optimization ~ the last time for which the control is optimizable, 
                            This is the time-point from which 
                            the adjoint rolls backwards'''
        if None == Tf_optimization:
            Tf_optimization = Tf/2.0;
        self.setTfOpt(Tf_optimization);

    #Grid management routines:    
    def refine(self, dx_factor, dt_factor):
        'dx_factor<1, dt_factor<1'
        x_min = self._xs[0];
        x_thresh = self._xs[-1];
        Tf = self._ts[-1];
        self._xs, self._dx = self._space_discretize(self._dx*dx_factor, x_min, x_thresh);
        self._ts, self._dt = self._time_discretize(Tf,self._dt*dt_factor);
    
    def rediscretize(self, dx, dt, Tf, x_min, x_thresh = 1.0):
        self._xs, self._dx = self._space_discretize(dx, x_min, x_thresh)
        self._ts, self._dt = self._time_discretize(Tf,dt)
        print 'nodes count = %.2g GB per forward-backward solution'%(len(self._xs)*len(self._ts)*8*2*1e-9)
    
    def _space_discretize(self, dx, x_min, x_thresh = 1.0):
        xs = arange(x_thresh, x_min - dx, -dx)[-1::-1];
        return xs,dx
#        num_steps = ceil( (x_thresh - x_min) / dx)
#        xs = linspace(x_min, x_thresh, num_steps)
#        dx = xs[1]-xs[0];
#        return xs, dx
    
    def _time_discretize(self, Tf, dt):
        num_steps = ceil( Tf/ dt )
        ts = linspace(.0, Tf, num_steps)
        dt = ts[1]-ts[0];
        return ts, dt
    def getOptTs_indices(self):
        return where(self._ts<=self.Tf_optimization)[0];
    def getOptTs(self):
#        return self._ts[self.getOptTs_indices()];
        return self._ts;
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts, self._dt = self._time_discretize(Tf, self._dt)
    def setTfOpt(self,Tf_optimization):
        if Tf_optimization>self.getTf():
            print('Warning - trying to set TfOPT > Tf - NOT allowed')
        else:
            self.Tf_optimization = Tf_optimization;    
        
    def getXthresh(self):
        return self._xs[-1]
    def getXmin(self):
        return self._xs[0]
    def setXmin(self, x_min):
        self._xs, self._dx = self._space_discretize(x_min, self._dx)        

    @classmethod
    def calculate_xmin(cls, alpha_bounds, tau_chars, mu_sigma, num_std = 2.0):     
        XMIN_AT_LEAST = -0.5;   
        mu,   sigma = [x for x in mu_sigma]
        tc_max = amax(tau_chars)
        alpha_min = max(alpha_bounds[0], -1.);
        xmin = tc_max*(alpha_min +mu) - num_std*sigma*sqrt(tc_max/2.0);
        return min([XMIN_AT_LEAST, xmin])
    @classmethod
    def calculate_dx(cls, alpha_bounds, tau_chars, mu_sigma, xmin, factor = 0.5, xthresh = 1.0):
        #PEclet number based calculation:
        mu,   sigma = [x for x in mu_sigma]
        tc_min = amin(tau_chars)
        max_speed = max(alpha_bounds) + (abs(mu) + max( [abs(xmin), abs(xthresh)])) / tc_min;     
        return factor * (sigma / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, tau_chars,
                      mu_sigma, dx, xmin, factor=2., xthresh = 1.0):
        ''' dt = factor (dx / max_speed)'''
        mu, sigma = [x for x in mu_sigma]     
        tc_min = amin(tau_chars)
        max_speed = max(alpha_bounds) + max( abs( mu - array([ xmin, xthresh])/tc_min)) ;
        return factor * (dx / max_speed) 
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
    def _num_backward_steps(self):
        return len(self._ts)
#        return self.getOptTs().shape[0];     
    
    def _getICs(self, xs, alpha0, sigma):
        #WARNING! TODO: HOw do you choose 'a' correctly! 
        a = 0.1;
        pre_ICs = exp(-xs**2 / a**2) / (a * sqrt(pi))
        ICs = pre_ICs / (sum(pre_ICs)*self._dx) 
        return ICs
    
    def _getAdjointBCs(self, tau_char_weights, fs, sigma):            
    
#        return self._getOldWrongBCs(tau_char_weights ,fs);
    
        return self._getNewMaybeRightBCs(tau_char_weights, fs)
    
    
    def _getNewMaybeRightBCs(self, tau_char_weights, fs ):    
        di_x_fs  = -squeeze(fs[:, -1,:] - fs[:, -2,:]) / self._dx;    

        di_x_fs_mean = dot(tau_char_weights, di_x_fs);
        
        dixfs_over_dixfs_mean_bayesian = di_x_fs / di_x_fs_mean;
        ''' deal with 0/0:'''
        zero_ids = di_x_fs_mean<1e-8;        
        dixfs_over_dixfs_mean_bayesian[:, zero_ids] = 1.;
        
        '''return'''
        bcs = -log(dixfs_over_dixfs_mean_bayesian);
        
        return bcs
    
    def _getOldWrongBCs(self, tau_char_weights, fs ):    
        di_x_fs  = -squeeze(fs[:, -1,:] - fs[:, -2,:]) /\
                           self._dx;    

        di_x_fs_mean = dot(tau_char_weights, di_x_fs);
        
        dixfs_over_dixfs_mean_bayesian = di_x_fs / di_x_fs_mean;
        ''' deal with 0/0:'''
        zero_ids = di_x_fs_mean<1e-8;        
        dixfs_over_dixfs_mean_bayesian[:, zero_ids] = 1.;
        
        '''return'''
        bcs = -log(dixfs_over_dixfs_mean_bayesian) - 1 + dixfs_over_dixfs_mean_bayesian
        
        return bcs

    ###########################################
    def solve_hittime_distn_per_parameter(self, tau_char,
                                          mu_sigma,
                                          alphas,
                                          force_positive=False,
                                          visualize=False):
        'Forward Distn'
        fs = squeeze( self._fsolve( [tau_char], [1.0],
                                    mu_sigma, alphas, visualize));
        'hitting time distn:'                            
        gs = self.hittime_distn_via_flow(mu_sigma[1], fs)
#        gs = self.hittime_distn_via_conservation(fs);
       
        if visualize:
            figure(); plot(self._ts, gs); title('HItting Time distn'); xlabel('t'); ylabel('t');
        
        if force_positive:
            gs[gs<=0] = 1e-8;
        return gs;
    
    def hittime_distn_via_flow(self, sigma, fs):
        'Diffusion Const'
        D = sigma*sigma / 2.0;
        'g is the outflow at the upper boundary:'
        return D*(fs[-2,:] - fs[-1,:])/self._dx;
    
    def hittime_distn_via_conservation(self, fs):
        Ntaus = fs.shape[0];
        return r_[0, -diff(sum(fs, axis=0))*self._dx/self._dt];        
    
    '''The main (solve) routine of the solver. 
    RETRUNS: xs,ts,   fs,ps,   -J,grad_H'''
    def solve(self, tau_chars, tau_char_weights, mu_sigma,
               alphas,
               visualize=False):
         
        '''the forward/adjoint states:''' 
        fs = self._fsolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas, visualize)
        ps = self._psolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas, fs, visualize)

        
        '''the Mutial Information Objective:''' 
        J = self.calcObjective(tau_char_weights, fs)
        
        '''the Hamiltonian gradient:''' 
        grad_H = self.calcObjectiveGradient(tau_char_weights, fs, ps)
        
        return self._xs, self._ts, fs, ps, -J, grad_H
    
    '''Only solve for the Objective
     (i.e. compute the hitting times distn and then the Mutual Information)
     RETURNS: fs, -J'''    
    def solveObjectiveOnly(self, tau_chars, tau_char_weights, mu_sigma,
                            alphas,
                            visualize=False):         
        '''the forward states:''' 
        fs = self._fsolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas, visualize);
        
        '''the Mutual Information Objective:''' 
        J = self.calcObjective(tau_char_weights, fs);
                
        return fs, -J;
    
    
    ###########################################
    def calcObjectiveGradient(self, tau_char_weights, fs, ps):         
        '''Compute The Hamiltonian Gradient:
        
        NOTE: WE need to divide by dx in approximating the derivative and 
        multiply by dx in approximating the integral so we just drop both, 
        since (dx/dx = 1)''' 
         
        'diff ps vs. xs:'
        dxps = diff(ps, axis=1);
        
        'subset to optimization-interval fs:'        
#        var_fs = fs[:, 1:,self.getOptTs_indices()];
        var_fs = fs[:, 1:,:];
        
        'Compute  int (di_x p \cdot f)'
        dxp_times_f = squeeze( sum(var_fs*dxps, axis=1) )

        'Grad_H = sum w_t \cdot  int (\di_x p \cdot f)'
        grad_H = -dot(tau_char_weights, dxp_times_f);
            
        return grad_H;
        
    def calcObjective(self, tau_char_weights, fs):
        '''Calculate the Mutual Information between the hitting-time
           gs (\di_x fs_{xthresh}) distribution and the prior (discrete weights)'''
        xs, ts = self._xs, self._ts;
        dx, dt = self._dx, self._dt;
                
        di_x_fs = -squeeze(fs[:, -1,:] - fs[:, -2,:]) /\
                           (xs[-1] - xs[-2])  ;

#        di_x_fs = -diff(sum(fs, axis=1)); 
        
        di_x_fs_mean = dot(tau_char_weights, di_x_fs);
        
        
        
        dixfs_over_dixfs_mean_bayesian = di_x_fs / di_x_fs_mean;
        '''manually set 0/0 to 0'''
        zero_ids = di_x_fs < 1e-8;        
        dixfs_over_dixfs_mean_bayesian[zero_ids] = 1.;
        
        integrand = dot(tau_char_weights, 
                        di_x_fs*log(dixfs_over_dixfs_mean_bayesian))
        
        '''return'''                 
        J  = sum(integrand)*dt
             
        return J
    
    'The Forward Solve interface'
    def _fsolve(self,  tau_chars, tau_char_weights,
                    mu_sigma, alphas, 
                    visualize=False):
        ''' level of indirection'''
        if visualize:
            return self._fsolve_Py(tau_chars, tau_char_weights,
                               mu_sigma, alphas, 
                               visualize)
        else:
            return self._fsolve_C(tau_chars, tau_char_weights,
                               mu_sigma, alphas, 
                               visualize)
            
    'The Backward Solve interface'            
    def _psolve(self, tau_chars, tau_char_weights, mu_sigma,
                 alphas, fs, visualize=False):
        'Interface to the underlying P(adjoint)-solver'
        return self._psolve_C(tau_chars, tau_char_weights,
                               mu_sigma, alphas, fs,
                               visualize)
#        return self._psolve_Py(tau_chars, tau_char_weights,
#                               mu_sigma, alphas, fs,
#                               visualize)
    
    def _psolve_Py(self, tau_chars, tau_char_weights, mu_sigma,
                   alphas, fs,
                   visualize=False):
        'rip params:'
        mu, sigma = [x for x in mu_sigma]
        
        'rip discretization grid:'
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self.getOptTs();
        
        if visualize:
#            print 'tauchar = %.2f,  sigma = %.2f,' %(tauchar, sigma)
#            print 'amax = %.2f,'%alpha_max
            print 'Tf = %.2f' %self.getTf()
            print 'xmin = %.f, dx = %f, dt = %f' %(self.getXmin(), dx,dt)
        
        num_weights = len(tau_char_weights);
#        fs_mean = tensordot(fs , tau_char_weights, axes=[0,0]);
        
        #Allocate memory for solution:
        ps = zeros((num_weights, 
                    self._num_nodes(),
                    self._num_backward_steps() ));
                    
        #Impose TCs: automoatic they are 0
#        ps[:,-1] = self._getTCs(xs, alpha_max+mu, tauchar, sigma)
        
        'Impose BCs at upper end:' 
        ps[:, -1, :] = self._getAdjointBCs(tau_char_weights,
                                           fs, 
                                           sigma);
            
        'Reset the TCs'
        ps[:, :, -1] = 0;
        if visualize:
            figure()
            subplot(311)
            plot(xs, ps[0, :,-1]); 
#            title(r'$\alpha=%.2f, \tau=%.2f, \sigma=%.2f$'%(alphas[-1], tauchar, sigma) +
#                   ':TCs', fontsize = 24);
            xlabel('x'); ylabel('p')
             
            subplot(312)
            plot(ts, ps[0, -1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('p')
            
            subplot(313)
            plot(ts, alphas);
            title('Control Input', fontsize = 24) ; xlabel('t'); ylabel(r'\alpha')
        
        #Solve it using C-N/C-D:
        D = sigma * sigma / 2.; #the diffusion coeff
        dx_sqrd = dx * dx;
        
        #Allocate mass mtx:    
        active_nodes = self._num_nodes() - 1
        M = lil_matrix((active_nodes, active_nodes));
        
        #Centre Diagonal:        
        e = ones(active_nodes);
        d_on = D * dt / dx_sqrd;
        
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        
        soln_fig = None;  
        if visualize:
            soln_fig = figure()
        
        for pdx, tau_char in enumerate(tau_chars):
            for tk in xrange(self._num_backward_steps() - 2, -1, -1):
                #Rip the forward-in-time solution:
                p_forward = ps[pdx, :, tk+1];
    
                #Rip the control:
                alpha_forward = alphas[tk+1]
                alpha_current = alphas[tk]
                
                #Calculate the velocity field
                U_forward = (alpha_forward + (mu - xs[1:-1])/ tau_char)
                U_current = (alpha_current + (mu - xs[1:-1])/ tau_char)
                
                #Form the RHS:
                L_forward = U_forward*(p_forward[2:] - p_forward[:-2]) / (2.* dx) + \
                            D        *(p_forward[2:] - 2*p_forward[1:-1] + p_forward[:-2] ) / dx_sqrd;  
                
                #Impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
                RHS = r_[0.,
                         p_forward[1:-1] + 0.5 * dt * L_forward];
                
                #Reset the Mass Matrix:
                u =  U_current / (2*dx);
                d_off = D / dx_sqrd;
                        
                #Lower Diagonal
                L_left = -0.5*dt*(d_off - u[:-1]);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal:
                L_right = -0.5*dt*(d_off + u[:-1]);
                M.setdiag(r_[NaN,
                             L_right], 1);
                             
                #Bottom BCs:
                M[0,0] = -1.; M[0,1] = 1.;
                
                #add the terms coming from the upper BC at the backward step to the end of the RHS
                p_upper_boundary = ps[pdx, -1, tk];
                RHS[-1] += 0.5*dt*(d_off * p_upper_boundary + \
                                   u[-1] * p_upper_boundary )
                
                #Convert mass matrix to CSR format:
                Mx = M.tocsr();            
                #and solve:
                p_current = spsolve(Mx, RHS);
                
                #Store solutions:
                ps[pdx, 0:-1, tk] = p_current;
                              
                if visualize:
                    mod_steps = 40;  num_cols = 4;
                    num_rows = ceil(double(self._num_steps())/num_cols / mod_steps) + 1
                    
                    step_idx = self._num_steps() - 2 - tk;
                    
                    if 0 == mod(step_idx,mod_steps) or 0 == tk:
                        plt_idx = 1 + floor(tk / mod_steps) + int(0 < tk)
                        ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                        ax.plot(xs, ps[0, :,tk], label='k=%d'%tk); 
                        if self._num_steps() - 2 == tk:
                            ax.hold(True)
                            ax.plot(xs, ps[0, :,tk+1], 'r', label='TCs')
                        ax.legend(loc='upper left')
    #                        ax.set_title('k = %d'%tk); 
                        if False : #(self._num_steps()-1 != tk):
                            ticks = ax.get_xticklabels()
                            for t in ticks:
                                t.set_visible(False)
                        else:
                            ax.set_xlabel('$x$'); ax.set_ylabel('$p$')
                            for t in ax.get_xticklabels():
                                t.set_visible(True)
                         
        #Return:
        if visualize:
            for fig in [soln_fig]:
                fig.canvas.manager.window.showMaximized()
        return ps
        
    def _fsolve_Py(self,  tau_chars, tau_char_weights,
                            mu_sigma, alphas, 
                             visualize=False):
        ''' returns fs[tau_ids, x_ids, t_ids];
        NOTE: tau_char_weights is irrelevant, I don't know why it's in the arg-list,
             it's not used''' 
        mu, sigma = [x for x in mu_sigma];
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'tauchar, mu, sigma = ' , tau_chars, mu, sigma
            print 'Tf = %.2f' %self.getTf()
            print 'xmin = %.f, dx = %f, dt = %f' %(self.getXmin(), dx,dt)
        num_weights = len(tau_char_weights);
        #Allocate memory for solution:
        fs = zeros((num_weights, 
                    self._num_nodes(),
                    self._num_steps() ));
        'WARNING: Unfortunately, we use the var-order: [taus, xs, ts], whereas it should be [taus, ts, xs] - for legacy reasons' 
                    
        #Impose Dirichlet BCs: = Automatic 
        #Impose ICs: 
        for idx in xrange(num_weights):
            fs[idx, :, 0] = self._getICs(xs, alphas[0], sigma)
            
        
        #Solve it using C-N/C-D:
        D = sigma * sigma / 2.; #the diffusion coeff
        dx_sqrd = dx * dx;
        
        #Allocate mass mtx:    
        active_nodes = self._num_nodes() - 1
        M = lil_matrix((active_nodes, active_nodes));
        
        #Centre Diagonal:        
        e = ones(active_nodes);
        d_on = D * dt / dx_sqrd;
        
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        
        soln_fig = None;  
        if visualize:
            soln_fig = figure();

        for tk in xrange(1, self._num_steps()):
            #Rip the control:
            alpha_prev = alphas[tk-1]
            alpha_next = alphas[tk]
            
            for pdx, tau_char in enumerate(tau_chars):
                #Rip the forward-in-time solution:
                f_prev = fs[pdx, :,tk-1];      
                          
                #Calculate the velocity field
                U_prev = ( alpha_prev + (mu - xs/ tau_char) )
                U_next = ( alpha_next + (mu - xs/ tau_char) )                
                
                #Form the RHS:
                L_prev = -(U_prev[2:]*f_prev[2:] - U_prev[:-2]*f_prev[:-2]) / (2.* dx) + \
                          D * diff(f_prev, 2) / dx_sqrd;  
                
                #impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
                RHS = r_[0.,
                         f_prev[1:-1] + .5 * dt * L_prev];
                
                #Reset the Mass Matrix:
                #Lower Diagonal
                u =  U_next / (2*dx);
                d_off = D / dx_sqrd;
                        
                L_left = -.5*dt*(d_off + u[:-2]);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal
                L_right = -.5*dt*(d_off - u[2:]);
                M.setdiag(r_[NaN,
                             L_right], 1);
                #Bottome BCs:
                M[0,0] = -U_next[0] - D / dx;
                M[0,1] = D / dx;
                
                #add the terms coming from the upper BC at the backward step to the end of the RHS
                #RHS[-1] += 0 #Here it is 0!!!
                
                #Convert mass matrix to CSR format:
                Mx = M.tocsr();            
                #and solve:
                f_next = spsolve(Mx, RHS);
                
                #Store solutions:
                fs[pdx, 0:-1, tk] = f_next;
            
                              
                if visualize:
                    mod_steps = 20;  num_cols = 4;
                    num_rows = ceil(double(self._num_steps())/num_cols / mod_steps)
                    
                    step_idx = tk;
                    
                    if 0 == mod(step_idx,mod_steps) or 1 == tk:
                        plt_idx = floor(tk / mod_steps) + 1
                        ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                        ax.plot(xs, fs[0, :,tk], label='k=%d'%tk); 
                        if 1 == tk:
                            ax.hold(True)
                            ax.plot(xs, fs[0, :,tk-1], 'r', label='ICs')
                        ax.legend(loc='upper left')
    #                        ax.set_title('k = %d'%tk); 
                        if False : #(self._num_steps()-1 != tk):
                            ticks = ax.get_xticklabels()
                            for t in ticks:
                                t.set_visible(False)
                        else:
                            ax.set_xlabel('$x$'); ax.set_ylabel('$f$')
                            for t in ax.get_xticklabels():
                                t.set_visible(True)
            if amax(fs[:, :, tk])< 1e-12:
                break
         
        #Return:
        if visualize:
            lower_flow = D*(fs[0,1,:]- fs[0,0,:])/self._dx + (xs[0]/tau_char + alphas)*fs[0,0,:];
            
            figure() 
            subplot(211)
            plot(ts, lower_flow); ylim(1e-4*array([-1,1]))
            title('BCs at xmin (flow)', fontsize = 24) ; xlabel('t'); ylabel( r'$\phi_{xmin}$')
            subplot(212)
            plot(ts, fs[0, -1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('f'); ylim([-.1,.1])
            
            for fig in [soln_fig]:
                fig.canvas.manager.window.showMaximized()                 
        return fs
    
    
    '''interface to the adjoint C solver'''
    def _psolve_C(self, tau_chars, tau_char_weights,
                    mu_sigma, alphas, fs,
                    visualize=False):
        
        'return array:'
        array_shape = ( len(tau_chars), 
                        self._num_nodes(),
                        self._num_backward_steps());        
        ps = empty(array_shape);
        
        ''' get the BCs - main link with fs '''
        BCs = self._getAdjointBCs(tau_char_weights,
                                   fs, 
                                    mu_sigma[1] );
        
        'time-steps for the solve:'
        lts = self.getOptTs();        
        
        for tdx, tau in enumerate(tau_chars):        
                
            mu_tau_sigma = array([mu_sigma[0], tau, mu_sigma[1]]);
            'main ext.lib call: for some bizarre reason you need to wrap arrays inside an array() call:'            
            lp = ext_fpc.solve_p(mu_tau_sigma,                                 
                                 array(alphas),
                                 array(lts),
                                 array(self._xs),
                                 array(squeeze(BCs[tdx,:])));
                                  
            ps[tdx,:,:] = lp;
            
        return ps
        
        
    '''Interface to the C routines for the forward FP soln'''
    def _fsolve_C(self,  tau_chars, tau_char_weights,
                    mu_sigma, alphas, 
                    visualize=False):
#        fs = self._fsolve(tau_chars, tau_char_weights,
#                          mu_sigma, alphas); 
        array_shape = ( len(tau_chars), self._num_nodes(), self._num_steps() )
        fs = empty(array_shape);
        
        for tdx, tau in enumerate(tau_chars):        
                
            mu_tau_sigma = array([mu_sigma[0], tau, mu_sigma[1]]);
            'main ext.lib call: for some bizarre reason you need to wrap arays inside an array() call:'            
            lf = ext_fpc.solve_f(mu_tau_sigma,
                                 array(alphas),
                                 array(self._ts),
                                 array(self._xs));
                                  
            fs[tdx,:,:] = lf;
            
        return fs

'''A fagtory constructor to return a by-default parameterized Solver for a
 model-parameter-set'''    
def generateDefaultAdjointSolver(tau_chars,
                                 mu_sigma,
                                 alpha_bounds = [-2,2],
                                 Tf=16.0,
                                 Tf_opt = None):
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma, num_std = 2.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin, factor=0.5)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma, dx, xmin, factor = 2.0)
    
    #Set up solver
    lS =  FPAdjointSolver(dx, dt, Tf, xmin, 
                          Tf_optimization=Tf_opt);
    print 'Solver params: xmin, dx, dt, Tf, Tfopt', lS.getXmin(), lS._dx, lS._dt, lS.getTf(), lS.Tf_optimization;
    
    return lS;

def visualizeAdjointSolver(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the v solution:
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=True)
    
    Fth = sum(fs, axis = 0)*S._dx
    figure()
    plot(ts, Fth); xlabel('t'); ylabel('Fth')
    
    #Visualize:
    from mpl_toolkits.mplot3d import Axes3D
    for vs, tag in zip([fs, ps],
                       ['f', 'p']):
        az = -45; #    for az in arange(-65, -5, 20):
        fig = figure();
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev = None, azim= az)
        X, Y = np.meshgrid(ts, xs)
        ax.plot_surface(X, Y, vs, rstride=4, cstride=4, cmap=cm.jet,
                                   linewidth=0, antialiased=False)
        xlabel('t', fontsize = 18); ylabel('x',fontsize = 24)
        title('$'+tag +'(x,t)$', fontsize = 36);    
        get_current_fig_manager().window.showMaximized()



'''relic from the OptSpike project:'''
def stylizedVisualizeForwardAdjoint(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                        fig_name = None):
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the fs, ps :
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps,
                                                visualize=False)
    
    figure()
    plot(xs, fs[:, 0], linewidth = 3); 
    xlabel('$x$', fontsize = 16); ylabel('$f$', fontsize = 16)
    get_current_fig_manager().window.showMaximized()
    file_name = os.path.join(FIGS_DIR, fig_name + '_f.png')
    print 'saving to', file_name
    savefig(file_name)
    
    
    figure()
    plot(xs, ps[:, -1], linewidth = 3); 
    xlabel('$x$', fontsize = 16); ylabel('$p$', fontsize = 16)
    get_current_fig_manager().window.showMaximized()
    file_name = os.path.join(FIGS_DIR, fig_name + '_p.png')
    print 'saving to', file_name
    savefig(file_name)
    
 
    
def calculateOutflow(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.)):
    '''What does this function do???!!!!'''
    
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the f,p,J solution:
    xs, ts, fs, ps, J, minus_gradH =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
    
    #the
    D = tb[1]**2/2.0
    upwinded_outflow = -D*(-fs[-2,:]) /S._dx
    central_outflow  = -D*(-fs[-2,:]) /(2*S._dx)
    
    upwinded_cumulative_outflow = cumsum(upwinded_outflow)*S._dt;
    central_cumulative_outflow = cumsum(central_outflow)*S._dt;
    
    remaining_mass =sum(fs, axis=0)*S._dx
    
    upwinded_conservation = remaining_mass +upwinded_cumulative_outflow
    central_conservation = remaining_mass   +central_cumulative_outflow
    
    figure(); hold(True)
    plot(ts, upwinded_conservation, 'b', label=r'mass + upwinded outflow')
    plot(ts, central_conservation, 'g', label='mass+central outflow'); 
    plot(ts, sum(fs, axis=0)*S._dx, 'r', label='mass');
    xlabel('t');    legend(loc='upper left')



def timeAdjointSolver(tb = [.5, 1.25], Tf = 1.5, energy_eps = .1, alpha_bounds = (-2., 2.)):
    import time
    
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    start = time.clock()
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    #the f,p solution:
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=False)
  
    end = time.clock()
    print 'compute time = ',end-start, 's'
    

###############################################################################
def SingleSolveHarness(tau_chars, tau_char_weights, mu_sigma,  
                       alpha_bounds=(-2,2),
                       Tf = 16, ts=None, Tfopt=10) :
    print('Model Params: m,b, ts', mu_sigma, tau_chars)
    'Define Solver:'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma,  num_std= 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin, factor = 2.0)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma,
                                      dx, xmin, factor = 4.0)
    
    'Warning: Hard-coding dx,dt:'
    dx = 0.02;
    dt = 0.005;
    
    print 'Solver params: xmin, dx, dt', xmin,dx,dt
    print 'Control bounds', alpha_bounds    
    S = FPAdjointSolver(dx, dt, Tf, xmin, Tf_optimization=Tfopt)
    print 'Tf, Tf_opt', S.getTf(), S.Tf_optimization;
    
    'Define Applied Control:'
#    alpha_current = 2.0 * tanh( 2*(S._ts - Tf*0.5 ) );    
    alpha_current =  (4.0) * (S._ts /S.Tf_optimization) - 2.0
#    alpha_current =  zeros_like(S._ts) + alpha_bounds[1]*(S._ts>S.Tf_optimization)    
    alpha_current[where(alpha_current>alpha_bounds[1])] = alpha_bounds[1];
    
#    alpha_max = alpha_bounds[1];
#    alpha_current = zeros_like(S._ts) + alpha_max*(S._ts>S.Tf_optimization);    
    
    print 'fs+ps size = 2x%.2f MB'%(len(tau_char_weights)*len(S._ts)*len(S._xs)*8*1e-6)


    '''MAIN CALL:'''
    xs, ts, fs, ps, J, grad_H =\
            S.solve(tau_chars, tau_char_weights, mu_sigma, alpha_current,
                    visualize=False);     
    
        
    'Make Simple Increment:'
    alpha_next = deepcopy(alpha_current);
    ids=S.getOptTs_indices();
    alpha_next[ids] = alpha_current[ids] + 10*grad_H[ids];

    'SOLVE incremented Objective:'
    fs_post, J_next = S.solveObjectiveOnly(tau_chars, tau_char_weights, mu_sigma, alpha_next);

    print 'J_init =',J
    print 'J_post_increment =',J_next;
    
    'Hitting Time Density:'    
    D = mu_sigma[1]**2/2;  
    lfs = squeeze(fs_post[0,:,:]);
    gs_flow = S.hittime_distn_via_flow(mu_sigma[1], lfs)
    gs_conservation = S.hittime_distn_via_conservation(lfs);       
    survivor_function = sum(lfs, axis=0)*S._dx
    lower_flow = D*(lfs[ 1,:] - lfs[0,:])/S._dx  + \
                ((S._xs[0] - mu_sigma[0])/tau_chars[0] - alpha_next)*lfs[0,:];

    print 'gs_flow= : %.4f'%(sum(gs_flow*S._dt));
    print 'gs_conservation= : %.4f'%(sum(gs_conservation*S._dt));
    print 'lowerinflow = %4f'  %(sum(lower_flow)*S._dt)
            
    'Hitting Time Density: post-increment'    
    lfs = squeeze(fs_post[-1,:,:]);
    gs_flow_post = S.hittime_distn_via_flow(mu_sigma[1], lfs)
    gs_conservation_post = S.hittime_distn_via_conservation(lfs);
                     
    'Visualize Hitting Time Distn:'
    figure();  subplot(411)
    plot(S._ts, survivor_function)
    title('Survivor Function')
    hlines([.0, 1.], S._ts[0], S._ts[-1]);
    subplot(412)
    plot(S._ts, lower_flow); title('Lower BCs');
    ylim( ymin=-1e-4, ymax=1e-4 )    
    subplot(413);
    plot(S._ts, gs_flow, 'r');
    plot(S._ts, gs_conservation, 'g');
    legend(['g-flow', 'g-conservation'])
    subplot(414);
    plot(S._ts, gs_flow_post, 'r');
    plot(S._ts, gs_conservation_post, 'g');
    legend(['g-flow', 'g-conservation'])
    
    
    'Visualize Control Evolution:'
    cfig = figure(figsize=(17,12));subplots_adjust(hspace = 0.6, left=0.15, right=0.975 )
    
    'Controls Subplot:'
    ax1 = subplot(211);hold(True);
    plot(S._ts, alpha_current, 'b--', label='pre-increment'); 
    plot(S._ts, alpha_next, 'r', label='post-incrment');
    legend( prop={'size':label_font_size} ); 
    ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    vlines(S.Tf_optimization, ylim()[0], ylim()[1], linestyles='dashed');    
    title('Control Increment', fontsize = xlabel_font_size)
    
    sub_sample = 100
    for t, ap, an in zip(S._ts[50::sub_sample],
                         alpha_current[50::sub_sample],
                         alpha_next[50::sub_sample]):
        if abs(an-ap)>2e-1:
            annotate('', xy=(t, ap), xycoords='data',
                     xytext=(t, an), textcoords='data',
                     arrowprops={'arrowstyle': '<-'})
        
    
    ticks = [alpha_bounds[0], 0 ,alpha_bounds[1] ];
    ax1.set_yticks(ticks) 
    ax1.set_yticklabels(('$%.1f$'%alpha_bounds[0], '$0$','$%.1f$'%alpha_bounds[1]),
                         fontsize = label_font_size) 
    for t in ax1.get_xticklabels():
        t.set_visible(False) 
        
    'Gradient Sub-Plot:'
    ax2 = subplot(212);hold(True);
    plot(S._ts, grad_H); 
    vlines(S.Tf_optimization, ylim()[0], ylim()[1], linestyles='dashed');
    ylabel(r'$\nabla_\alpha I$', fontsize = xlabel_font_size);
    xlabel(r'$t$', fontsize = xlabel_font_size)
    title('Objective Gradient', fontsize = xlabel_font_size)
 
    time_ticks = [0. , Tfopt  ,S._ts[-1] ];    
    xtick_labels = ['$%.1f$'%x for x in time_ticks];
    xtick_labels[1] = '$t_{opt}$';
    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels(xtick_labels); 
    
    ticks = [-.1, 0, .1];
    ax2.set_yticks(ticks) 
    ax2.set_yticklabels(['$%.1f$'%tick for tick in ticks],
                         fontsize = label_font_size) 
    
    
    'save fig:'
    lfig_name = os.path.join(FIGS_DIR, 'control_increment_example.pdf');
    print 'saving to ', lfig_name
    cfig.savefig(lfig_name)
    
    

'''Compare the Python vs C implementation of the solver routines:'''
def PyVsCDriver(tau_chars,
                 mu_sigma, 
                 Tf):    
    tau_char_weights = ones_like(tau_chars) / len(tau_chars);
    
    'Define Solver:'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars ,
                                           mu_sigma, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars,
                                       mu_sigma, xmin, factor=0.2)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars,
                                       mu_sigma,  dx, xmin, factor = 4.0)
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    print 'Solver params: xmin, dx, dt, Nx, Nt', S._xs[0], S._dx, S._dt, S._num_nodes(), S._num_steps(); 
    
    'Define Control:'
    ts = S._ts;
#    alphas = amin(c_[sin(2*pi*ts) + ts, 2*ones_like(ts)]);
#    alphas = 2*ones_like(ts);
    alphas = sin(2*pi*ts)*(ts<S.Tf_optimization) + 2*(ts>=S.Tf_optimization)
    figure(); plot(ts, alphas); title('Applied Control')
             
    
    '''Forward Solution:'''
    pystart = time.clock();
    pyfs = S._fsolve_Py( tau_chars, tau_char_weights,
                         mu_sigma, alphas, visualize=False)
    print 'pytime = %.8f' %(time.clock() - pystart)

        
    Cstart = time.clock();
    Cfs = S._fsolve( tau_chars, tau_char_weights,
                       mu_sigma, alphas, visualize=False);
    print 'Ctime = %.8f' %(time.clock() - Cstart)
#    pyfs = Cfs;
    
    'Compare f:'
    ids = r_[arange(0,S._num_steps(), 100),
             S._num_steps() - 2, S._num_steps()-1];
    Nids = len(ids);
 
    py2c_error = sum(abs(pyfs - Cfs))
    print "forward py2c_error", py2c_error 
    'compare G'  
    Gpy = sum(pyfs[0,:, :], axis=0)*S._dx;
    Gc =  sum(Cfs[0,:, :], axis=0)*S._dx;
    
    'visualize f comparison:'
    ffig = figure();
    for pdx, idx in enumerate(ids):
        subplot(ceil(Nids/2), 2, pdx+1)
        hold(True);
        Cf0s = Cfs[0,:, idx];
        pyf0s = pyfs[0,:, idx]
        plot(S._xs, Cf0s, 'rx-')
        plot(S._xs, pyf0s, 'go-')        
        title('Cps vs pyps (t=%.2f)'%S._ts[idx])
    ffig.canvas.manager.window.showMaximized()
            
    figure(); hold(True);
    plot(S._ts, Gpy, 'go-', label='py');
    plot(S._ts, Gc,'rx-', label='C');
    title('Survival Distn py vs. C')
    legend();

#    print('Bailing early '); return
    
    '''Backward Solution:'''
    Cps = S._psolve_C( tau_chars, tau_char_weights, mu_sigma, alphas, Cfs); #print Cps.shape
    pyps = S._psolve_Py(tau_chars, tau_char_weights, mu_sigma, alphas, Cfs);
    
    'Compare p'
    ids = r_[arange(0,S._num_backward_steps(), 20),
             S._num_backward_steps() - 2, S._num_backward_steps()-1];
    Nids = len(ids);
 
    pfig = figure();
    for pdx, idx in enumerate(ids):
        subplot(ceil(Nids/2), 2, pdx+1)
        hold(True);
        Cp0s = Cps[0,:, idx];
        pyp0s = pyps[0,:, idx]
        plot(S._xs, Cp0s, 'rx-', label='C')
        plot(S._xs, pyp0s, 'go-', label='Python')
        title('Cfs vs pyfs (t=%.2f)'%S._ts[idx])
    legend()
    pfig.canvas.manager.window.showMaximized()
   
    figure();    hold(True);
    lts = S.getOptTs();
    plot(lts, pyps[0, -1,:], 'go-', label='py')
    plot(lts, Cps[0,  -1,:], 'rx-', label='C')
    title('BCs at xthresh'); legend();  
                       
    'compare p:'
    py2c_error = sum(abs(pyps - Cps))
    print "backward py2c_error", py2c_error    
    
        
    
'What are you trying to accomplish here?'
def CVsCDriver():
    simPs   = SimulationParams(tau_char = 1.)
    Tf = 5.;    
    'Load OptSoln (functionality test):'
    fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = [simPs.mu, simPs.sigma, Tf, 3])
    S = deepcopy(fbkSoln._Solver) 
#    alphas = array([.0,.0])
    
    
    S.setTf(10.0);
    
    ts = S._ts;
#    alphas = 1.5 + sin(2*pi*ts);
    alphas = ones_like(2*pi*ts);
    mu_sigma = [simPs.mu, simPs.sigma];    
    tau_char = 1/.9;
    
    print 'Simulation params: mtcs', [simPs.mu, tau_char, simPs.sigma];
    print 'Solver params: xmin, Tf, dx, dt, Nx, Nt', S._xs[0], S.getTf(), S._dx, S._dt, S._num_nodes(), S._num_steps(); 
    
    gs_base = S.solve_hittime_distn_per_parameter(tau_char, mu_sigma,  alphas)
    
    for idx in xrange(10000):
        lS =     deepcopy(S)         
        lgs = lS.solve_hittime_distn_per_parameter(tau_char, mu_sigma,  alphas)
    
        base_error = sum(abs(gs_base - lgs))
#        l_norm_const =  sum(lgs[1:]*diff(ts))
        l_norm_const =  sum(lgs*lS._dt) 
        if  base_error > 1e-2 or isnan(base_error):
            print '%d: l1 error = %.4f, sum base = %.8f sum l = %.8f'%(idx,
                                                                   base_error,
                                                                   sum(gs_base[1:]*diff(ts)),
                                                                  l_norm_const)
                                                                    
            figure(); hold(True);
            plot(S._ts, gs_base, label='base');
            plot(S._ts, lgs, label='l');
            legend(); 
            title('idx_%d'%idx)
#        
#    tau_char_weights = ones_like(tau_chars) / len(tau_chars); 
#    
#    pystart = time.clock();
#    pyfs = S._fsolve_Py( tau_chars, tau_char_weights,
#                       mu_sigma, alphas, visualize=False)
#    print 'pytime = %.8f' %(time.clock() - pystart)
#        
#    Cstart = time.clock();
#    Cfs = S._fsolve_Py( tau_chars, tau_char_weights,
#                       mu_sigma, alphas, visualize=False);
#    print 'Ctime = %.8f' %(time.clock() - Cstart)
#
#    'compare G'    
##    print 'py  sum:', sum(pyfs[0,:, 1:-1:5], axis=0)*S._dx;
##    print 'C sum:'  , sum(Cfs[0,:, 1:-1:5], axis=0)*S._dx;
#         
#    'compare norms:'
#    C2c_error = sum(abs(pyfs - Cfs))
#    print "C2c_error", C2c_error
#    
#    G1 = sum(pyfs[0,:, :], axis=0)*S._dx;
#    G2 =  sum(Cfs[0,:, :], axis=0)*S._dx;
#    
#    figure(); hold(True);
#    plot(S._ts, G1, label='C1');
#    plot(S._ts, G2, label='C2');
#    legend(); 
     
def ForwardSolveBox( ):
    Tf = 16;

    mts_init = [0.5,log(.5), log(1.)];
                      
    alphaF = lambda t : 2*tanh( (t-Tf/2))
    mts_est = [0.13837874  ,0.29010642,  0.7];
    cols = ['g', 'r', 'b'];
        
    mts_init[1] = exp(mts_init[1]);
    mts_init[2] = exp(mts_init[2]);
    mts_true = [.0, 1., 1.];
        
    dist_fig = figure(figsize=(17,10))
    for pdx, (param_tag,mts) in enumerate(zip(['init', 'final', 'true'],
                                              [mts_init, mts_est, mts_true])):
#    for pdx, (param_tag,mts) in enumerate(zip(['final'], [mts_true])):
#        print param_tag, ':', mts
        
        
        tau = mts[1]; mu_sigma =  [mts[0], mts[2]];
        lSolver =  generateDefaultAdjointSolver(tau, mu_sigma,  Tf=Tf);
        lSolver.refine(0.01, 0.5);
        
        alphas_for_f = alphaF(lSolver._ts);
        
        'Compute hitting time density:'
        gs = lSolver.solve_hittime_distn_per_parameter(tau,
                                                       mu_sigma,
                                                       alphas_for_f)
        
        lfs = squeeze( lSolver._fsolve( [tau], [1.0],
                                        mu_sigma, alphas_for_f,
                                        visualize=True));
#        
        gs_conservation = lSolver.hittime_distn_via_conservation(lfs);
        
        gs_flow = lSolver.hittime_distn_via_flow(mu_sigma[1], lfs)

                    
        g_integral = sum(gs)*lSolver._dt;                         
        print 'flow-integral = %.3f, conservation-integral=%.3f'%(g_integral,
                                                                  sum(gs_conservation)*lSolver._dt);
        
        print 'flow- -ve integral = %.3f, conservation -ve integral=%.3f'%(sum(gs[gs<0])*lSolver._dt,
                                                                           sum(gs_conservation[gs_conservation<0])*lSolver._dt);
                                                                           
        figure(dist_fig.number);
        subplot(211); hold(True); 
        plot(lSolver._ts, gs/g_integral, cols[pdx]+'--', linewidth=2)
         
        subplot(212); hold(True); 
        plot(lSolver._ts, gs_flow, cols[pdx]+'--', linewidth=2, label=param_tag+ ' flow') 
        plot(lSolver._ts, gs_conservation, cols[pdx]+'+-', linewidth=2, label=param_tag + ' conserv')
        
        del lSolver
            
    legend()                                    
                                                   

if __name__ == '__main__':
    from pylab import *
#    rcParams.update({'legend.fontsize': label_font_size,                     
#                     'axes.linewidth' : 2.0,
#                     'axes.titlesize' : xlabel_font_size ,  # fontsize of the axes title
#                     'axes.labelsize'  : xlabel_font_size,  
#                     'xtick.labelsize'      : label_font_size,
#                     'ytick.labelsize'      : label_font_size});
    

    mu_sigma = [0.5, 1.5] 
    
    tau_chars = linspace(0.25, 4, 2);
    tau_chars = linspace(0.5, 2,  2);
    tau_char_weights = ones_like(tau_chars) / len(tau_chars)
    
    alpha_bounds = (-2., 2.);
      
    Tf =  15.0;
    Tfopt=10.0;
    
    '''This shows how to solve just the forward equation without the whole opt-control solve''' 
    ForwardSolveBox()

    
    '''move the main PDE-solves to C:'''
#    PyVsCDriver(tau_chars, mu_sigma, 3.0);
    
    'Explore Solver <something>???'
#    CVsCDriver()


    'Solve a Single Iteration of the Forward-Adjoint System'
#    tau_chars_list =  [ linspace(0.25, 4, 2), linspace(0.5, 2,  2)]
#    for tau_chars in tau_chars_list[0:1]:
#        tau_char_weights = ones_like(tau_chars) / len(tau_chars);
#        SingleSolveHarness(tau_chars, tau_char_weights, mu_sigma, Tf=Tf, Tfopt=Tfopt);  
#        print('-'*16)    
    
    show()
    