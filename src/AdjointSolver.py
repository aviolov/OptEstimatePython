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
#from PathSimulator import ABCD_LABEL_SIZE
from matplotlib.font_manager import FontProperties
from scipy import interpolate 

import ext_fpc
from HitTime_MI_Beta_Estimator import SimulationParams

from TauParticleEnsemble import TauParticleEnsemble

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
    
    def __init__(self, dx, dt, Tf, x_min, x_thresh = 1.0):  
        #DISCRETIZATION:
        self.rediscretize(dx, dt, Tf, x_min, x_thresh)

    #Grid management routines:    
    def rediscretize(self, dx, dt, Tf, x_min, x_thresh = 1.0):
        self._xs, self._dx = self._space_discretize(dx, x_min, x_thresh)
        self._ts, self._dt = self._time_discretize(Tf,dt)
    
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
    
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts, self._dt = self._time_discretize(Tf, self._dt)
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
     
    
    def _getICs(self, xs, alpha0, sigma):
        #WARNING! TODO: HOw do you choose 'a' correctly! 
        a = 0.1;
        pre_ICs = exp(-xs**2 / a**2) / (a * sqrt(pi))
        ICs = pre_ICs / (sum(pre_ICs)*self._dx) 
        return ICs
    
    def _getAdjointBCs(self, tau_char_weights, fs, sigma):
            
        D = sigma*sigma/2.;    
        di_x_fs  = -squeeze(fs[:, -1,:] - fs[:, -2,:]) /\
                           self._dx;
#        di_x_fs = -diff(sum(fs, axis=1), axis=1);        

        di_x_fs_mean = dot(tau_char_weights, di_x_fs);
        
        
        dixfs_over_dixfs_mean_bayesian = di_x_fs / di_x_fs_mean;
        ''' deal with 0/0:'''
        zero_ids = di_x_fs_mean<1e-8;        
        dixfs_over_dixfs_mean_bayesian[:, zero_ids] = 1.;
        
        '''return'''
        bcs = log(dixfs_over_dixfs_mean_bayesian) + 1 - dixfs_over_dixfs_mean_bayesian
        
#        figure();
#        subplot(211);
#        plot(di_x_fs[0,:]);
#        hold(True);
#        plot(di_x_fs[1,:]); legend(['t.5', 't2'])
#        subplot(212);
#        plot(di_x_fs_mean)
#        title('Marginal fs')

        return bcs

    ###########################################
    def solve_hittime_distn_per_parameter(self, tau_char,
                                          mu_sigma,
                                          alphas,
                                          visualize=False):
        'Forward Distn'
        fs = squeeze( self._fsolve( [tau_char], [1.0],
                                    mu_sigma, alphas, visualize));
        
        'Diffusion Const'
        D = mu_sigma[1]**2 / 2.0
#        'Hitting Time Density'
        gs = D*(fs[-2,:] - fs[-1,:])/self._dx;

#        gs = r_[-diff(sum(fs, axis=0))*self._dx/self._dt,
#                D*(fs[-2,-1] - fs[-1,-1])/self._dx];
        
        if visualize:
            figure()
            plot(self._ts, gs); title('HItting Time distn'); xlabel('t'); ylabel('t');
            
        return gs;
    
    
    def solve(self, tau_chars, tau_char_weights, mu_sigma,
               alphas, alpha_max,
               visualize=False, save_fig=False):
         
        '''the forward/adjoint states:''' 
        fs = self._fsolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas, visualize, save_fig)
        ps = self._psolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas, fs, visualize, save_fig)

        
        '''the Mutial Information Objective:''' 
        J = self.calcObjective(tau_char_weights, fs)
        
        '''the Hamiltonian gradient:''' 
        grad_H = self.calcObjectiveGradient(tau_char_weights, fs, ps)
        
        return self._xs, self._ts, fs, ps, -J, -grad_H
    
    ###########################################
    def calcObjectiveGradient(self, tau_char_weights, fs, ps):         
        '''The Hamiltonian gradient:
        NOTE: THAT WE need to divide by dx in approximating the derivative and 
        multiply by dx in approximating the integral so we just drop both (dx/dx = 1)''' 
         
        dxps = diff(ps, axis=1);
        
        dxp_times_f = squeeze( sum(fs[:, 1:,:]*dxps, axis=1) )

        grad_H = dot(tau_char_weights, dxp_times_f);
            
        
        return grad_H;
        
    def calcObjective(self, tau_char_weights, fs):
        
    
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
    
    def _psolve(self, tau_chars, tau_char_weights, mu_sigma,
                 alphas, fs, visualize=False, save_fig=False):
        'Interface to the underlying P(adjoint)-solver'
        return self._psolve_C(tau_chars, tau_char_weights,
                               mu_sigma, alphas, fs,
                               visualize, save_fig)
#        return self._psolve_Py(tau_chars, tau_char_weights,
#                               mu_sigma, alphas, fs,
#                               visualize, save_fig)
    
    def _psolve_Py(self, tau_chars, tau_char_weights, mu_sigma,
                 alphas, fs, visualize=False, save_fig=False):
        'rip params:'
        mu, sigma = [x for x in mu_sigma]
        
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
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
                    self._num_steps() ));
                    
        #Impose TCs: automoatic they are 0
#        ps[:,-1] = self._getTCs(xs, alpha_max+mu, tauchar, sigma)
        
        'Impose BCs at upper end:' 
        ps[:, -1, :] = self._getAdjointBCs(tau_char_weights,
                                           fs, 
                                           sigma);
            
        'Reset the TCs'
        ps[:, :, -1] = .0;
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
            for tk in xrange(self._num_steps()-2,-1, -1):
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
                            D        *(p_forward[2:]- 2*p_forward[1:-1] + p_forward[:-2] ) / dx_sqrd;  
                
                #Impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
                RHS = r_[0.,
                         p_forward[1:-1] + .5 * dt * L_forward];
                
                #Reset the Mass Matrix:
                #Lower Diagonal
                u =  U_current / (2*dx);
                d_off = D / dx_sqrd;
                        
                L_left = -0.5*dt*(d_off - u[:-1]);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal
                L_right = -0.5*dt*(d_off + u);
                M.setdiag(r_[NaN,
                             L_right], 1);
                #Bottom BCs:
                M[0,0] = -1.; M[0,1] = 1.;
                
                #add the terms coming from the upper BC at the backward step to the end of the RHS
                p_upper_boundary = ps[pdx, -1, tk];
                RHS[-1] += 0.5*dt*(d_off * p_upper_boundary + \
                                   u[-1]*p_upper_boundary )
                
                #Convert mass matrix to CSR format:
                Mx = M.tocsr();            
                #and solve:
                p_current = spsolve(Mx, RHS);
                
                #Store solutions:
                ps[pdx, :-1, tk] = p_current;
                              
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

            if save_fig:
                file_name = os.path.join(FIGS_DIR, 'f_t=%.0f_b=%.0f.png'%(10*tauchar, 10*beta))
                print 'saving to ', file_name
                soln_fig.savefig(file_name)
                
        return ps
    
    def _fsolve(self,  tau_chars, tau_char_weights,
                    mu_sigma, alphas, 
                    visualize=False, save_fig=False):
        ''' level of indirection'''
        if visualize:
            return self._fsolve_Py(tau_chars, tau_char_weights,
                               mu_sigma, alphas, 
                               visualize, save_fig)
        else:
            return self._fsolve_C(tau_chars, tau_char_weights,
                               mu_sigma, alphas, 
                               visualize, save_fig)
        
    def _fsolve_Py(self,  tau_chars, tau_char_weights,
                            mu_sigma, alphas, 
                             visualize=False, save_fig=False):
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
            soln_fig = figure()

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
                
            if save_fig:
                file_name = os.path.join(FIGS_DIR, 'f_mu=%.0f_sigma=%.0f.png'%(10*mu, 10*sigma))
                print 'saving to ', file_name
                soln_fig.savefig(file_name)
                
                
        return fs
    
    
    '''interface to the adjoint C solver'''
    def _psolve_C(self, tau_chars, tau_char_weights,
                    mu_sigma, alphas, fs,
                    visualize=False, save_fig=False ):
        array_shape = ( len(tau_chars), 
                        self._num_nodes(),
                        self._num_steps() )
        ps = empty(array_shape);
        
        ''' get the BCs - main link with fs '''
        BCs = self._getAdjointBCs(tau_char_weights,
                                   fs, 
                                    mu_sigma[1] );
        
        for tdx, tau in enumerate(tau_chars):        
                
            mu_tau_sigma = array([mu_sigma[0], tau, mu_sigma[1]]);
            'main ext.lib call: for some bizarre reason you need to wrap arrays inside an array() call:'            
            lp = ext_fpc.solve_p(mu_tau_sigma,                                 
                                 array(alphas),
                                 array(self._ts),
                                 array(self._xs),
                                 array(squeeze(BCs[tdx,:])));
                                  
            ps[tdx,:,:] = lp;
            
        return ps
        
        
    '''Interface to the C routines for the forward FP soln'''
    def _fsolve_C(self,  tau_chars, tau_char_weights,
                    mu_sigma, alphas, 
                    visualize=False, save_fig=False):
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
    
    
def compareControlTerm(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):
    ' This is a relic from the OPtSpike PRoject - what does it do here???'
    
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
    xs, ts, fs, ps =  S.solve(tb, alphas, alpha_bounds[1], visualize=False)
    
    
    #the gradients
    dxfs = diff(fs, axis=0)/S._dx;
    dxps = diff(ps, axis=0)/S._dx;
    
    pdxf = sum(ps[1:,:]*dxfs, axis=0) 
    
    pf_minus_dxpf = (ps[-1,:]*fs[-1,:] - ps[0,:]*fs[0,:]) - sum(fs[1:,:]*dxps, axis=0)  
       
    figure(); hold(True)
    plot(ts, pdxf, 'b', label=r'$\int p \nabla_x f$')
    plot(ts, pf_minus_dxpf, 'r', label=r'$ pf|_{x-}^{x+} - \int f \nabla_x p$')
    legend(loc='upper left')
    
    figure(); hold(True)
    plot(xs[1:], dxfs[:, 1], 'b', label=r'$\nabla_x \, f$')
    plot(xs[1:], dxps[:, 1], 'g', label=r'$\nabla_x \, p$'); xlabel('x')
    legend(loc='upper left')
        

def calcGradH(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                           fig_name = None):

        
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
    xs, ts, fs, ps, J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
    
    STEP_SIZE = .05;
    
    e = ones_like(alphas); alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    alpha_next = alphas + minus_grad_H  * STEP_SIZE
    alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
            
    alpha_next = amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #VISUALIZE:
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.top'] = .9
    figure(); hold(True)
    
#    plot(ts, grad_H, 'b', label=r'$\nabla_\alpha \, H$')
    plot(ts, minus_grad_H, 'g', label=r'$-\nabla_\alpha \, H$', linewidth = 4); 
    plot(ts, alphas, 'r--', label=r'$\alpha_0(t)$', linewidth = 4);
    plot(ts, alpha_next, 'b--', label=r'$\alpha_1(t)$', linewidth = 4);
    ylabel(r'$\alpha$', fontsize=24);xlabel('$t$', fontsize=24);    
    legend(loc='upper left')
    title('First Control Iteration', fontsize=36)
    
    if None != fig_name:
        get_current_fig_manager().window.showMaximized()
        file_name = os.path.join(FIGS_DIR, fig_name + '.png')
        print 'saving to ', file_name
        savefig(file_name);
        
    print 'J_0 = ', J
      
    
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
    

############################################################
def SingleSolveHarness(tau_chars, tau_char_weights,
                       mu_sigma,  alpha_bounds=(-2,2), ts=None) :
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma,  num_std= 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin, factor = 2.0)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma,
                                      dx, xmin, factor = 4.0)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt
    print alpha_bounds
 
    Tf = 15
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    alpha_current = 2.0 * tanh( 2*(S._ts - Tf*0.5 ) );    
    alpha_current =  (4.0) * (S._ts /S._ts[-1]) - 2.0
#    alpha_current = zeros_like(S._ts)
    
    alpha_max = alpha_bounds[1];
    
    print 'fs/ps size = %d'%(len(tau_char_weights)*len(S._ts)*len(S._xs)*8)

    '''MAIN CALL:'''
    xs, ts, fs, ps, J_first, grad_H =\
            S.solve(tau_chars, tau_char_weights, mu_sigma, alpha_current, alpha_max,
                    visualize=True);
     
#    fs = squeeze( S._fsolve(tau_chars, tau_char_weights,
#                            mu_sigma, alpha_current, visualize=True));
         
    D = mu_sigma[1]**2/2;
    
    fs = squeeze(fs[-1,:,:]);
    
    'Hitting Time Density'
    gs_flow = D*(fs[-2,:])/(S._dx);
    gs_conservation = -diff(sum(fs, axis=0))*S._dx;

    survivor_function = sum(fs, axis=0) *S._dx
    lower_flow = D*(fs[ 1,:]- fs[ 0,:])/S._dx  + \
                (S._xs[0]/tau_chars[0] - alpha_current)*fs[0,:];

    print 'gs_flow= : %.4f'%(sum(gs_flow*S._dt));
    print 'gs_conservation= : %.4f'%(sum(gs_conservation));
    print 'lowerinflow = %4f'  %(sum(lower_flow)*S._dt)
            
    'Visualize:'
    figure();  subplot(211)
    plot(S._ts, survivor_function)
    title('Survivor Function')
    hlines([.0, 1.], S._ts[0], S._ts[-1]);
    subplot(212)
    plot(S._ts, lower_flow); title('Lower BCs');
    ylim( ymin=-1e-4, ymax=1e-4 )
    
    
    figure();
    plot(S._ts, gs_flow, 'r');
    plot(S._ts[:-1], gs_conservation/S._dt, 'g');
    legend(['g-flow', 'g-conservation'])
    
    
    figure()
    subplot(211)
    plot(S._ts, alpha_current); title('Current Control')
    subplot(212)
    plot(S._ts, grad_H);
    title('Gradient')
    
  

def PyVsCDriver(tau_chars,
                 mu_sigma, 
                 Tf):
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars ,
                                           mu_sigma, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars,
                                       mu_sigma, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars,
                                       mu_sigma,  dx, xmin, factor = 4.0)
  
    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    print 'Solver params: xmin, dx, dt, Nx, Nt', S._xs[0], S._dx, S._dt, S._num_nodes(), S._num_steps(); 
    
    ts = S._ts;
    alphas = sin(2*pi*ts);
    tau_char_weights = ones_like(tau_chars) / len(tau_chars); 
    
#    pystart = time.clock();
#    pyfs = S._fsolve_Py( tau_chars, tau_char_weights,
#                       mu_sigma, alphas, visualize=False)
#    print 'pytime = %.8f' %(time.clock() - pystart)
        
    Cstart = time.clock();
    Cfs = S._fsolve( tau_chars, tau_char_weights,
                       mu_sigma, alphas, visualize=False);
    print 'Ctime = %.8f' %(time.clock() - Cstart)

    'Compare f'
    ids = r_[arange(0,S._num_steps(), 10),
             S._num_steps() - 2, S._num_steps()-1];
    Nids = len(ids);
 
#    py2c_error = sum(abs(pyfs - Cfs))
#    print "py2c_error", py2c_error
 
#    'compare G'  
#    Gpy = sum(pyfs[0,:, :], axis=0)*S._dx;
#    Gc =  sum(Cfs[0,:, :], axis=0)*S._dx;
#    
#    'visualize f comparison:'
#    figure();
#    for pdx, idx in enumerate(ids):
#        subplot(ceil(Nids/2), 2, pdx+1)
#        hold(True);
#        Cf0s = Cfs[0,:, idx];
#        pyf0s = pyfs[0,:, idx]
#        plot(S._xs, Cf0s, 'rx-')
#        plot(S._xs, pyf0s, 'gx-')
#        title('Cps vs pyps (t=%.2f)'%S._ts[idx])
#    figure(); hold(True);
#    plot(S._ts, Gpy, 'gx-', label='py');
#    plot(S._ts, Gc,'rx-', label='C');
#    title('Survival Distn py vs. C')
#    legend();


    
    ''' backward solution '''
    pyps = S._psolve( tau_chars, tau_char_weights,
                       mu_sigma, alphas, Cfs)
    
    Cps = S._psolve_C( tau_chars, tau_char_weights,
                       mu_sigma, alphas, Cfs); 
    print Cps.shape
    
    figure();
    for pdx, idx in enumerate(ids):
        subplot(ceil(Nids/2), 2, pdx+1)
        hold(True);
        Cp0s = Cps[0,:, idx];
        pyp0s = pyps[0,:, idx]
        plot(S._xs, Cp0s, 'rx-')
        plot(S._xs, pyp0s, 'gx-')
        title('Cfs vs pyfs (t=%.2f)'%S._ts[idx])
    
    figure()
    plot(S._ts, Cps[0,-1,:], 'rx-')
    hold(True);
    plot(S._ts, pyps[0,-1,:], 'gx-')
    title('BCs')  
                       
    'compare p'
    py2c_error = sum(abs(pyps - Cps))
    print "py2c_error", py2c_error
    
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
     
def ForwardSolveBox(tau_chars,
                     mu_sigma, 
                     Tf):
#    fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = mu_sigma + [Tf] + [len(tau_chars)])


    lS = fbkSoln._Solver; 
    
    lS.setTf(2*Tf);
    
    alphas = fbkSoln._cs_iterates[-1];
    alpha_bounds= fbkSoln._alpha_bounds;
    
    alpha_extra = 0*alpha_bounds[1]*ones_like(lS._ts);
    
    alpha_extra[0:len(alphas)] = alphas;
    
    for tau_char in tau_chars:
        gs = lS.solve_hittime_distn_per_parameter(tau_char,
                                                   mu_sigma,
                                                   alpha_extra,
                                                   visualize=False);
        print sum(gs)*lS._dt;                                       
                                                   

if __name__ == '__main__':
    from pylab import *
    Tf =  20;
    alpha_bounds = (-2., 2.);
    
#    tau_chars = linspace(.25, 4, 2);
    tau_chars = linspace(0.5, 2, 2);
#    tau_chars = [2.5];
    tau_char_weights = ones_like(tau_chars) / len(tau_chars)
    
    mu_sigma = [-0., 1.] 
  
    'Solve a Single Iteration of the Forward-Adjoint System (test-box)'
    SingleSolveHarness( tau_chars, tau_char_weights, mu_sigma )  
    
    '''This shows how to solve just the forward equation without the whole opt-control solve'''
#    ForwardSolveBox(tau_chars,
#                    mu_sigma,
#                    Tf)

    
    '''move the main PDE-solves to C:'''
#    PyVsCDriver(tau_chars,
#                 mu_sigma, 5.0)
#    CVsCDriver( )
    
    show()
    