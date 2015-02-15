# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
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


#def deterministicControlHarness(params = [.1, .75, 1.25],
#                                 Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.), visualize=False, fig_name = None):
#    from scipy.integrate import odeint
#    xth = 1.0;
#    
#    mu, tauchar = params[0], params[1];
#    print r'\tc=', tauchar
#    a_max = alpha_bounds[1]
#    print r'\amax=', a_max
#    print r'\T=', Tf
#    print r'\e=', energy_eps
#    
#    def alpha(t, p_0):
#        return amin( [p_0 * exp(t/tauchar) / (2*energy_eps), a_max]  )
##        return amin(c_[p_0 * exp(t/tauchar) / (2*energy_eps), amax*ones(len(t))], axis=1)
#    
#    def dx(x, t, p_0):
#        return mu + alpha(t,p_0) - x /tauchar
#        
#    ts = linspace(.0, Tf, 100)
#    def p0_root(p0):
#        xs = odeint(dx, .0, ts, args=(p0,))
#        return xs[-1,0] - xth
#
##    if (p0_root(.0)*p0_root(a_max* energy_eps * 2) > 0):
##        '''same sign - return amax'''
##        return ts, a_max * ones_like(ts)
#                
#    p0 = brentq(p0_root, -.01, a_max* energy_eps * 2)
#    
#    xs = odeint(dx, .0, ts, args = (p0,))
#    alphas = [alpha(t,p0) for t in ts]
#    
#    if visualize:
#        figure()
#        subplot(211)
#        plot(ts,xs, linewidth=4); xlim((.0, Tf)) 
#        title(r'State Evolution: $\tau_c=%.2f,\alpha_{max}=%.2f$'%(tauchar, a_max), fontsize=24); ylabel('$x(t)$', fontsize=24)
#        subplot(212)
#        plot(ts,alphas, linewidth=4); xlim((.0, Tf))
#        title('Control Evolution')
#        xlabel('t', fontsize=24); ylabel(r'$\alpha(t)$', fontsize=24)
#        get_current_fig_manager().window.showMaximized()
#        
#        if None != fig_name:
#            file_name = os.path.join(FIGS_DIR, fig_name+ '.png')
#            print 'saving to ', file_name
#            savefig(file_name)
#    
#    return ts, alphas


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
        alpha_min = alpha_bounds[0]
        xmin = tc_max*alpha_min - num_std*sigma*sqrt(tc_max/2.0);
        return min([XMIN_AT_LEAST, xmin])
    @classmethod
    def calculate_dx(cls, alpha_bounds, tau_chars, mu_sigma, xmin, factor = 1e-1, xthresh = 1.0):
        #PEclet number based calculation:
        mu,   sigma = [x for x in mu_sigma]
        tc_min = amin(tau_chars)
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc_min;     
        return factor * (sigma / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, tau_chars,
                      mu_sigma, dx, xmin, factor=2., xthresh = 1.0):
        ''' dt = factor (dx / max_speed)'''
        mu,   sigma = [x for x in mu_sigma]     
        tc_min = amin(tau_chars)
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc_min;
        return factor * (dx / max_speed) 
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
     
    
    def _getICs(self, xs, alpha0, sigma):
        #WARNING! TODO: HOw do you choose 'a' correctly! 
        a = .1;
        pre_ICs = exp(-xs**2 / a**2) / (a * sqrt(pi))
        ICs = pre_ICs / (sum(pre_ICs)*self._dx) 
        return ICs
    
    def _getAdjointBCs(self, tau_char_weights, fs, sigma):
            
        D = sigma*sigma/2.;    
        di_x_fs = -squeeze(fs[:, -1,:] - fs[:, -2,:]) /\
                           self._dx;
        
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
 
        'Hitting Time Density'
        gs = D*(fs[-2,:] - fs[-1,:])/self._dx;
        
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
        multiply by dx in approximating the integral so we just drop that (dx/dx = 1)''' 
         
        dxps = diff(ps, axis=1);
        
        dxp_times_f = squeeze( sum(fs[:, 1:,:]*dxps, axis=1) )

        grad_H = dot(tau_char_weights, dxp_times_f);
            
        
        return grad_H;
        
    def calcObjective(self, tau_char_weights, fs):
        
    
        xs, ts = self._xs, self._ts;
        dx, dt = self._dx, self._dt;
        
        di_x_fs = -squeeze(fs[:, -1,:] - fs[:, -2,:]) /\
                           (xs[-1] - xs[-2])  ;
        
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
        
        #Impose BCs at upper end: 
        ps[:, -1, :] = self._getAdjointBCs(tau_char_weights,
                                           fs, 
                                           sigma);
            
        
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
                p_forward = ps[pdx, :,tk+1];
    
                #Rip the control:
                alpha_forward = alphas[tk+1]
                alpha_current = alphas[tk]
                
                #Calculate the velocity field
                U_forward = (alpha_forward + (mu - xs[1:-1])/ tau_char)
                U_current = (alpha_current + (mu - xs[1:-1])/ tau_char)
                
                #Form the RHS:
                L_forward = U_forward*(p_forward[2:] - p_forward[:-2]) / (2.* dx) + \
                            D        * diff(p_forward, 2) / dx_sqrd;  
                
                #Impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
                RHS = r_[0.,
                         p_forward[1:-1] + .5 * dt * L_forward];
                
                #Reset the Mass Matrix:
                #Lower Diagonal
                u =  U_current / (2*dx);
                d_off = D / dx_sqrd;
                        
                L_left = -.5*dt*(d_off - u[1:-1]);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal
                L_right = -.5*dt*(d_off + u[1:]);
                M.setdiag(r_[NaN,
                             L_right], 1);
                #Bottom BCs:
                M[0,0] = -1.; M[0,1] = 1.;
                
                #add the terms coming from the upper BC at the backward step to the end of the RHS
                p_upper_boundary = ps[pdx, -1,tk];
                RHS[-1] += .5* dt*(D * p_upper_boundary / dx_sqrd + U_current[-1] *p_upper_boundary / (2*dx) )
                
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
        return self._fsolve_C(tau_chars, tau_char_weights,
                               mu_sigma, alphas, 
                               visualize, save_fig)
        
    def _fsolve_Py(self,  tau_chars, tau_char_weights,
                            mu_sigma, alphas, 
                             visualize=False, save_fig=False):
        ''' returns fs[tau_ids, x_ids, t_ids];
        NOTE: tau_char_weights is irrelevant, I don't know why it's in the arg-list, it's not used'''
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
            
        
        if visualize:
            figure()
            subplot(311)
#            plot(xs, fs[:,-1]); 
#            title(r'$\alpha=%.2f,   \sigma=%.2f$'%(alphas[0],tauchar, sigma) + ':ICs', fontsize = 24);
#            xlabel('x'); ylabel('f')
             
            subplot(312)
            plot(ts, fs[0, -1, :]);
            title('BCs at xth', fontsize = 24) ; xlabel('t'); ylabel('f')
            
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
                    mod_steps = 40;  num_cols = 4;
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
            for fig in [soln_fig]:
                fig.canvas.manager.window.showMaximized()

            if save_fig:
                file_name = os.path.join(FIGS_DIR, 'f_mu=%.0f_sigma=%.0f.png'%(10*mu, 10*sigma))
                print 'saving to ', file_name
                soln_fig.savefig(file_name)
                
                
        return fs
    def _psolve_C(self, tau_chars, tau_char_weights,
                    mu_sigma, alphas, fs,
                    visualize=False, save_fig=False ):
        '''interface to the adjoint C solver'''
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
                                 array(squeeze(BCs[tdx,:])),
                                 array(alphas),
                                 array(self._ts),
                                 array(self._xs));
                                  
            ps[tdx,:,:] = lp;
            
        return ps
        
        
    def _fsolve_C(self,  tau_chars, tau_char_weights,
                    mu_sigma, alphas, 
                    visualize=False, save_fig=False):
        '''Interface to the C routines for the forward FP soln'''
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
    
'''The main routine to calculate the Optimal MI Control, it just wraps 
    gdOptimalControl_Aggressive'''    
def calculateOptimalControl(tau_chars, tau_char_weights,
                            mu_sigma,   
                            Tf = 1.0,
                              energy_eps=.001,
                              alpha_bounds = (-2., 2.),
                              grad_norm_tol = 1e-5,
                              obj_diff_tol = 5e-3,
                              soln_diff_tol = 1e-3,
                              dt_factor =4.,
                              step_size_base = 10.,
                              initial_ts_cs = None,
                               visualize=False):
    #Interface for drivers:
    return gdOptimalControl_Aggressive(tau_chars, tau_char_weights,
                                       mu_sigma,
                                       Tf=Tf,   
                                       alpha_bounds=alpha_bounds,
                                       grad_norm_tol=grad_norm_tol,
                                       obj_diff_tol=obj_diff_tol,
                                       soln_diff_tol=soln_diff_tol,
                                          dt_factor=dt_factor,
                                          step_size_base = step_size_base,
                                          initial_ts_cs=initial_ts_cs,
                                          visualize=visualize)

def gdOptimalControl_Old(params, Tf,
                            energy_eps = .001, alpha_bounds = (-2., 2.),
                            J_tol = 1e-3, gradH_tol = 1e-2, K_max = 100,  
                            alpha_step = .05,
                            visualize=False,
                            initial_ts_cs = None,
                            dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;


    min_es = ones_like(ts); max_es = ones_like(ts)
    switch_point = Tf/(1.01)
    min_es[ts>switch_point] = .0; max_es[ts<switch_point] = .0 
    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
#    initial_control = zeros_like(ts)
#    initial_control = alpha_min*min_es + alpha_max *  max_es;
    initial_control = None;
    
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
#    deterministic_control = deterministicControlHarness(params,Tf, alpha_bounds)
#    initial_control = interp(ts, deterministic_ts, deterministic_control)
#    initial_control = deterministic_control*ones_like(ts)
    alphas = initial_control
    
    alpha_iterations = [alphas]
    J_iterations = []

    J_prev = Inf;
    
    def incrementAlpha(alpha_prev, direction):
        e = ones_like(alpha_prev); 
        alpha_next = alphas + direction * alpha_step
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    for k in xrange(K_max):
        #the f,p, J, gradJ solution:
        xs, ts, fs, ps, J, minus_grad_H =  S.solve(params, alphas, alpha_bounds[1], energy_eps, visualize=False)
        print k, J
        
        #Convergence check:
        if abs(J - J_prev) < J_tol:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev) , ' ==> breaking!'
            break
        else:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev)
            J_iterations.append(J);
            J_prev = J
        if amax(abs(minus_grad_H)) < gradH_tol:
            break
        
        alphas = incrementAlpha(alphas, minus_grad_H)
        alpha_iterations.append(alphas)
    
    
    if visualize:   
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9     
        
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
       
#        if fig_name != None:
#            for fig,tag in zip([J_fig, controls_fig],
#                               ['_objective.png', '_control.png']):
#                fig.canvas.manager.window.showMaximized()
#                file_name = os.path.join(FIGS_DIR, fig_name + tag)
#                print 'saving to ' , file_name
#                fig.savefig(file_name)
            
    return xs, ts, fs, ps, alpha_iterations, J_iterations
  


def ncg4OptimalControl_NocedalWright(params, Tf,
                                    energy_eps = .001, alpha_bounds = (-2., 2.),
                                    J_tol = 1e-3, grad_tol = 1e-3, soln_norm_tol = 1e-3, K_max = 100,
                                    step_tol = .1, step_u_tol = .1, K_singlestep_max = 10,  
                                    orthogonality_tol = .1,
                                    alpha_hat = .5,
                                    visualize=False,
                                    initial_ts_cs = None,
                                    dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
    
    #Initial eval:
    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
                                                       alpha_current,
                                                        alpha_bounds[1],
                                                         energy_eps)
    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
    delta_t = ts[1] - ts[0];
    e = ones_like(alpha_current)
    def incrementAlpha(a_k,
                       d_k):
        #Push alpha in direction up to constraints:
        alpha_next = alpha_current + a_k * d_k; 
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1);
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = [alpha_current]
    J_iterations = [J_current]
    k = 0; ##outer iteration counter
    active_nodes = (alpha_current>alpha_bounds[0]) &  (alpha_current<alpha_bounds[1])
    grad_norm = sqrt(dot(minus_grad_H[active_nodes],
                         minus_grad_H[active_nodes]));
                             
    while (k< K_max and grad_norm > grad_tol * len(active_nodes)):
        #the f,p, J, gradJ solution:
        
        
        minus_grad_H_prev = minus_grad_H;
        soln_norm = sqrt(dot(alpha_current,
                             alpha_current));
        descent_norm = sqrt(dot(descent_d,
                                descent_d));
        print 'k, J_k, ||g_k||, g_tol, ||d_k||, ||c_k||',\
              '=\n %d, %.4f, %.4f, %.4f, %.4f, %.4f'%(k,
                                                    J_current,
                                                    grad_norm,
                                                    grad_tol * len(active_nodes),
                                                    descent_norm,
                                                    soln_norm)
        #Single step search:
        k_ss = 0;
        step_size = 100.;         
        alpha_next, J_next = None, None
        wolfe_1_condition, wolfe_2_condition = False, False
        c1_wolfe = 1e-4; #c1, c2 from page 144 of N+W, wolfe conditions eq. 5.42a,b
        c2_wolfe = 0.1;
        while (k_ss < K_singlestep_max):
            #generate proposed control
            alpha_next = incrementAlpha(a_k=step_size,
                                        d_k=descent_d)
            print '\t|a_k+1 - a_k|= %.4f'%(sum(abs(alpha_next-alpha_current)))
            #evaluate proposed control
            xs, ts, fs, ps, J_next, minus_grad_H =  S.solve(params,
                                                        alpha_next,
                                                         alpha_bounds[1],
                                                          energy_eps)
#            #Sufficient decrease?
            print '\tstep search: k_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            
            active_nodes = (alpha_next>alpha_bounds[0]) &\
                           (alpha_next<alpha_bounds[1])
            print '\t num active nodes %d / %d'%(len(alpha_current[active_nodes]),
                                                 len(alpha_current));

            cos_descent_dir = dot(minus_grad_H_prev,
                                  descent_d)
            wolfe_1_condition = (J_next <= J_current + c1_wolfe*step_size*cos_descent_dir);
            wolfe_2_condition = (abs(dot(minus_grad_H, descent_d)) <= c2_wolfe*abs(cos_descent_dir));
            print '\t w1:%.3f ? %.3f'%(J_next,
                                       J_current + c1_wolfe*step_size*cos_descent_dir)
            print '\t w2:%.3f ? %.3f'%(abs(dot(minus_grad_H, descent_d)),
                                       c2_wolfe*abs(cos_descent_dir)     ) 
#            if (wolfe_1_condition and wolfe_2_condition):
            if (wolfe_1_condition):
                print 'sufficient decreases for for wolfe{1,2} breaking'
                break;
            #reduce step_size
            step_size *=.8
            k_ss+=1
            
        if K_singlestep_max == k_ss:        
            print 'Single Step Failed::Too many iterations'
        
        alpha_current = alpha_next
        J_current = J_next
        #store latest iteration;
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
        
        #calculate grad_norm:
        grad_norm_squared = dot(minus_grad_H,
                                minus_grad_H)
        grad_norm = sqrt(grad_norm_squared);
                             
        delta_g = minus_grad_H_prev - minus_grad_H#Note that it is in reverse order since the minuses are already included
        
        beta_proposal = dot(minus_grad_H,
                            delta_g) / dot(minus_grad_H_prev,
                                           minus_grad_H_prev);
        beta_PR = max([.0,
                       beta_proposal]);
        print 'beta_PR+=%.3f' %beta_PR
        
        #Restart???
        
        #recompute descent dir:
        descent_d = minus_grad_H + beta_PR*descent_d;
                
        
        k+=1;

    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
            
    return xs, ts, fs, ps,\
           alpha_iterations, J_iterations
  


def ncg4OptimalControl_BorziAnnunziato(params, Tf,
                        energy_eps = .001, alpha_bounds = (-2., 2.),
                        J_tol = 1e-3, grad_tol = 1e-3, soln_norm_tol = 1e-3, K_max = 100,
                        step_tol = .1, step_u_tol = .1, K_singlestep_max = 10,  
                        orthogonality_tol = .1,
                        alpha_hat = .5,
                        visualize=False,
                        initial_ts_cs = None,
                        dt_factor = 4.):
    print 'simple Gradient Descent'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, params, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, params, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, params,
                                       dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
        initial_control = (alpha_max-alpha_min)*ts / Tf + alpha_min
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
    
    #Initial eval:
    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
                                                       alpha_current,
                                                        alpha_bounds[1],
                                                         energy_eps)
    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
    delta_t = ts[1] - ts[0];
        
    def incrementAlpha():
    #Push alpha in direction up to constraints:
        alpha_next = alpha_current + step_size * descent_d; 
        alpha_bounded_below = amax(c_[alpha_min*e,
                                      alpha_next], axis=1);
        return amin(c_[alpha_max*e,
                       alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = [alpha_current]
    J_iterations = [J_current]
    k = 0; ##outer iteration counter
    grad_norm = sqrt(dot(minus_grad_H,
                             minus_grad_H));
    while (k < K_max and grad_norm > grad_tol * Tf / delta_t):
        #the f,p, J, gradJ solution:
        
        minus_grad_H_prev = minus_grad_H;
        soln_norm = sqrt(dot(alpha_current,
                             alpha_current));
        descent_norm = sqrt(dot(descent_d,
                                descent_d));
        print 'k, J_k, ||g_k||, ||d_k||, ||c_k|| =%d, %.4f, %.4f, %.4f, %.4f'%(k,
                                                                    J_current,
                                                                    grad_norm,
                                                                    descent_norm,
                                                                    soln_norm)
        #Single step search:
        k_ss = 0;
        step_size_hat = .0;
        delta_decrease = .01; #sufficient decrease constant: put in top-function arg list:
        step_size_base = 1.;
        if 0 == k:
            step_size = step_size_base
        else:
            step_size = min([step_size_base,
                             3*sqrt(dot(alpha_current, alpha_iterations[-2]))/descent_norm]); #initial step size 
        delta_step_size = step_size;
        step_ratio_tol = .1; #used in 'delta_step_size > step_size_hat * step_ratio_tol' convergence criterion
        e = ones_like(alpha_current); 
        print 'step_size_init = %.3f'%step_size
        alpha_next, J_next = None, None
        while (delta_step_size > step_size_hat * step_ratio_tol and
               step_size*descent_norm > soln_norm * soln_norm_tol and
               k_ss < K_singlestep_max):
            
            #generate proposed control
            alpha_next = incrementAlpha()
            #evaluate proposed control
            xs, ts, fs, ps, J_next, minus_grad_H =  S.solve(params,
                                                        alpha_next,
                                                         alpha_bounds[1],
                                                          energy_eps,
                                                           visualize=False)
            #Sufficient decrease?
            print '\tstep search: k_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            print '\t J_next= %.3f, J_required = %.3f'%(J_next,
                                                        J_current - delta_decrease* step_size*dot(descent_d,
                                                                                                  minus_grad_H))
            if (J_next < J_current - delta_decrease* step_size*dot(descent_d,
                                                                   minus_grad_H)):
                step_size_hat = step_size;            
            delta_step_size /= 2.;
            step_size = step_size_hat + delta_step_size
            k_ss+=1
            print 'Loop Conditions: %d, %d, %d'%(delta_step_size > step_size_hat * step_tol,
                                                 step_size*descent_norm > soln_norm * soln_norm_tol,
                                                 k_ss < K_singlestep_max)
        if K_singlestep_max == k_ss:        
            print 'Single Step Failed::Too many iterations'
        if (1e-8 > step_size_hat):
            print 'Single Step Failed::step_size_hat ~ .0'
        alpha_current = alpha_next
        J_current = J_next
        #store latest iteration;
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
        
        #calculate grad_norm:
        grad_norm_squared = dot(minus_grad_H,
                                minus_grad_H)
        grad_norm = sqrt(grad_norm_squared);
                             
        delta_g = minus_grad_H_prev - minus_grad_H#Note that it is in reverse order since the minuses are already included
        
        beta_DY = None;
        if abs(dot(minus_grad_H, minus_grad_H_prev)) / grad_norm_squared > orthogonality_tol:
            beta_DY = .0;
        else:
            beta_DY = grad_norm_squared / dot(descent_d,
                                              delta_g);  
        print 'beta_DY=%.3f' %beta_DY
        
        #recompute descent dir:
        descent_d = minus_grad_H + beta_DY*descent_d;
                
        
        k+=1;

    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in [0,1,-2,-1]:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
            
    return xs, ts, fs, ps,\
           alpha_iterations, J_iterations
  


def gdOptimalControl_Aggressive(tau_chars, tau_char_weights,
                                mu_sigma,
                                Tf = 1.0,                                
                                alpha_bounds = (-2., 2.),
                                grad_norm_tol = 1e-5,
                                soln_diff_tol = 1e-3, #this should be related to alpha_min,max
                                obj_diff_tol =  5e-3, #we want three sig digids; 
                                K_max = 100,
                                K_singlestep_max = 10,
                                step_size_base = 10.,
                                step_size_reduce_factor = .5,
                                visualize=False,
                                initial_ts_cs = None,
                                dt_factor = 4.):
    
    print 'Aggresive Gradient Descent: TODO: Redefine active nodes to include those at the boundary but pointing inwards'
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma,
                                      dx, xmin, factor = dt_factor)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt
    print 'Tf = ',Tf
    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
    initial_control = None;
    if (None == initial_ts_cs):
#        initial_control = alpha_max/2.0*ones_like(ts)
        initial_control = 0*ones_like(ts)
    else:
        initial_control = interp(ts, initial_ts_cs[0], 
                                     initial_ts_cs[1])
    alpha_current = initial_control
    
#    
#    #Initial eval:
#    xs, ts, fs, ps, J_current, minus_grad_H = S.solve(params,
#                                                       alpha_current,
#                                                        alpha_bounds[1],
#                                                         energy_eps)
#    descent_d = minus_grad_H;
        
#    num_active_nodes= len(ts)-2
#    delta_t = ts[1] - ts[0];
    e = ones_like(alpha_current)
    
    def incrementAlpha(a_k,
                       d_k):
        #Push alpha in direction up to constraints:
        alpha_next = alpha_current + a_k * d_k; 
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1);
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    #The return lists:
    alpha_iterations = []
    J_iterations = []
    '''MAIN CALL:'''
    xs, ts, fs, ps, J_current, minus_grad_H =\
        S.solve(tau_chars, tau_char_weights, mu_sigma,  alpha_current, alpha_max );
    fs_init = fs;
    '''Begin Increment Procedure:'''
    step_size = step_size_base;
    for k in xrange(K_max):                         
#    while (k< K_max and grad_norm > grad_tol * len(active_nodes)):
        #the f,p, J, gradJ solution:
        #Calculate descent direction:
        alpha_iterations.append(alpha_current);
        J_iterations.append(J_current);
                                                         
        active_nodes = (alpha_current>alpha_bounds[0]) &  (alpha_current<alpha_bounds[1])
        print 'active_nodes = %d'%(len(alpha_current[active_nodes]))
        active_grad_norm = sqrt(dot(minus_grad_H[active_nodes],
                                    minus_grad_H[active_nodes]));
        effective_grad_tol = grad_norm_tol * len(alpha_current[active_nodes])
        if active_grad_norm <= effective_grad_tol:
            print 'active grad_norm = %.6f < %.6f, convergence!'%(active_grad_norm,
                                                                  effective_grad_tol);
            break
                         
        #Single line minimization: (step_size selection:
         
        print 'k=%d, J_k=%.4f, ||g_k||_active=%.4f, g_tol_effective=%.4f,'%(k,
                                                J_current,
                                                active_grad_norm,
                                                effective_grad_tol)
        #Single step search:
#        step_size /=step_size_reduce_factor;    #try to be a little more aggressive
        alpha_next, J_next = None, None
        single_step_failed = False;
        for k_ss in xrange(K_singlestep_max):
            #generate proposed control
            alpha_next = incrementAlpha(a_k=step_size,
                                        d_k=minus_grad_H);
#            print '\t|a_{k+1} - a_k|= %.4f'%(sum(abs(alpha_next-alpha_current)))
            'Inner Main Call:'
            xs, ts, fs, ps, J_next, minus_grad_H =\
                S.solve(tau_chars, tau_char_weights, mu_sigma,  alpha_next, alpha_bounds[1] ); 
                
#           #Sufficient decrease?
            print '\tk_ss=%d, step_size=%.4f, J=%.4f '%(k_ss, step_size, J_next)
            
             
            
#            sufficent_decrease = J_current - c1_wolfe*step_size * active_grad_norm*active_grad_norm
            sufficent_decrease = J_current;
            wolfe_1_condition = (J_next <= sufficent_decrease);
            
            if (wolfe_1_condition):
                print '\t sufficient decrease: %.6f < %.6f breaking' %(J_next,
                                                                       sufficent_decrease);
                step_size /= step_size_reduce_factor;
                step_size = min([10*step_size_base,
                                 step_size]); #make sure it's not too big!                                    
                break;
#            if step_size < step_size_tol:
#                print 'too many '
                single_step_failed = True;
            if K_singlestep_max-1 == k_ss:        
                single_step_failed = True;
            #reduce step_size
            step_size *=step_size_reduce_factor
            ###Single step loop
            
        if single_step_failed:
            break;
            
#        #calculate grad_norm:
#        delta_soln = alpha_next - alpha_current;
#        delta_J = J_next - J_current;
#        active_soln_diff_norm = sqrt(dot(delta_soln[active_nodes],
#                                        delta_soln[active_nodes]));
#        print 'active_soln_diff = %.6f, J_diff = %.6f'%(active_soln_diff_norm,
#                                                        abs(delta_J))
#        print 'active_soln_diff_tol = %.6f, J_diff_rel_tol = %.6f'%(soln_diff_tol*len(alpha_current[active_nodes]),
#                                                                                       obj_diff_tol)
#        if (active_soln_diff_norm <= soln_diff_tol*len(alpha_current[active_nodes])) and \
#            (abs(delta_J)/J_current <= obj_diff_tol):
#            print 'convergence!'
#            break
#        else:
        #Update current control, objective
        alpha_current = alpha_next        
        J_current = J_next
        ###Main Loop
    
    if visualize:   
        plot_every = int(floor(k/4));
        controls_fig = figure(); hold(True)
        iter_ids = [0,-1]
        for iter_idx in iter_ids:
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
       
        fs_fig = figure(); hold(True)
        for (tdx,_) in enumerate(tau_chars):
            lf = (fs[tdx,-2,:] - fs[tdx,-1,:])/S._dx;
            plot(ts, lf)
        legend(['t.5', 't2'])
        
    return  S, fs, ps, alpha_iterations, J_iterations
  


def exactStepOptimalControl(tb = [.6, 1.25], Tf = 1.5, energy_eps = .001, alpha_bounds = (-2., 2.),
                                    J_tol = .001, gradH_tol = .1, K_max = 100,  
                                    visualize=False, fig_name = None):
    print 'Congugate-Gradient Descent'
    
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tb, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tb, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tb, dx, xmin, factor = 4.)
    
    deterministic_ts, deterministic_control = deterministicControlHarness(tb, Tf, energy_eps, alpha_bounds)

    #Set up solver
    #TODO: The way you pass params and the whole object-oriented approach is silly. Tf changes for each solve and atb don't, so maybe rething the architecture!!!
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;
    alphas = interp(ts, deterministic_ts, deterministic_control)
    
    alpha_iterations = [alphas]
    J_iterations = []

    J_prev = Inf;
    
    def incrementAlpha(alpha_prev, direction, step):
        alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
        e = ones_like(alpha_prev); 
        alpha_next = alphas + direction  * step
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1)
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
    
    def exactLineSearch(alpha_prev, direction):
        alphas = None
        def line_objective(step):
            alphas = incrementAlpha(alpha_prev, direction, step)
            xs, ts, fs, ps,J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
#            print 'inner J = ', J
            return J
        best_step = fminbound(line_objective, .0, 1.0, xtol = 1e-2, maxfun = 16, disp=3)
        alphas = incrementAlpha(alpha_prev, direction, best_step)
        return alphas
        
         
    #THE MONEY LOOP:
    for k in xrange(K_max):
        #the f,p solution:
        xs, ts, fs, ps,J, minus_grad_H =  S.solve(tb, alphas, alpha_bounds[1], energy_eps, visualize=False)
        print k, J
        
        #Convergence check:
        if abs(J - J_prev) < J_tol:
            break
        else:
            if visualize:
                print 'J-J_prev = ',  abs(J - J_prev)
            J_iterations.append(J);
            J_prev = J
        if amax(abs(minus_grad_H)) < gradH_tol:
            break
        
        alphas = exactLineSearch(alphas, minus_grad_H)
        alpha_iterations.append(alphas)
    
    
    if visualize:   
        mpl.rcParams['figure.subplot.left'] = .1
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .1
        mpl.rcParams['figure.subplot.top'] = .9     
        
        plot_every = int(ceil(k/4));
        controls_fig = figure(); hold(True)
        for iter_idx in xrange(0,k, plot_every):
            plot(ts, alpha_iterations[iter_idx],  linewidth = 3, label=str(iter_idx))
        title('Control Convergence', fontsize = 36)
        ylabel(r'$\alpha(t)$',fontsize= 24); xlabel('$t$', fontsize= 24);    legend(loc='upper left')
        
        J_fig = figure();
        plot(J_iterations,  linewidth = 3, label='J_k'); 
        title('Objective Convergence', fontsize = 36)
        ylabel(r'$J_k$',fontsize= 24); xlabel('$k$', fontsize= 24);    legend(loc='upper right')
       
        if fig_name != None:
            for fig,tag in zip([J_fig, controls_fig],
                               ['_objective.png', '_control.png']):
                fig.canvas.manager.window.showMaximized()
                file_name = os.path.join(FIGS_DIR, fig_name + tag)
                print 'saving to ' , file_name
                fig.savefig(file_name)
            
            
    return alphas, S._ts, J_iterations[-1], k  

########################
class FBKSolution():
    def __init__(self,tau_chars, tau_char_weights,
                 mu_sigma, 
                 alpha_bounds, 
                 Solver,
                 fs, ps,  
                 cs_iterates, J_iterates):
         
        
        self._mu = mu_sigma[0]
        self._tau_chars = tau_chars;
        self._sigma = mu_sigma[1] 
        self._alpha_bounds = alpha_bounds;

        self._Solver = Solver
        
        self._fs = fs;
        self._ps = ps;
#        self._ts = self._Solver._ts;
        self._cs_iterates = cs_iterates
        self._J_iterates = J_iterates;
        
    @classmethod
    def _default_file_name(cls, mu, sigma, Tf, Ntaus):
        return 'FBK_MIHitTimeSoln_m=%.1f_s=%.1f_Tf=%.1f_Ntau=%d.fbk'%(mu, sigma,Tf, Ntaus)
    
    
    def getOptControls(self):
        return self._cs_iterates[-1]
                
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = self._default_file_name(self._mu, self._sigma,
                                                 self._Solver._ts[-1], len(self._tau_chars));
            
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name )
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name=None, mu_beta_Tf_Ntaus=None):
        ''' not both args can be None!!!'''
        if None == file_name:
            mu,sigma,Tf, Ntaus = [x for x in mu_beta_Tf_Ntaus  ]
            file_name = cls._default_file_name(mu, sigma, Tf, Ntaus);

        file_name = os.path.join(RESULTS_DIR, file_name ) 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln

############################################################3

def SingleSolveHarness(tau_chars, tau_char_weights,
                       mu_sigma,  alpha_bounds, Tf=2.0) :
        
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma,
                                      dx, xmin, factor = 4.0)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt

    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
    ts = S._ts;

    alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
   
#    alpha_current = ( alpha_max-alpha_min )*ts/Tf + alpha_min 
    alpha_current = ones_like(ts); 

    '''MAIN CALL:'''
    xs, ts, fs, ps, J_current, grad_H =\
            S.solve(tau_chars, tau_char_weights, mu_sigma,  alpha_current, alpha_max,
                     visualize=False );
    
    print 'J_current = ', J_current;
    
    
def FBKDriver(tau_chars, tau_char_weights,
              mu_sigma, Tf= 3.,
              alpha_bounds= (-1,1) ,                
              save_soln = True):
    print tau_chars, tau_char_weights, mu_sigma, Tf
     
    lSolver, fs, ps, cs_iterates, J_iterates = \
        calculateOptimalControl(tau_chars, tau_char_weights,
                                 mu_sigma,
                                  Tf=Tf ,  
                                 alpha_bounds=alpha_bounds, 
                                 visualize=True)
    
    
    if save_soln:
        (FBKSolution(tau_chars, tau_char_weights,
                     mu_sigma, alpha_bounds,
                     lSolver,
                     fs, ps,
                     cs_iterates, J_iterates)).save()
                   
                   

def usingMultiprocsExample(regimeParams, Tf, energy_eps = .001,
                 initial_ts_cs = None):
    from multiprocessing import Process
    procs = [];
    for params in regimeParams:
        print 'm,tc,b =' , params
        #Simulate:
#        procs.append( Process(target=FBKDriver,
#                                 args=(params,Tf,energy_eps),
#                                 kwargs = {'save_soln':True}))
#        procs[-1].start()

        FBKDriver(params, Tf,
                  energy_eps,
                  initial_ts_cs=initial_ts_cs,
                  save_soln=True)
        
#    for proc in procs:
#        proc.join()
 

def visualizeRegimes(tau_chars,
                     mu_sigma, 
                     Tf,
                     soln_name = None,
                     fig_name = None):
    'load soln:'
    if None == soln_name:
        fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = mu_sigma + [Tf] + [len(tau_chars)])
    else:
        fbkSoln = FBKSolution.load(file_name = soln_name);    

    'create figure:'
    solnfig = figure()
#    solnfig = figure(figsize = (17, 18))
#    subplots_adjust(hspace = .2, wspace = .25,
#                     left=.15, right=.975,
#                     top = .95, bottom = .05)
    
    '''Load Results''' 
    ts,xs, cs_init, cs_opt, Js, fs = fbkSoln._Solver._ts, fbkSoln._Solver._xs,\
                                 fbkSoln._cs_iterates[0], fbkSoln._cs_iterates[-1],\
                                 fbkSoln._J_iterates,\
                                 fbkSoln._fs;
                        
    cmin, cmax = [x for x in fbkSoln._alpha_bounds]
    
    '''Controls Figure'''
    axc = solnfig.add_subplot(3, 1, 1)
    axc.hold(True);
    axc.plot(ts, cs_init, 'b', linewidth=3)
    axc.plot(ts, cs_opt, 'r', linewidth = 3)
    axc.hlines(0, ts[0], ts[-1], linestyles='--')
    axc.legend(['Init Guess', 'Opt Soln (%d iterations)'%len(fbkSoln._cs_iterates) ])
    
    axc.set_xlim(ts[0], ts[-1]);
    axc.set_ylim(cmin ,cmax);
    ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
    axc.set_xticks(ticks)
    ticks = [cmin,   .0 ,cmax]
    axc.set_yticks(ticks)
    axc.set_xlabel('$t$', fontsize = xlabel_font_size);
    axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    axc.set_title('Optimal Control (argmax $I$)', fontsize=xlabel_font_size) 
    axc.set_yticklabels(('$%.1f$'%cmin, '$0$','$%.1f$'%cmax),
                         fontsize = label_font_size) 
    axc.set_xticks((.5, 1., 1.5, ))
    axc.set_xticklabels(('$.5$' ,'$1.0$' ,'$1.5$'), fontsize = label_font_size)
 
    '''Objective Progress'''
    axj = solnfig.add_subplot(3, 1, 2)
#    axj.hold(True);
    axj.plot(-array(Js), 'bx-', linewidth=3)
    axj.set_title('Gradient Ascent of the Objective $I$', fontsize=xlabel_font_size )
    axj.set_ylabel(r'$I_k$',fontsize= xlabel_font_size); 
    axj.set_xlabel('$k$', fontsize= xlabel_font_size);  
    axj.set_ylim(0, max(-array(Js))) 
    
    '''hitting time density'''
    axg = solnfig.add_subplot(3, 1, 3); hold(True)
    D = fbkSoln._sigma**2/2
    for (tdx,tau_char) in enumerate(tau_chars):
            lg =  D*(fs[tdx,-2,:] - fs[tdx,-1,:])/fbkSoln._Solver._dx;
            
            axg.plot(ts, lg, label = r'$\tau_c=%.2g$'%tau_char,
                     linewidth = 2)
            
            print 'tau_char =%f : int(g) = %f'%(tau_char, sum(lg)*fbkSoln._Solver._dt)
    axg.legend()
    axg.set_xlabel('$t$', fontsize = xlabel_font_size);
    axg.set_ylabel(r'$g (t| \tau_c)$', fontsize = xlabel_font_size);
    
    '''save?'''    
    get_current_fig_manager().window.showMaximized()        
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '.pdf')
        print 'saving to ', lfig_name
        solnfig.savefig(lfig_name, dpi=300)       
 

def PyVsCDriver(tau_chars,
                 mu_sigma, 
                 Tf):
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma, num_std = 1.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma,
                                      dx, xmin, factor = 4.0)
  
    #Set up solver
    S = FPAdjointSolver(dx, dt, Tf, xmin)
#    S = FPAdjointSolver(0.5, .99, 1, -0.5);
#    tau_chars = tau_chars[:1];
#    alphas = array([.0,.0])
    print 'Solver params: xmin, dx, dt, Nx, Nt', S._xs[0], S._dx, S._dt, S._num_nodes(), S._num_steps(); 
    
    ts = S._ts;
    alphas = sin(2*pi*ts);
    tau_char_weights = ones_like(tau_chars) / len(tau_chars); 
    
    pystart = time.clock();
    pyfs = S._fsolve_Py( tau_chars, tau_char_weights,
                       mu_sigma, alphas, visualize=False)
    print 'pytime = %.8f' %(time.clock() - pystart)
        
    Cstart = time.clock();
    Cfs = S._fsolve( tau_chars, tau_char_weights,
                       mu_sigma, alphas, visualize=False);
    print 'Ctime = %.8f' %(time.clock() - Cstart)

    'compare G'    
#    print 'py  sum:', sum(pyfs[0,:, 1:-1:5], axis=0)*S._dx;
#    print 'C sum:'  , sum(Cfs[0,:, 1:-1:5], axis=0)*S._dx;
         
    'compare norms:'
    py2c_error = sum(abs(pyfs - Cfs))
    print "py2c_error", py2c_error
    
    Gpy = sum(pyfs[0,:, :], axis=0)*S._dx;
    Gc =  sum(Cfs[0,:, :], axis=0)*S._dx;
    
    figure(); hold(True);
    plot(S._ts, Gpy, label='py');
    plot(S._ts, Gc, label='C');
    legend();
#    pygs = S.solve_hittime_distn_per_parameter(tau_chars[0], mu_sigma, alphas, visualize=False)
    
    ''' backward solution '''
    pyps = S._psolve( tau_chars, tau_char_weights,
                       mu_sigma, alphas, Cfs)
    Cps = S._psolve_C( tau_chars, tau_char_weights,
                       mu_sigma, alphas, visualize=False);
    
    
    

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
    fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = mu_sigma + [Tf] + [len(tau_chars)])


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
                                                   

def NtausBox(mu_sigma, Tf, 
             alpha_bounds = (-2,2), resimulate=False):
    tau_bounds = [.5, 2];
#    for Ntaus in arange(3, 6):
    for Ntaus in arange(3, 6):
        print Ntaus
        tau_chars = exp( linspace(log(tau_bounds[0]), log(tau_bounds[-1]), Ntaus))
        print tau_chars
        tau_char_weights = ones_like(tau_chars)/Ntaus
        print sum(tau_char_weights )
     
        if resimulate:
            lSolver, fs, ps, cs_iterates, J_iterates = \
                calculateOptimalControl(tau_chars, tau_char_weights,
                                        mu_sigma,
                                        Tf=Tf ,  
                                        alpha_bounds=alpha_bounds, 
                                        visualize=True)
    
    
            
            (FBKSolution(tau_chars, tau_char_weights,
                         mu_sigma, alpha_bounds,
                         lSolver,
                         fs, ps,
                         cs_iterates, J_iterates)).save()
                         
        visualizeRegimes(tau_chars, mu_sigma, Tf)
    
            
     
def PriorSpreadBox(mu_sigma, Tf, 
                   alpha_bounds = (-2,2), resimulate=False):
    
    '''Chech whether different priors, here the spread of the prior makes a difference on the optimal control
    CONCLUSION: It looks like it really does!!!
    '''
    
#    tau_chars_list = [ exp(randn(10)),  
#                      exp(randn(10)*.1)];
#    prior_tags = ['wide_prior', 'concentrated_prior'] 
    
    tau_chars_list = linspace(.65, 1.35, 2),
    prior_tags = [ 'concentrated_prior'] 

    for tdx, (tau_chars, prior_tag) in enumerate(zip(tau_chars_list, 
                                                     prior_tags)):
        
        print tdx,':', prior_tag, '\t', tau_chars
        tau_char_weights = ones_like(tau_chars)/len(tau_chars); 
        print sum(tau_char_weights )
     
        if resimulate:
            lSolver, fs, ps, cs_iterates, J_iterates = \
                calculateOptimalControl(tau_chars, tau_char_weights,
                                        mu_sigma,
                                        Tf=Tf ,  
                                        alpha_bounds=alpha_bounds, 
                                        visualize=True)
    
    
            
            (FBKSolution(tau_chars, tau_char_weights,
                         mu_sigma, alpha_bounds,
                         lSolver,
                         fs, ps,
                         cs_iterates, J_iterates)).save( prior_tag )
                         
        visualizeRegimes(tau_chars, mu_sigma, Tf, soln_name = prior_tag)
    
               
    
def PriorSpreadWidthBox(mu_sigma, Tf, 
                        alpha_bounds = (-2,2), 
                        Npts_in_prior = 2, delta_w = 0.1,
                        resimulate=False, fig_name = None):
    
    '''Chech whether different priors, here the spread of the prior makes a difference on the optimal control
    CONCLUSION: It looks like it really does!!!
    '''
    
    controlsDict = {};
    figure( figsize = (17, 6)); hold(True);
    
    widths = arange(0.1, 1., delta_w)
    widths = [.1,     .5   ,.9];
#    widths = [.8 , .9];
    for width in widths:          
        tau_chars = linspace(1-width, 1./(1-width), Npts_in_prior)
        tau_char_weights = ones_like(tau_chars)/len(tau_chars); 
        print width,':',  '\t', tau_chars, ' check sumprior = %.4f'%sum(tau_char_weights )
        
     
        fbk_save_name = 'width_experiment_%.2f'%(width) 
        if resimulate:
            lSolver, fs, ps, cs_iterates, J_iterates = \
                calculateOptimalControl(tau_chars, tau_char_weights,
                                        mu_sigma,
                                        Tf=Tf ,  
                                        alpha_bounds=alpha_bounds, 
                                        visualize=False)
    
    
            
            (FBKSolution(tau_chars, tau_char_weights,
                         mu_sigma, alpha_bounds,
                         lSolver,
                         fs, ps,
                         cs_iterates, J_iterates)).save(fbk_save_name )
                         
        fbkSoln = FBKSolution.load(file_name=fbk_save_name);
        
        controlsDict[width] = fbkSoln._cs_iterates[-1];
        
        'visualize:'
        plot(fbkSoln._Solver._ts, controlsDict[width],
             linewidth =2, label = 'width=%.1f'%width)
    
     
    'anotate plot:'
    legend();
    xlabel('$t$', fontsize = xlabel_font_size);
    ylabel(r'$\alpha(t)$', fontsize=xlabel_font_size)
    title('Opt Stimulus as a function of the prior spread')
    
    if None != fig_name:
        file_name = os.path.join(FIGS_DIR, fig_name+ '.pdf')
        print 'saving to ', file_name
        savefig(file_name)
    
         
               
def OptControlProfiler(mu_sigma = [0.,1.],
                       Tf = 2., 
                       alpha_bounds = (-2,2),
                       recompute = False):
    
    '''Profile the Opt Control pipeline  
    '''
    
    tau_chars_list = [ .5,  1. , 2.];
    tau_char_weights = ones_like(tau_chars)/len(tau_chars);
                
    
    'Profile Bottleneck:'
    if recompute:
        import cProfile
        cProfile.runctx('calculateOptimalControl(tau_chars, tau_char_weights, mu_sigma, Tf )',
                     globals(),
                     {'tau_chars':tau_chars, 'tau_char_weights':tau_char_weights, 
                      'mu_sigma':mu_sigma, 'Tf':Tf}, 'optcontrol.prof')
    
    import pstats
    pyprof = pstats.Stats('optcontrol.prof')
    pyprof.sort_stats('cumulative')
    pyprof.print_stats()
#    cprof  = pstats.Stats('cfsolve.prof')
#    cprof.sort_stats('time')
#    cprof.print_stats()    

    '''Conclusion: The bottleneck is by far in the adjoint calculation!
    fsolve(in C) takes .066 secs
    psolve (in python) takes 24.164 secs'''    

        
if __name__ == '__main__':
    from pylab import *
    Tf =  10;
    alpha_bounds = (-2., 2.);
    
    tau_chars = [0.5, 1.0,  2.];
    tau_char_weights = ones_like(tau_chars) / len(tau_chars)
    
    mu_sigma = [.0, 1.] 
  
#    SingleSolveHarness( tau_chars, tau_char_weights,
#                        mu_sigma, alpha_bounds, Tf)  
    
    
#    FBKDriver(tau_chars, tau_char_weights,
#              mu_sigma,
#              alpha_bounds=alpha_bounds,
#              Tf=Tf);              
#      
#    visualizeRegimes(tau_chars,
#                      mu_sigma,
#                       Tf,
#                        'ExampleOptControl_MI_HT')

    '''This shows how to solve just the forward equation without the whole opt-control solve'''
#    ForwardSolveBox(tau_chars,
#                    mu_sigma,
#                    Tf)

    ''' sweep over the number of taus'''
#    NtausBox(mu_sigma, Tf, alpha_bounds=alpha_bounds,
#             resimulate=False);

    ''' check if there is a difference between a wide spread prior vs. a concentrated prior:
      there is'''
#    Tf = 5;
#    PriorSpreadBox(mu_sigma, Tf, alpha_bounds, resimulate=True)
#
#    PriorSpreadWidthBox(mu_sigma, Tf, alpha_bounds,
#                        Npts_in_prior=4,
#                         resimulate=True) #, fig_name = 'Effect_of_prior_spread')
    
    '''move the forward computation to C:'''
#    PyVsCDriver(tau_chars, mu_sigma, Tf)
#    CVsCDriver( )
    
    
    '''move the backward computation to C:'''
#    CVsCDriver( )
#    PyVsCDriver(tau_chars, mu_sigma, Tf)


    '''Profile the bottlenecks:'''
#    OptControlProfiler();
    
    show()
    