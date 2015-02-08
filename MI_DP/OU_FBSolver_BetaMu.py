# -*- coding:utf-8 -*-
"""
@author: alex
"""
from __future__ import division

from numpy import *
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from scipy import interpolate
from scipy.optimize.zeros import brentq

from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import xlim
from timeit import itertools

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/OUFBSolver_BetaMu/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/OUFBSolver_BetaMu'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 20
xlabel_font_size = 32

#import ext_fpc
class FBSolver():    
    def __init__(self, dt, Tf, num_nodes=100, xmin=-5, xmax=5.):  
        #DISCRETIZATION:
        self.rediscretize(dt, Tf, num_nodes, xmin, xmax)

    #Grid management routines:    
    def rediscretize(self, dt, Tf, num_nodes, xmin=-5, xmax=5.):
        self._xs, self._dx = self._space_discretize(num_nodes, xmin, xmax)
        self._ts, self._dt = self._time_discretize(dt,Tf)
    
    def _space_discretize(self, num_nodes, xmin=-5, xmax=5.):
        xs = linspace(xmin, xmax, num_nodes)
        dx = xs[2]-xs[1];
        return xs,dx
    
    def _time_discretize(self, dt, Tf):
        num_steps = ceil( Tf/ dt )
        ts = linspace(.0, Tf, num_steps)
        dt = ts[1]-ts[0];
        return ts, dt
    #Setter/getters:
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts, self._dt = self._time_discretize(self._dt, Tf)
    def getXbounds(self):
        return self._xs[0],self._xs[-1]
    
    @classmethod
    def calculate_dx(cls, alpha_bounds, params, xmin, factor = 1e-1, xthresh = 1.0):
        #PEclet number based calculation:
        mu, tc, beta = [x for x in params]
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc;
        return factor * (beta / max_speed);
    @classmethod
    def calculate_dt(cls, alpha_bounds, params, dx, xmin, factor=2., xthresh = 1.0):
        mu, tc, beta = params[0], params[1], params[2]        
        max_speed = abs(mu) + max(alpha_bounds) + max([xmin, xthresh]) / tc;
        return factor * (dx / max_speed) 
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
    
    #private helpers:
    def _getTCs(self):
        TCs = zeros_like(self._xs)
        return TCs
    
    def _getICs(self, xs, width = 1.0,
                x_0=.0):
        #WARNING! TODO: HOw do you choose 'width' the width correctly! 
        pre_ICs = exp(-(xs-x_0)**2 / width**2) / (width * sqrt(pi))

#        pre_ICs = exp(-(xs-1.)**2 / width**2) / (width * sqrt(pi)) + \
#                  exp(-(xs+1.)**2 / width**2) / (width * sqrt(pi))
        
        'normalize to a prob. density'
        ICs = pre_ICs / (sum(pre_ICs)*self._dx) 
        return ICs
    
    ###################################
    ### MAIN FUNCTION CALLS:
    ###################################
    def solve(self, thetas, pThetas, 
              params,
               alphas,
               ICs = None,
                visualize=False):
        '''joint solve method, which passes the real work ot _fsolve and _bsolve'''
        #TODO: Upwind the drift term?
        start = time.clock()
        fs = self._fsolve(thetas, 
                          params,
                           alphas,
                           ICs,
                            visualize)
        f_time = time.clock();
        
        ps = self._bsolve(thetas, pThetas, fs[:,:, 1:-1], 
                          params, 
                          alphas[:, 1:-1], 
                          visualize)
        p_time = time.clock();

                
        J = self._calcObjective(fs, pThetas)
        J_time = time.clock();
        
        grad_J = self._calcGradientObjective(fs, ps, pThetas,
                                             visualize) 
        grad_J_time = time.clock()
        
        if visualize:
            print 'f_time, p_time, J_time, grad_J_time = ', f_time-start, p_time-f_time, J_time- p_time, grad_J_time - J_time
        
        return self._xs, self._ts, fs, ps, J, grad_J
    
    ###########################################
    def generateAlphaField(self, alpha_of_x_func):
        return tile(alpha_of_x_func(self._xs), 
                    (len(self._ts), 1));
    def _calcObjective(self, fs, pThetas):
        fs_averaged = tensordot(pThetas, fs, axes=1);
        
        J = .0
        
        dt = (self._ts[1] - self._ts[0]);
        dx = (self._xs[1] - self._xs[0]);
        for ak, pA in enumerate(pThetas):
            fs_local = fs[ak,:,:]
            non_zero_indices = (fs_local > 1e-8)
            iJ = sum( log(fs_local[non_zero_indices] /  \
                          fs_averaged[non_zero_indices]) * fs_local[non_zero_indices]) * dt * dx *pA
            
            J += iJ; 
        return J; 
        
        
    def _calcGradientObjective(self, fs, ps, pThetas,
                               visualize=False):
        grad_J = empty((self._num_steps(),
                        self._num_nodes()));
#        dxps = diff(ps, axis= 2)/ (self._xs[2] - self._xs[1]);
        dxps = (ps[:,:, 2:] - ps[:,:, :-2]) / \
               (self._xs[2] - self._xs[1]);
        
        dxps_times_fs = dxps * fs[:, :, 1:-1];
        grad_J[:, 1:-1] = tensordot(pThetas,dxps_times_fs , axes=1)
        
        #set the boundaries:
        grad_J[:, [0,-1]] = grad_J[:, [1,-2]]
        
        if visualize:
            soln_fig = figure()
            mod_steps = 20;
            num_cols = 4;
            num_rows = ceil(double(self._num_steps())/num_cols / mod_steps)
            xs = self._xs;
            for tk in xrange(1,self._num_steps()):
                step_idx = tk;
                if 0 == mod(step_idx,mod_steps) or 1 == tk:
                    plt_idx = floor(step_idx / mod_steps) + (mod_steps != 1) 
                    ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                    ax.plot(xs, grad_J[tk,:]); 
                    if 1 == tk:
                        ax.hold(True)
                        ax.plot(xs, grad_J[tk-1,:], 'r', label='ICs')
                        ax.set_xlabel('$x$'); ax.set_ylabel(r'$\delta J$')
                        for t in ax.get_xticklabels():
                            t.set_visible(True)
            soln_fig.canvas.manager.window.showMaximized() 
            
            
        
        return grad_J;
        
        
        
    def _fsolve(self, thetas, 
                fixed_params, 
                alphas,
                ICs = None,
                visualize=False):
        #The parameters are in order: A, c, sigma
        sigma, = [x for x in fixed_params]
        
        num_prior_params = len(thetas);
        
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'thetas, sigma = ', thetas, sigma
            print 'Tf = %.2f' %self.getTf()
            print 'x-,x+ %f,%f, dx = %f, dt = %f' %(self.getXbounds()[0], 
                                                    self.getXbounds()[1],
                                                    dx,dt)
        
        #Allocate memory for solution:
        fs = zeros((num_prior_params,
                    self._num_steps(),
                    self._num_nodes()));
        #Impose ICs:
        if None == ICs:
            ICs = self._getICs(xs);
        fs[:, 0,:] = tile( ICs ,
                           (num_prior_params, 1))
        
#        if visualize:
#            figure()
#            subplot(311)
#            plot(xs, fs[0, 0,:]); 
##            title(r'$\alpha=%.2f, \tau=%.2f, \beta=%.2f$'%(alphas[0],tauchar, beta) + ':ICs', fontsize = 24);
#            xlabel('x'); ylabel('f')
#            subplot(312)
#            plot(xs, alphas);
#            title('Control Input', fontsize = 24) ;
#            xlabel('x'); ylabel(r'\alpha')
        
        #Solve it using C-N/C-D:
        D = sigma**2 / 2.; #the diffusion coeff
        dx_sqrd = dx * dx;
        
#            if visualize:
#                title('Drift Field', fontsize = 24) ;
#                xlabel('x'); ylabel(r'$U$')
#                subplot(313)     
#                plot(xs, Us);
            #Allocate mass mtx:
                
        active_nodes = self._num_nodes()
        M = lil_matrix((active_nodes, active_nodes));
            
        #Centre Diagonal:        
        e = ones(active_nodes);
        d_on = D * dt / dx_sqrd;
        
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        #Off Diagonals:
        d_off = D / dx_sqrd;
     
            #Convert mass matrix to CSR format:
        for tk in xrange(1, self._num_steps()):
            for theta_idx, theta in enumerate(thetas):
                #Rip the forward-in-time solution:
                f_prev = fs[theta_idx, tk-1,:];
                
                mu, tau_char = theta;
                #load the drift field:
                Us_internal_field = (mu - xs) / tau_char;
                Us_prev = Us_internal_field + alphas[tk-1,:]
                Us_next = Us_internal_field + alphas[tk,:] 
       
                #Form the RHS:
                L_prev = -(Us_prev[2:]*f_prev[2:] - Us_prev[:-2]*f_prev[:-2]) / (2.* dx) + \
                           D * diff(f_prev, 2) / dx_sqrd;
                #impose the x_min BCs: homogeneous Newmann: and assemble the RHS: 
                RHS = r_[0.,
                         f_prev[1:-1] + .5 * dt * L_prev,
                         0.];

#                TODO Upwind:
                #Lower Diagonal
                u =  Us_next / (2*dx);
                        
                L_left = -.5*dt*(d_off + u[:-2]);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal
                L_right = -.5*dt*(d_off - u[2:]);
                M.setdiag(r_[NaN,
                             L_right], 1);
                
                #Bottom BCs:
                M[0,0] = Us_next[0] + D / dx;
                M[0,1] = -D / dx;
                
                #Upper BCs:
                M[-1,-2] = D / dx;
                M[-1,-1] = Us_next[-1] - D / dx;
                
                #and solve:
                Mx = M.tocsr();  
                f_next = spsolve(Mx, RHS);
                
                #Store solutions:
                fs[theta_idx  , tk, :] = f_next;
             #End ak loop
        #end tk loop:
        
        #Visualize:         
        def visualize_me(fs, As): #this is a function b/c i'm too lazy to make a bookmark:
            soln_fig = figure()
            mod_steps = 20;
            num_cols = 4;
            num_rows = ceil(double(self._num_steps())/num_cols / mod_steps)
            
#            fs_stat = calculateStationary();
            for tk in xrange(1,self._num_steps()):
                step_idx = tk;
                
                if 0 == mod(step_idx,mod_steps) or 1 == tk:
                    for tc_idx, A in enumerate(As):
                        plt_idx = floor(step_idx / mod_steps) + (mod_steps != 1) 
                        ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                        ax.plot(xs, fs[tc_idx, tk,:], label='A=%f'%As[tc_idx]);
#                        ax.plot(xs, fs_stat[tc_idx,:], '--') 
                        if 1 == tk:
                            ax.hold(True)
                            ax.plot(xs, fs[tc_idx,tk-1,:], 'r', label='ICs')
            #                        ax.legend(loc='upper left')
            #                        ax.set_title('k = %d'%tk); 
            #                        if False : #(self._num_steps()-1 != tk):
            #                            ticks = ax.get_xticklabels()
            #                            for t in ticks:
            #                                t.set_visible(False)
            #                        else:
                        ax.set_xlabel('$x$'); ax.set_ylabel('$f$')
                        for t in ax.get_xticklabels():
                            t.set_visible(True)
            soln_fig.canvas.manager.window.showMaximized()
        
        if visualize:
            visualize_me(fs, thetas)
                
        return fs
    
    def _bsolve(self, thetas, pThetas, fs,
                 params, 
                 alphas,
                  visualize=False):
        sigma, = [x for x in params]
       
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        if visualize:
            print 'x-,+', xs[[0,-1]]
            print 'Tf = %.2f' %self.getTf()
            print 'pThetas', pThetas
            print 'tcs', thetas
        
        #Allocate memory for solution:
        ps = zeros((len(thetas),
                    self._num_steps(),
                    self._num_nodes()   ));
        
        #WARNING:!!!
        fs[fs<0]= 1e-8;
        fs_averaged = tensordot(pThetas, fs, axes=1);
        adjoint_source = ones_like(fs);
        for idx, pTh in enumerate(pThetas):
            adjoint_source[idx, :, :]
            fAs = fs[idx, :, :];
            adjoint_source[idx,:,:] += -pTh*fAs / fs_averaged + log( fAs / fs_averaged );
        
        #Impose TCs: 
        #already done they are 0
        
        #Solve it using C-N/C-D:
        D = sigma * sigma / 2.; #the diffusion coeff
        dx_sqrd = dx * dx;
        
        #Allocate mass mtx:    
        active_nodes = self._num_nodes()
        M = lil_matrix((active_nodes, active_nodes));
        
        #Centre Diagonal:        
        e = ones(active_nodes);
        d_on = D * dt / dx_sqrd;
        
        centre_diag = e + d_on;
        M.setdiag(centre_diag)
        
        for tk in xrange(self._num_steps()-2,-1, -1):
            for theta_idx, theta in enumerate(thetas):
            #Rip the forward-in-time solution:
                mu, tau_char = theta;
                
                p_forward = ps[theta_idx, tk+1,:];
                di_x_p_forward = (p_forward[2:] - p_forward[:-2]) / (2*dx)
                di2_x_p_forward = (p_forward[2:] - 2*p_forward[1:-1] + p_forward[:-2]) / (dx_sqrd)
                               
                #Calculate the velocity field
                Us_internal_field = ((mu - xs)/tau_char)[1:-1];
                Us_forward = Us_internal_field + alphas[tk-1,:]
                Us_current = Us_internal_field + alphas[tk,:] 
                
                source_term = adjoint_source[theta_idx, tk,:];
                             
                #Form the RHS:
                L_forward  =  D * di2_x_p_forward + \
                            Us_forward * di_x_p_forward + \
                            2*source_term;
    
                #impose the x_min BCs: homogeneous Neumann: and assemble the RHS: 
                RHS = r_[(.0,
                          p_forward[1:-1] + .5 * dt * L_forward,
                          .0)];
                          
                #Reset the Mass Matrix:
                #Lower Diagonal
                u =  Us_current/ (2*dx);
                d_off = D / dx_sqrd;
                        
                L_left = -.5*dt*(d_off - u);
                M.setdiag(L_left, -1);
                
                #Upper Diagonal
                L_right = -.5*dt*(d_off + u);
                M.setdiag(r_[NaN,
                             L_right], 1);
    
                #Bottom BCs \di_x w = 0:
                M[0,0] = -1;    M[0,1] = 1;
                #Upper BCs:
                M[-1,-2] = -1;  M[-1,-1] = 1;
            
                #Convert mass matrix to CSR format:
                Mx = M.tocsr();            
                #and solve:
                p_current = spsolve(Mx, RHS);
                
                #Store solutions:
                ps[theta_idx, tk,:] = p_current;
            #theta loop
        #t loop
                     
        #Return:
        def visualize_me(ps, tcs, adjoint_source):
            soln_fig = figure()
            rewards_fig = figure();

            mod_steps = 20#64;  
            num_cols = 4;
            num_rows = ceil(double(self._num_steps())/num_cols / mod_steps) + 1
            #time plots:
            for tk in xrange(1,self._num_steps()):
                step_idx = self._num_steps() - tk - 2;
                
                if 0 == mod(step_idx,mod_steps) or 0 == tk or self._num_steps() - 2 == tk:
                    plt_idx = 1 + floor(step_idx / mod_steps) + int(0 == tk)
                    #param plots:
                    for tc_idx, A in enumerate(tcs):
                        ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx)
                        ax.plot(xs, ps[tc_idx,tk,:]);
                        if self._num_steps() - 2 == tk:
                            ax.hold(True)
                            ax.plot(xs,ps[tc_idx, tk+1,:], 'k+', label='TCs')
                        ax.set_title('tk=%d'%tk)
                        ax.set_ylabel('$p(x)$')
                        
                        ax = rewards_fig.add_subplot(num_rows, num_cols, plt_idx)
                        ax.plot(xs[1:-1], adjoint_source [tc_idx, tk,:])
                        ax.set_title('tk=%d'%tk)
                        ax.set_ylabel(r'$r(x | \theta)$')
            
            for fig in [soln_fig, rewards_fig]:
                fig.canvas.manager.window.showMaximized()
       
        if visualize:
            visualize_me(ps, thetas, adjoint_source)    
        return ps
     
    
   
########################
def incrementAlpha(alpha_current, alpha_gradient,
                    step_size = 1., alpha_max = 6):
    alpha_next = alpha_current + step_size * alpha_gradient;
    e = ones_like(alpha_next);
    alpha_bounded_below = amax(array([-alpha_max*e,
                                      alpha_next]), axis=0);
    return amin(array([alpha_max*e,
                       alpha_bounded_below]), axis=0)
     
    
    
class FBSolution():
    FILE_EXTENSION = '.fbs'
    def __init__(self,
                 fixed_params, 
                 Thetas, pThetas,
                 xs, ts,
                 fs, ps, alphas,
                 grad_J, J):
        self._ts  = ts;
        self._xs  = xs;
        self._fs = fs;
        self._ps = ps;
        
        self._alphas = alphas;
        self._grad_J = grad_J;
        self._J = J;
        
        self._sigma = fixed_params[0]
        
        self._Thetas = Thetas;
        self._pThetas= pThetas;
        
                
    def save(self, file_name=None):
#       path_data = {'path' : self}
        if None == file_name:
            file_name = 'FB_dblwell_Soln_s=%.1f_Tf=%.1f'%(
                                                         self._sigma,
                                                         self._ts[-1]);
        print 'saving path to ', file_name
        file_name = os.path.join(RESULTS_DIR, file_name + self.FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name=None, sigma_Tf=None, energy_eps=.001):
        ''' not both args can be None!!!'''
        if None == file_name:
            mu,beta,Tf = [x for x in sigma_Tf]
            file_name = 'FBKSoln_m=%.1f_b=%.1f_Tf=%.1f_eps=%.3f'%(mu,
                                                         beta,
                                                         Tf,
                                                         energy_eps);

        file_name = os.path.join(RESULTS_DIR, file_name +  cls.FILE_EXTENSION) 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln

class FBIterates():
    FILE_EXTENSION = '.fbi'
    def __init__(self, FBSolnsList):
        self._iteratesList = FBSolnsList;
                
    def save(self, file_name):
#        path_data = {'path' : self}
        file_name = os.path.join(RESULTS_DIR, file_name + self.FILE_EXTENSION)
        print 'saving path to ', file_name
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name):
        file_name = os.path.join(RESULTS_DIR, file_name +  cls.FILE_EXTENSION) 
        print 'loading ', file_name
        import cPickle
        load_file = open(file_name, 'r')
        solns = cPickle.load(load_file)
        return solns


################################################## 
################################################## 
################################################## 

def OUIllustrator():
    #Illustrates the Potential Well, the potential gradient
    Tf= 10;
    
    amax = 1;
    tc = 20.;
    c = .3;
    
    m = .0;
    
    x_min,x_max = -10,10 
    
    xs = linspace(x_min,x_max,1001);
    
    Vs = (xs**2 - m*xs) / tc   
    Us = -(xs - m)
    alphas = -amax*sign(xs)
    Us_tilted = Us +alphas
    Vs_tilted = Vs + alphas*xs
    
    for idx, ys in enumerate([Vs, Us, Us_tilted, Vs_tilted]):
        subplot(4,1,1+idx)
        plot(xs, ys)
#        if idx == 0:
##            xlim((-2,2))
#            ylim((-1,8))
#        if idx == 1:
##            xlim((-2,2))
#            ylim((-12,12))
#            hlines(0, -2,2)
#        if idx == 2:
##            xlim((-2,2))
#            ylim((-12,12))
#            hlines(0, -2,2)
        
    file_name = os.path.join(FIGS_DIR, 'illustrate_potential.pdf')
    print 'saving to ', file_name
    savefig(file_name)

 


def compareBasicAlphas(resimulate=True,
                       Tf = 2.,
                       tau_chars = [.5,2],
                       mus = [-1,1]):
    sigma = 1.;
    params = array([sigma]);
    amax = 1.0
    alpha_bounds = [-amax, amax];  
    dt = .01; num_nodes = 200;
    
    lSolver = FBSolver(dt, Tf, num_nodes, xmin=-5., xmax=5.)
    
    thetas = list( itertools.product(mus, tau_chars))
     
    pThetas = ones(len (thetas)) / len(thetas);
    
    'ALPHAS::'
    alpha_bang_bang = lambda x: amax* sign(x)
    alpha_null = lambda x: .0*x
    alpha_antibang = lambda x: -sign(x)* amax
    alpha_H1 = lambda x: amax* sign(x)*(abs(x) > 1.)
    
    
#    alpha_tags = ['bang_bang', 'atan_bang_bang',  'null', 'atan_bang_bang']
#    alpha_funcs = [alpha_bang_bang, alpha_atan, alpha_null, alpha_antiatan]
    
    alpha_tags = ['bang_bang', 'anti_bang', 'alpha_null', 'alpha_H1']
    alpha_funcs = [alpha_bang_bang, alpha_antibang, alpha_null, alpha_H1]        
    alpha_fig = figure()
    for alpha_func, alpha_tag in zip(alpha_funcs,
                                      alpha_tags): 
        fb_file_name = 'alpha_'+ alpha_tag;
        if resimulate:
            alphas = lSolver.generateAlphaField(alpha_func);
            xs = lSolver._xs;
            width = 2.0; x_0 = .0;
            pre_ICs = exp(-(xs-x_0)**2 / width**2) / (width * sqrt(pi))
            ICs = pre_ICs / (sum(pre_ICs)*lSolver._dx) 
            xs, ts, fs, ps, J, grad_J = lSolver.solve(thetas, pThetas,
                                           params, alphas, 
                                           ICs,
                                           visualize=False)
            (FBSolution(params, thetas, pThetas, xs, ts,
                         fs, ps, alphas, grad_J, J)).save(fb_file_name)

        fbSoln = FBSolution.load(fb_file_name)
        xs = lSolver._xs;
        ax = alpha_fig.add_subplot(111)
        ax.plot(xs, alpha_func(xs), label=alpha_tag)
        ax.set_ylim((-amax-1,+amax+1))
        ax.legend()
        
        print alpha_tag + 'J=%.3g'%lSolver._calcObjective(fbSoln._fs, fbSoln._pThetas)
       

def compareBangBang_vs_GradOptimal_Alphas(Tf = 2.):
    sigma = 1.;
    params = array([sigma]);
    
    amax = 6 #6 #10
    alpha_bounds = [-amax, amax];  
    dt = .01; num_nodes = 100;
    
    
    lSolver = FBSolver(dt, Tf, num_nodes, xmin=-2., xmax=2.)
    
    As = linspace(2,5, 2)#4
    pThetas = ones_like(As) / len(As);
    
    #load bang-bang:
#    fb_file_name = 'alpha_bang_bang_T=%d'%int(Tf)
    fb_file_name = 'alpha_bang_bang'
    fbBangSoln = FBSolution.load(fb_file_name)
    print 'J_bang_bang=', fbBangSoln._J;
        
    #load opt: 
#    opt_file_name = 'alpha_iterates_example_Tf=%.1f'%Tf
    opt_file_name = 'alpha_iterates_uICs_Tf=%.1f'%Tf  
    FBSolnsList = (FBIterates.load(opt_file_name))._iteratesList
    FBOptSoln = FBSolnsList[-1]
    mid_k = int(ceil(len(FBSolnsList) / 2));
    print 'J_mid(%d) = '%mid_k, FBSolnsList[mid_k]._J
    print 'J_opt=', FBOptSoln._J;
    
         

def visualizeAlphaHarness():
    amax = 6.
    alpha_null = lambda x: 0*x;
    visualizeAlpha(resimulate=True, alpha_func=alpha_null,
                    alpha_tag='null', amax = amax);
        
    alpha_bang_bang = lambda x: -amax* sign(x)
    visualizeAlpha(resimulate=True, alpha_func=alpha_bang_bang,
                    alpha_tag='bang_bang', amax = amax);

def visualizeAlpha(resimulate, alpha_func, alpha_tag, Tf=5., amax = 6.):
    c = 0.3;
    sigma = 1.;
    params = array([c, sigma]);

    dt = .01; num_nodes = 200;
        
    As = linspace(2,5, 2)#4
    pThetas = ones_like(As) / len(As);
    
    lSolver = FBSolver(dt, Tf, num_nodes, xmin=-2., xmax=2.)
    fb_file_name = 'alpha_%s_T=%d'%(alpha_tag, int(Tf));
    if resimulate:
        alphas = lSolver.generateAlphaField(alpha_func);
        xs, ts, fs, ps, J, grad_J = lSolver.solve(As, pThetas,
                                                  params, alphas, 
                                                  visualize=True)
        #save:
        (FBSolution(params, As, pThetas,
                     xs, ts, fs, ps, alphas, grad_J, J)).save(fb_file_name)
   
    #load:
    fbSoln = FBSolution.load(fb_file_name)
    print 'J=%.3f'%lSolver._calcObjective(fbSoln._fs, fbSoln._pThetas)
    
    xs, ts = fbSoln._xs, fbSoln._ts;
    fs, ps, grad_J, As = fbSoln._fs, fbSoln._ps, fbSoln._grad_J, fbSoln._As;
    
    #Visualize Script:
#    tks = r_[1, range(99,len(fbSoln._ts), 200)];
    tks = [1, 25, 50, 100,399, 474, 498]
    print tks,    ts[tks]

    soln_fig = figure(figsize = (17,21));
    subplots_adjust(hspace = .2,wspace = .35,
                     left=.025, right=1.,
                     top = .95, bottom = .05)
    num_rows = len(tks);
    num_cols = 3;
    for row_idx, tk in enumerate(tks):
        plt_idx = row_idx*3
        for tc_idx, A in enumerate(As):
            #fs: 
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+1)
            ax.plot(xs, fs[tc_idx, tk,:], label='A=%.1f'%As[tc_idx]); 
            ax.set_ylabel('$f$', fontsize = xlabel_font_size)
            ax.legend(prop={'size':label_font_size})
            ax.set_ylim((.0, 2.0))
            #ps:
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+2)
            ax.plot(xs, ps[tc_idx, tk,:], label='A=%.1f'%As[tc_idx]); 
            ax.set_ylabel('$p$', fontsize = xlabel_font_size)
            ax.legend(prop={'size':label_font_size})
#            ax.set_ylim((-.05, 3.0))
            
        #grad_J    
        ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+3)
        ax.plot(xs, grad_J[tk,:]); 
        ax.set_ylabel(r'$\nabla_{\alpha} J$', fontsize = xlabel_font_size)
        
        #common to all drawing:
        for col_idx in xrange(1,4):
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+col_idx)
            ax.set_title('t=%.2f'%ts[tk], fontsize = xlabel_font_size)
            ax.vlines(0,ax.get_ylim()[0], ax.get_ylim()[1],linestyles='dashed')
            ax.hlines(0,ax.get_xlim()[0], ax.get_xlim()[1],linestyles='dashed')
            ax.set_xlim(xs[0], xs[-1])
            if tk == tks[-1]:
                ax.set_xlabel('$x$', fontsize = xlabel_font_size);
            else:
                for x in ax.get_xticklabels():
                    x.set_visible(False)
    
    soln_fig.canvas.manager.window.showMaximized()
    lfig_name = os.path.join(FIGS_DIR, 'FB_alpha_%s_solution.pdf'%alpha_tag)
    print 'saving to ', lfig_name
    savefig(lfig_name, dpi=300)     
     

 

def IncrementAlphaHarness_ICs(recalculate = False,
                              num_iterates = 20,
                              Tf   = 1.,
                              tau_chars = [.5,2],
                              mus = [-1,1],
                              step_size = 5.):
    
    sigma = 1.;
    params = array([sigma]);
    amax = 1
    alpha_bounds = [-amax, amax];  
    alpha_bang_bang = lambda x: amax* sign(x)
    alpha_forward = lambda x: arctan(5*x) / (pi/ 2.) * amax
    alpha_null = lambda x: .0*x
    alpha_H1 = lambda x: amax* sign(x)*(abs(x) > 1.)
    
    
    
    dt = .01; num_nodes = 200;
    
    lSolver = FBSolver(dt, Tf, num_nodes, xmin=-5., xmax=5.)
    
    thetas = list( itertools.product(mus, tau_chars))
 
    pThetas = ones(len(thetas)) / len(thetas);
    
    file_name = 'alpha_iterates_uICs_Tf=%.1f'%Tf 
    if recalculate:                
        
        FBSolnsList = []
        xs = lSolver._xs;
        width = 2.0; x_0 = .0;
        pre_ICs = exp(-(xs-x_0)**2 / width**2) / (width * sqrt(pi))
        ICs = pre_ICs / (sum(pre_ICs)*lSolver._dx) 
        
        #Calculate bang-bang reference:
        alphas = lSolver.generateAlphaField(alpha_bang_bang);
        xs, ts, fs, ps, J, grad_J = lSolver.solve(thetas, pThetas,
                                                  params, alphas, 
                                                  ICs,
                                                  visualize=False)
        (FBSolution(params, thetas, pThetas, xs, ts,
                     fs, ps, alphas, grad_J, J)).save('alpha_bang_bang')
        print 'J_bang_bang = ', J
        
        #Calculate null reference:
        alphas = lSolver.generateAlphaField(alpha_null);
        xs, ts, fs, ps, J, grad_J = lSolver.solve(thetas, pThetas,
                                                  params, alphas, 
                                                  ICs,
                                                  visualize=False)
        (FBSolution(params, thetas, pThetas, xs, ts,
                     fs, ps, alphas, grad_J, J)).save('alpha_null')

        #NOw reset to the Null Solution:
        print 'ICs for alphas = bang-null-bang'
        alphas = lSolver.generateAlphaField(alpha_H1);

        for ak in xrange(num_iterates):

            xs, ts, fs, ps, J, grad_J = lSolver.solve(thetas, pThetas,
                                                  params, alphas, 
                                                  ICs,
                                                  visualize=False)
            print ak, J
            fbSoln = FBSolution(params, thetas, pThetas,
                                 xs, ts, fs, ps,
                                  alphas, grad_J, J)
            FBSolnsList.append(fbSoln)
            
            #calculate next alpha field:
            alphas = incrementAlpha(alphas, grad_J,
                                    step_size=step_size,
                                    alpha_max=amax)
        
        FBIterates(FBSolnsList).save(file_name)
        #//Resimulate
    
    #Load: 
    FBSolnsList = (FBIterates.load(file_name))._iteratesList
    num_iterates = len(FBSolnsList);
    
    
    #Visualize controls:
#    tks = [1, 20, 40,60,80, 99]
#    tks = array([1, 3, 10, 40, 60, 90, 97, 99])*int(Tf)
    tks = array([ 5, 20, 50, 80, 95 ])*int(Tf)

#    aks = [5, 10, 14, 16, 18, len(FBSolnsList)-1];
#    aks = [16, num_iterates - 1];
    aks = [0, int((num_iterates - 1)/2), num_iterates - 1];



    soln_fig = figure(figsize = (17,18));
    subplots_adjust(hspace = .2,wspace = .35,
                     left=.05, right=1.,
                     top = .95, bottom = .05)
    num_rows = len(tks);
    num_cols = 3;
    for row_idx, tk in enumerate(tks):
        for ak in aks:
            FBSoln = FBSolnsList[ak]
            plt_idx = row_idx*num_cols;
            xs = FBSoln._xs;
            ts = FBSoln._ts;
            fs, ps, alphas = FBSoln._fs, FBSoln._ps,FBSoln._alphas;
            
            tc_idx = 0;
            
            #fs: 
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+1)
            ax.plot(xs, fs[tc_idx, tk,:], label='k=%d'%ak); 
            if tk ==  tks[1]:
                ax.set_ylabel(r'$f_{\mu=%.1g, \tau=%.1g}$'%(thetas[tc_idx][0],thetas[tc_idx][1]),
                                                    fontsize = xlabel_font_size)
#            ax.set_ylim((.0, 2.0))
            #ps: 
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+2)
            ax.plot(xs, ps[tc_idx, tk,:], label='k=%d'%ak); 
            if tk ==  tks[1]:
                ax.set_ylabel(r'$p_{\mu=%.1g, \tau=%.1g}$'%(thetas[tc_idx][0], thetas[tc_idx][1]),
                                                    fontsize = xlabel_font_size)
#            ax.set_ylim((.0, 2.0))
            
            
            #alphas:
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+3)
            ax.plot(xs, alphas[tk,:], label='k=%d'%ak); 
            if tk == tks[1]:
                ax.set_ylabel(r'$\alpha_k$', fontsize = xlabel_font_size)
            
        #common to all drawing:
        for col_idx in xrange(1,num_cols+1):
            ax = soln_fig.add_subplot(num_rows, num_cols, plt_idx+col_idx)
            ax.set_title('t=%.2f'%ts[tk], fontsize = xlabel_font_size)
#            ax.vlines(0,ax.get_ylim()[0], ax.get_ylim()[1],linestyles='dashed')
#            ax.hlines(0,ax.get_xlim()[0], ax.get_xlim()[1],linestyles='dashed')
            ax.set_xlim(xs[0], xs[-1])
            if tk == tks[0]:
                ax.legend(prop={'size':label_font_size})
            if tk == tks[-1]:
                ax.set_xlabel('$x$', fontsize = xlabel_font_size);
            else:
                for x in ax.get_xticklabels():
                    x.set_visible(False)
                    
    #J figure:
    Js = [fb._J for fb in FBSolnsList];
    fbBangSoln = FBSolution.load('alpha_bang_bang')
    fbNullSoln = FBSolution.load('alpha_null')
    J_null  = fbNullSoln._J;
    J_bangbang = fbBangSoln._J;
    
    J_fig = figure(figsize = (17,6));
    subplots_adjust(left=.025, right=1.,
                     top = .95, bottom = .05)
    ax = J_fig.add_subplot(111);
    J_fig.subplots_adjust(left=.025, right=1.,
                     top = .95, bottom = .05)
    ax = J_fig.add_subplot(111);
    ax.plot(Js, 'b', label='Gradient Ascent');
    ax.plot(J_bangbang*ones_like(Js), 'r', label='Bang-Bang') 
    ax.plot(J_null*ones_like(Js), 'g', label=r'Do nothing ($\alpha =0$)') 
    ax.set_ylim([0., ceil(J_bangbang) ]);
    ax.legend(loc = 'lower right', prop={'size':label_font_size})
#   
    
    ax.set_ylabel(r'$J_{k}$', fontsize = xlabel_font_size);
    ax.set_xlabel('k', fontsize = xlabel_font_size)
    ax.set_title('$J$ evolution', fontsize = xlabel_font_size);
        
    #Save figs:
    for fig, fig_name in zip([soln_fig, J_fig],
                             ['FB_alpha_iterates_uICs_%d'%len(tks),
                              'FB_J_iterates_uICs']):       
#        fig.canvas.manager.window.showMaximized()
        lfig_name = os.path.join(FIGS_DIR, fig_name + '_Tf=%d.pdf'%(ceil(Tf)))
        print 'saving to ', lfig_name
        save_ret_val = fig.savefig(lfig_name, dpi=300)


if __name__ == '__main__':
    from pylab import *
    
    IncrementAlphaHarness_ICs(recalculate=False,
                              num_iterates = 25,
                              Tf = 2.,                              
                              step_size = 2)
    
#    visualizeAlphaHarness()
#    compareBasicAlphas(resimulate=True);
    
    
    
#    IncrementAlphaHarness_ICs(recalculate=False,
#                              num_iterates=50,
#                              Tf = 8.) #Tf= 2,4,8

#    compareBangBang_vs_GradOptimal_Alphas(Tf = 2.)
    
    
    show()
    