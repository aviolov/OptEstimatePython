# -*- coding:utf-8 -*-
"""
Created on Apr 23, 2012

@author: alex
"""
from __future__ import division

from numpy import *
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
from copy import deepcopy


RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/ForwardSolver/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/ForwardSolver'
import os
for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

ABCD_LABEL_SIZE = 32 
#import ext_fpc

#from hd5Driver import hd5Solution

class ForwardSolver():
    def __init__(self, params, 
                       dx, x_min,
                       dt, Tf):  
        self._mu = params[0],
        self._beta = params[1] 
        
        self._x_thresh = 1.
           
        #DISCRETIZATION:
        self.rediscretize(dx, x_min, 
                          dt, Tf)
    
    def rediscretize(self,dx, x_min, 
                          dt, Tf):
        self._dx = dx
        self._dt = dt

        self._xs = self._space_discretize(x_min)
        self._ts = self._time_discretize(Tf)
    
    def getTf(self):
        return self._ts[-1]
    def setTf(self, Tf):
        self._ts = self._time_discretize(Tf)
    def getXmin(self):
        return self._xs[0]
    def setXmin(self, X_min):
        self._xs = self._space_discretize(X_min)
    
    def _space_discretize(self, X_min):
#        xs = None
#        try:
        xs = arange(self._x_thresh, X_min - self._dx, -self._dx)[-1::-1];
#        except MemoryError:
#            print X_min, self._dx;
        return xs
    
    def _time_discretize(self, Tf):
        return arange(.0, Tf+self._dt, self._dt);

#TODO:
#    @classmethod
#    def calculate_xmin(cls, Tf, abg, theta):
#        #ASSERT max_speed is float and >0.
#        alpha, beta, gamma = abg[0], abg[1], abg[2]
#        xmin = alpha - abs(gamma)/sqrt(1.0 + theta**2) - 2.0*beta / sqrt(2.0);
#        return min([-.25, xmin])
#    
#    @classmethod
#    def calculate_dt(cls, dx, abg, x_min, factor=4.):
#        #ASSERT max_speed is float and >0.
#        alpha, beta, gamma = abg[0], abg[1], abg[2]
#        MAX_SPEED = abs(alpha) + max([abs(x_min), 1.0]) + abs(gamma)   
#        return dx / float(MAX_SPEED) * factor; 
#
#    @classmethod
#    def calculate_dx(cls, abg, xmin, factor = 1e-1):
#        max_speed = abg[0] + abs(abg[2]) - xmin;
#        return abg[1] / max_speed * factor;
        
    def _num_nodes(self):
        return len(self._xs)
    def _num_steps (self):
        return len(self._ts)
    
    def solve(self, alphas, thetas, visualize=False):
        dx, dt = self._dx, self._dt;
        xs, ts = self._xs, self._ts;
        
        mu, beta = self._mu, self._beta
        
        num_thetas = len(thetas)
        #Allocate memory for solution:
        Fs = zeros((num_thetas,
                     self._num_steps(),
                      self._num_nodes() ));
        
        #Impose ICs:
        initial_distribution = ones_like(xs) * (xs > .0);
        Fs[:, 0, :] = array(list(initial_distribution)*num_thetas).reshape((num_thetas,-1)); 
    
        if visualize:
            figure(100); hold(True)
            for theta_idx in xrange(num_thetas):
                plot(xs, Fs[theta_idx, 0, :]); 
                title('INITIAL CONDITIONS:');
                xlim((xs[0], xs[-1]) )
                ylim((-.1, 1.1))             
        
        #Solve it using C-N/C-D:
        D = beta * beta / 2.; #the diffusion coeff
        
        #AlloCATE mass mtx:    
        M = lil_matrix((self._num_nodes() - 1, self._num_nodes() - 1));
        e = ones(self._num_nodes() - 1);
        
        dx_sqrd = dx * dx;
        
        d_on = D * dt / dx_sqrd;
        centre_diag = e + d_on;
        centre_diag[-1] = 1.0;
        M.setdiag(centre_diag)

        for tk in xrange(1, self._num_steps()):
#            t_prev = ts[tk-1]
            t_next = ts[tk]
            
            alpha_prev = alphas[tk-1]
            alpha_next = alphas[tk]

            for theta_idx, theta in enumerate(thetas):
                #Rip the previous time solution:
                F_prev = Fs[theta_idx, tk - 1, :];
                if max(abs(F_prev)) < 1e-5:
                    continue
                 
                #Advection coefficient:
                U_prev = -(mu + alpha_prev  - theta*xs);
                U_next = -(mu + alpha_next - theta*xs);
        
                #Form the right hand side:
                L_prev = U_prev[1:] * r_[ ((F_prev[2:] - F_prev[:-2]) / 2. / dx,
                                               (F_prev[-1] - F_prev[-2]) / dx)] + \
                                D * r_[(diff(F_prev, 2),
                                           - F_prev[-1] + F_prev[-2])] / dx_sqrd; #the last term comes from the Neumann BC:
                RHS = F_prev[1:] + .5 * dt * L_prev;
                #impose the right BCs:
                RHS[-1] = 0.;
    
                #Reset the 'mass' matrix:
                flow = .5 * dt / dx / 2. * U_next[1:];
                
                d_off = -.5 * D * dt / dx_sqrd;
    #            d_on = D * dt / dx_sqrd;
                
                #With Scipy .11 we have the nice diags function:
                #TODO: local Peclet number should determine whether we central diff, or upwind it! 
                L_left = d_off + flow;
                M.setdiag(r_[(L_left[1:-1], -1.0)], -1);
                
                L_right = d_off - flow;
                M.setdiag(r_[(L_right[:-1])], 1);
                
                #Thomas Solve it:
                Mx = M.tocsr()
                F_next = spsolve(Mx, RHS);
                if visualize:
                    if rand() < 1./ (1+ log(self._num_steps())):
                        figure()
                        plot(xs[1:], F_next);
                        title('t=' + str(t_next) + ' \theta = %.2g'%theta);
                    
                #Store solution:
                Fs[theta_idx, tk, 1:] = F_next;  
            
            #Break out of loop?
            if amax(Fs[:, tk, :]) < 1e-5:
                break

        #Return:
        return Fs

    def c_solve(self, alphas, thetas):
        pass
#        #calls a C routine to solve the PDE (using the same algo as in solve): 
#        abgth = r_[abg, self._theta];
#        phis = array(self._phis);
#        ts = self._ts;
##        xs = self._xs; //I don't have a clue why, but this form of xs breaks the routine, while the one below does not!!!
#        xs = linspace(self._xs[0], self._xs[-1], len(self._xs))    
#        #TODO: enforce that all params passed donw are NUMPY ARRAYS
#        Fs = ext_fpc.solveFP(abgth,
#                             phis, ts, xs)
#        
#            
#        return Fs;
      
                
class ForwardSolution():
    _FILE_EXTENSION = '.fs'
    def __init__(self, params,
                 xs, ts, Fs,  
                 alphas,
                 thetas):
        self._mu = params[0]
        self._beta = params[1]

        self._ts  = ts;
        self._xs  = xs;
        
        self._Fs = Fs;
        self._alphas = alphas
        self._thetas = thetas
        
    @classmethod
    def _default_file_name(cls, params):
        mu = params[0];
        beta = params[1]
        Tf = params[2];
        return 'ForwardSoln_m=%.1f_b=%.1f_Tf=%.1f'%(mu, beta,Tf)
    
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = self._default_file_name([self._mu,
                                                 self._beta,
                                                 self._ts[-1]]);
        print 'saving path to ', file_name
        
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name=None, mu_beta_Tf=None):
        ''' not both args can be None!!!'''
        if None == file_name:
            file_name = cls._default_file_method(mu_beta_Tf);
        print 'loading ', file_name

        file_name = os.path.join(RESULTS_DIR,
                                 file_name + cls._FILE_EXTENSION) 
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)     
        load_file.close();   
        return soln

########################

def CvsPYsolver(abg = [1.5, .3, 1.0],
                   save_fig_name = ''):
    
    start =  time.clock()
#    Pys = S.solve(...)
    print 'Py time = ', time.clock() - start;
    
    start =  time.clock()
#    Cs = S.c_solve(...)
    print 'C time = ', time.clock() - start;   
    
    #TODO Compare Pys vs Cs 
#    print 'max error = ', max(abs(Cs-Pys))
    
def PickleVsHD5():
    pass
    


def basicCalculate(recalculate = False,
                   save_figs = False):
    mu_true = .0;
    theta_true  = .5;
    beta_true = 1.;
    
    alpha_crit = theta_true
    
    thetas = [.1, theta_true, 1, 2.]
    
    if recalculate:
        Tf = 16.;
        dx = .025;
        x_min  = -1.5;
        dt = .025;
        print 'Xmin = ', x_min, ' Tf=', Tf, 'dx = ', dx, ' dt = ', dt
        params = [mu_true, beta_true]
        print 'mu = ', params[0], ' beta = ', params[1]
        
        S = ForwardSolver(params, 
                          dx, x_min,
                          dt, Tf)
        
        'calculate for each alpha constant:'
        for alpha_const, soln_tag in zip([.0, alpha_crit],
                                         ['alpha_0', 'alpha_crit']):
            alphas = alpha_const * ones_like(S._ts)
            start =  time.clock()
            Fs = S.solve(alphas,
                          thetas)
            print 'solve time = ', time.clock() - start;
            
            soln_id = 'BasicTest_' + soln_tag
            (ForwardSolution(params, S._xs, S._ts,
                             Fs, alphas, thetas)).save(soln_id)
    
    #end if
    
    'visualize'
    for soln_tag in ['alpha_0', 'alpha_crit']:
        soln_id = 'BasicTest_%s'%soln_tag
        print soln_id
        FSoln = ForwardSolution.load(soln_id)
                        
        figure(figsize=(17,6)); hold(True)
        for theta_idx, theta in enumerate(thetas):
#            subplot(211)
#            plot(FSoln._Fs[theta_idx, :, -1], label='theta=%.2f'%theta)
#            title('survivor')
#            legend()
#            subplot(212);
            
            dt = FSoln._ts[1] - FSoln._ts[0]; 
            gs = -diff(FSoln._Fs[theta_idx, :, -1]) / dt
            plot(FSoln._ts[1:],
                 gs, label='theta=%.2f'%theta)
            title('density')
            legend()
        
        if save_figs:
            file_name = os.path.join(FIGS_DIR,
                                     'test_%s.pdf'%soln_tag)
            print 'saving to ', file_name
            savefig(file_name)
    
    figure(figsize=(17,18))
    for theta_idx, theta in enumerate(thetas):
        subplot(len(thetas), 1, theta_idx+1)
        for soln_col, soln_tag in zip(['r', 'b'],
                                      ['alpha_0', 'alpha_crit']):
            soln_id = 'BasicTest_%s'%soln_tag
            FSoln = ForwardSolution.load(soln_id)
            gs = -diff(FSoln._Fs[theta_idx, :, -1]) / dt
            plot(FSoln._ts[1:], gs, 
                    soln_col, label=soln_tag)
            legend()
 
 

def hd5Box(recalculate = False):
    test_iters = 50
    soln_id = 'BasicTest_alpha_crit'
    
    start =  time.clock()
    for j in xrange(test_iters):
        pickleSoln = ForwardSolution.load(soln_id)
    pickle_time = time.clock() - start
            
#    (hd5Solution([pickleSoln._mu, pickleSoln._beta],
#                  pickleSoln._xs[1]-pickleSoln._xs[0], pickleSoln._xs[0],
#                  pickleSoln._ts[1]-pickleSoln._ts[0], pickleSoln._ts[-1],
#                     pickleSoln._Fs,
#                      pickleSoln._alphas, 
#                        pickleSoln._thetas)).save(soln_id)
                        
    
    start =  time.clock()
    print soln_id
    for j in xrange(test_iters):
        hd5Soln = hd5Solution.load(soln_id)
   
    print 'hd5 time = ', time.clock() - start;
    print 'Pickle time = ', pickle_time
    
    
    print 'max error = ', amax (abs(hd5Soln._Fs - pickleSoln._Fs)), \
                          amax (abs(hd5Soln._alphas - pickleSoln._alphas)), \

                
if __name__ == '__main__':
    from pylab import *
    
#    basicCalculate(recalculate=True,
#                   save_figs=True)

    hd5Box()
    
    
    show()
    