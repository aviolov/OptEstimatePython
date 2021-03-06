'''
Created on Nov 4, 2013

@author: alex
'''
from __future__ import division

from numpy import *

from ForwardSolver import ForwardSolver, ForwardSolution
#from AdjointSolver import FPAdjointSolver as ForwardSolver
from ControlBasisBox import piecewiseConst
from matplotlib.pyplot import bar

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/ControlOptimizer/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/ControlOptimizer/'
import os
for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
import time

label_font_size = 24

class PiecewiseConstSolution():
    _FILE_EXTENSION = '.pcs'
    def __init__(self, ts, alpha_opt, thetas,  params, method_info):
        self.alpha_opt = alpha_opt
        self.ts = ts
        self._thetas = thetas;
        self._mu = params[0];
        self._sigma = params[1]
        self._method_info = method_info;
    
    @classmethod
    def _default_file_name(cls, self, params):
        mu = params[0];
        sigma = params[1]
        Tf = params[2];
        return 'PiecewsieConstSoln_m=%.1f_s=%.1f_Tf=%.1f'%(mu, sigma,Tf)
    
    def save(self, file_name=None):
#        path_data = {'path' : self}
        if None == file_name:
            file_name = self._default_file_name([self._mu,
                                                 self._sigma,
                                                 self._ts[-1]]);
        print 'saving path to ', file_name
        
        file_name = os.path.join(RESULTS_DIR,
                                 file_name + self._FILE_EXTENSION)
        import cPickle
        dump_file = open(file_name, 'wb')
        cPickle.dump(self, dump_file, 1) # 1: bin storage
        dump_file.close()
        
    @classmethod
    def load(cls, file_name=None, mu_sigma_Tf=None):
        ''' not both args can be None!!!'''
        if None == file_name:
            file_name = cls._default_file_method(mu_sigma_Tf);
        print 'loading ', file_name

        file_name = os.path.join(RESULTS_DIR,
                                 file_name + cls._FILE_EXTENSION) 
        import cPickle
        load_file = open(file_name, 'r')
        soln = cPickle.load(load_file)        
        return soln



def NonUniformPriorDriver(num_thetas=16,
                          num_intervals = 5,
                          recalculate=True,
                          save_figs=False):
    mu_true = .0;
    theta_true  = .5;
    sigma_true = 1.;
    
    Tf = 12.;

    energy_eps = .0001;
    
    #this should equal theta, but here equals average(thetas) to represent our ignorance
    alpha_max = 2.0
    alpha_crit_guess = 1.
    theta_min, theta_max = 0.1, 2.0
    thetas = array([ 1.19031198,  0.57208496,  0.07395076, .17824843,
                     0.27562893,  0.14913541,  0.24414001, .42702601])
    
    #Solver details:
    dx = .05;
    x_min  = -1.5;
    dt = .025;
    print 'Xmin = ', x_min, ' Tf=', Tf, 'dx = ', dx, ' dt = ', dt
    params = [mu_true, sigma_true]
    print 'mu = ', params[0], ' sigma = ', params[1]
    #INit Solver:
    S = ForwardSolver(params, 
                      dx, x_min,
                      dt, Tf)
    
    def f_objective(alpha_chars):
        
        alphas = piecewiseConst(alpha_chars, S._ts);
        Fs = S.solve(alphas,
                          thetas)
        dt = S._ts[1] - S._ts[0]; 
        gs = -diff(Fs[:, :, -1]) / dt
        
        
#        log_gs_normalized = zeros_like(gs);
#        not_zero_indices = where(gs > 1e-8); #WARNING: Magic number!
        
        mean_gs =  mean(gs, axis = 0)
        log_argument = gs / ( mean_gs + 1e-6 )#WARNING: Magic number!
        active_indices = (gs > 1e-8); #WARNING: Magic number!
        log_gs_normalized = zeros_like(gs)
        log_gs_normalized[active_indices] = log(log_argument[active_indices]) 
        KL_integrand = mean(gs * log_gs_normalized, axis=0);
        
        #sum it:
        KL_integral = dt * sum(KL_integrand)
        energy_integral = dt * energy_eps*dot(alphas,alphas)
        
        'log it:'
        print alpha_chars, ': %.4f &  %.4f &   %.4f'%(KL_integral,
                                                      energy_integral,
                                                      (KL_integral - energy_integral))
        return -(KL_integral - energy_integral)

    #MAIN OPTIMIZER CALL:
    if recalculate:
        from scipy.optimize import fmin
        alpha_init = theta_true * ones(num_intervals);
        alpha_opt = fmin(f_objective, alpha_init,
                          xtol=0.1, ftol=0.00001, maxiter = 50) 
        
        (PiecewiseConstSolution(S._ts, alpha_opt,
                               thetas, [mu_true, sigma_true],
                               method_info='Nelder-Mead - KL+energy')).save('NonUniformPrior')

    alpha_opt = PiecewiseConstSolution.load('NonUniformPrior').alpha_opt                
    
    figure(figsize=(17.,6)) 
#    subplot(211)       
    plot(S._ts, theta_true* ones_like(S._ts), 'b', label='crit')
    plot(S._ts, alpha_max* ones_like(S._ts), 'k', label='max')
    plot(S._ts, zeros_like(S._ts), 'g', label='nought')
    plot(S._ts, piecewiseConst(alpha_opt, S._ts), 'r', label='opt')
    ylim(-.5, 2.5)
    title('Different Controls');
    xlabel(r'$t$', fontsize = label_font_size)
    ylabel(r'$\alpha(t)$', fontsize = label_font_size)
    legend();
    
    if save_figs:
        file_name = os.path.join(FIGS_DIR,
                                 'basic_test_controls.pdf')
        print 'saving to ', file_name
        savefig(file_name)
    
    
    KL_nought = -f_objective(zeros(num_intervals));
    alpha_bang_bang = array(4*[alpha_max,-alpha_max])
    KL_bang_bang = -f_objective(alpha_bang_bang)
#    alphas_sin = sin(2*pi*S._ts[::5])
#    KL_sin = - f_objective(alphas_sin)
#    KL_crit   = -f_objective(theta_true * ones(num_intervals));
    KL_max    = -f_objective(alpha_max * ones(num_intervals))
    KL_opt    = -f_objective(alpha_opt);
#    print KL_crit, KL_max, KL_nought, KL_opt

def PiecewiseConstDriver_KL(num_thetas=4,
                         num_intervals = 5,
                         recalculate=True,
                         save_figs=False):
    
    mu_true = .0;
    theta_true  = .5;
    sigma_true = 1.;
    
    Tf = 12.;

    energy_eps = .00001;
    
    #this should equal theta, but here equals average(thetas) to represent our ignorance
    alpha_max = 2.0
    alpha_crit_guess = 1.
    theta_min, theta_max = 0.1, 2.0
    thetas = linspace(theta_min, theta_max, num_thetas)
    
    #Solver details:
    dx = .05;
    x_min  = -1.5;
    dt = .025;
    print 'Xmin = ', x_min, ' Tf=', Tf, 'dx = ', dx, ' dt = ', dt
    params = [mu_true, sigma_true]
    print 'mu = ', params[0], ' sigma = ', params[1]
    #INit Solver:
    S = ForwardSolver(params, 
                      dx, x_min,
                      dt, Tf)
    
    def f_objective(alpha_chars):
        
        alphas = piecewiseConst(alpha_chars, S._ts);
        Fs = S.solve(alphas,
                          thetas)
        dt = S._ts[1] - S._ts[0]; 
        gs = -diff(Fs[:, :, -1]) / dt
        
        
#        log_gs_normalized = zeros_like(gs);
#        not_zero_indices = where(gs > 1e-8); #WARNING: Magic number!
        
        mean_gs =  mean(gs, axis = 0)
        log_argument = gs / ( mean_gs + 1e-6 )#WARNING: Magic number!
        active_indices = (gs > 1e-8); #WARNING: Magic number!
        log_gs_normalized = zeros_like(gs)
        log_gs_normalized[active_indices] = log(log_argument[active_indices]) 
        KL_integrand = mean(gs * log_gs_normalized, axis=0);
        
        #sum it:
        KL_integral = dt * sum(KL_integrand)
        energy_integral = dt * energy_eps*dot(alphas,alphas)
        
        'log it:'
        print alpha_chars, ': %.4f &  %.4f &   %.4f'%(KL_integral,
                                                      energy_integral,
                                                      (KL_integral - energy_integral))
        return -(KL_integral - energy_integral)

    #MAIN OPTIMIZER CALL:
#   alpha_init = theta_true * ones(num_intervals);
    alpha_init = array([1.22457745,0.54751074,0.03406777,0.26787185,1.90000181]);
    if recalculate:
        from scipy.optimize import fmin
        alpha_opt = fmin(f_objective, alpha_init,
                          xtol=0.1, ftol=0.00001, maxiter = 50) 
        
        (PiecewiseConstSolution(S._ts, alpha_opt,
                               thetas, [mu_true, sigma_true],
                               method_info='Nelder-Mead - KL+energy')).save('BasicNMTest')

    alpha_opt = PiecewiseConstSolution.load('BasicNMTest').alpha_opt                
    
    figure(figsize=(17.,6)) 
#    subplot(211)       
    plot(S._ts, theta_true* ones_like(S._ts), 'b', label='crit')
    plot(S._ts, alpha_max* ones_like(S._ts), 'k', label='max')
    plot(S._ts, zeros_like(S._ts), 'g', label='nought')
    plot(S._ts, piecewiseConst(alpha_opt, S._ts), 'r', label='opt')
    plot(S._ts, piecewiseConst(alpha_init, S._ts), 'r', label='init')
    ylim(-.5, 2.5)
    title('Different Controls');
    xlabel(r'$t$', fontsize = label_font_size)
    ylabel(r'$\alpha(t)$', fontsize = label_font_size)
    legend();
    
    if save_figs:
        file_name = os.path.join(FIGS_DIR,
                                 'basic_test_controls.pdf')
        print 'saving to ', file_name
        savefig(file_name)
    
    
#    KL_nought = -f_objective(zeros(num_intervals));
#    alpha_bang_bang = array(4*[alpha_max,-alpha_max])
#    KL_bang_bang = -f_objective(alpha_bang_bang)
#    alphas_sin = sin(2*pi*S._ts[::5])
#    KL_sin = - f_objective(alphas_sin)
#    KL_crit   = -f_objective(theta_true * ones(num_intervals));
#    KL_max    = -f_objective(alpha_max * ones(num_intervals))
#    KL_opt    = -f_objective(alpha_opt);
#    print KL_crit, KL_max, KL_nought, KL_opt

#    alphas = piecewiseConst(alpha_opt, S._ts);
#    from numpy.polynomial.legendre import legval, legfit
#    leg_alphas = legval(S._ts, legfit(S._ts, alphas, deg = 10));
#    KL_leg    = -f_objective(leg_alphas);

#    from pygsl import chebyshev
#    s= chebyshev.cheb_series(10)
#    f= lambda x,params : interp(x, S._ts, alphas)
#    gslf = chebyshev.gsl_function(f, None) 
#    s.init(gslf, S._ts[0], S._ts[-1]);
#    cheb_alphas = empty_like(S._ts);
#    for k, t in enumerate(S._ts):
#        cheb_alphas[k] = s.eval(t)
#    KL_cheb    = -f_objective(cheb_alphas);
#    figure(); ts = S._ts;
#    plot(ts, alphas)
#    plot(ts, leg_alphas)
#    plot(ts, cheb_alphas)
        
        
def PiecewiseConstDriver_MI(num_thetas=2,
                            num_intervals = 6,
                            recalculate=True,
                            save_figs=False):
    mu_true    = .0;
    theta_true  = 1;
    sigma_true = 1.;
    
    Tf = 12.;
    
    #this should equal theta, but here equals average(thetas) to represent our ignorance
    alpha_max = 4.0
    alpha_crit_guess = 1./theta_true
#    theta_min, theta_max = 0.1, 2.0
    thetas =  [.5,2.];#linspace(theta_min, theta_max, num_thetas)
    
    #Solver details:
    dx = .05;
    x_min  = -1.5;
    dt = .025;
    print 'Xmin = ', x_min, ' Tf=', Tf, 'dx = ', dx, ' dt = ', dt
    params = [mu_true, sigma_true]
    print 'mu = ', params[0], ' sigma = ', params[1]
    #INit Solver:
    S = ForwardSolver(params, 
                      dx, x_min,
                      dt, Tf)
    
    def f_objective(alpha_chars):
        '''J = sum_params sum_time g(time|param) param_weight log( g / sum_params(g) )
        '''
        
        alphas = piecewiseConst(alpha_chars, S._ts);
        Fs = S.solve(alphas,
                     thetas)
        dt = S._ts[1] - S._ts[0]; 
        gs = -diff(Fs[:, :, -1]) / dt
        
        '''Calculate the marginal hitting time distribution'''
        mean_gs =  mean(gs, axis = 0) /num_thetas;
        
        '''Record Indices where unconditionally there is no probability of a hitting time'''
         
        
        log_term = zeros_like(gs);
         
        for p_k in xrange(num_thetas):
            lgs = gs[p_k,:];
            active_idxs = lgs>1e-8;
            log_term[p_k,active_idxs]  = log(lgs[active_idxs] /
                                              mean_gs[active_idxs])
            
        
        #sum it:
        MI_integral = dt * sum(log_term * gs) / num_thetas;
        
        'log it:'
        print alpha_chars, ': %.4f'%(MI_integral)
        return -(MI_integral)

    #MAIN OPTIMIZER CALL:
    #initialize:
    alpha_init = .0 * ones(num_intervals);
#    alpha_init = array([1.11812682, 0.48162286,
#                            0.41603237,  0.347,  0.5837312]);
    soln_tag = 'BasicNMTest_MI'
    if recalculate:
        from scipy.optimize import fmin
        #calculate soln:
        alpha_opt = fmin(f_objective, alpha_init,
                          xtol=0.1, ftol=0.000001, maxiter = 50) 
        
        (PiecewiseConstSolution(S._ts, alpha_opt,
                               thetas, [mu_true, sigma_true],
                               method_info='Nelder-Mead - MI')).save(soln_tag)

    alpha_opt = PiecewiseConstSolution.load(soln_tag).alpha_opt                
    
    '''The Control Plots:'''
    figure(figsize=(17.,10)) 
    subplot(311)       
    plot(S._ts, ones_like(S._ts) / theta_true, 'g', label='crit')
    plot(S._ts, alpha_max* ones_like(S._ts), 'c', label='max')
    plot(S._ts, zeros_like(S._ts), 'r', label='zero')
    plot(S._ts, piecewiseConst(alpha_opt, S._ts), 'b', label='opt')
    ylim(-.5, alpha_max+.5)
    title('Different Controls');
    xlabel(r'$t$', fontsize = label_font_size)
    ylabel(r'$\alpha(t)$', fontsize = label_font_size)
    legend();
        
    
#    figure()
    subplot(312)
    objectivevals = [];
    for alpha_knots, alpha_tag  in zip([alpha_opt, theta_true*ones_like(alpha_opt),
                                         .0*ones_like(alpha_opt), alpha_max*ones_like(alpha_opt)],
                                      ['opt', 'crit', 'zero', 'max']):
        alphas = piecewiseConst(alpha_knots, S._ts);
        Fs = S.solve(alphas,
                      array([theta_true])).squeeze();
        dt = S._ts[1] - S._ts[0]; 
        gs = -diff(Fs[:, -1]) / dt
        
        plot(S._ts[:-1], gs, label=alpha_tag)
        ylabel(r'Prob $t_sp = t$')
        xlabel('$t$')
        print alpha_tag 
        objectivevals.append(-f_objective(alpha_knots))
       
    legend()
    
    subplot(313) 
    bar(arange(4), array(objectivevals) );
    xticks(arange(4)+.5,  ['opt', 'crit', 'zero', 'max'])
        
    
    
    if save_figs:
        file_name = os.path.join(FIGS_DIR,
                                 'piecewise_const_controls.pdf')
        print 'saving to ', file_name
        savefig(file_name)
    

def exploreDifferentAlphas_MI():
    pass    
    
#    KL_nought = -f_objective(zeros(num_intervals));
#    alpha_bang_bang = array(4*[alpha_max,-alpha_max])
#    KL_bang_bang = -f_objective(alpha_bang_bang)
#    alphas_sin = sin(2*pi*S._ts[::5])
#    KL_sin = - f_objective(alphas_sin)
#    KL_crit   = -f_objective(theta_true * ones(num_intervals));
#    KL_max    = -f_objective(alpha_max * ones(num_intervals))
#    KL_opt    = -f_objective(alpha_opt);
#    print KL_crit, KL_max, KL_nought, KL_opt

#    alphas = piecewiseConst(alpha_opt, S._ts);
#    from numpy.polynomial.legendre import legval, legfit
#    leg_alphas = legval(S._ts, legfit(S._ts, alphas, deg = 10));
#    KL_leg    = -f_objective(leg_alphas);

#    from pygsl import chebyshev
#    s= chebyshev.cheb_series(10)
#    f= lambda x,params : interp(x, S._ts, alphas)
#    gslf = chebyshev.gsl_function(f, None) 
#    s.init(gslf, S._ts[0], S._ts[-1]);
#    cheb_alphas = empty_like(S._ts);
#    for k, t in enumerate(S._ts):
#        cheb_alphas[k] = s.eval(t)
#    KL_cheb    = -f_objective(cheb_alphas);
#    figure(); ts = S._ts;
#    plot(ts, alphas)
#    plot(ts, leg_alphas)
#    plot(ts, cheb_alphas)
        
    
    

if __name__ == '__main__':
    from pylab import *
    
#    PiecewiseConstDriver_KL(recalculate=True,
#                         save_figs = False)
    PiecewiseConstDriver_MI(recalculate=False,
                             save_figs = False)

#    NonUniformPriorDriver(recalculate=True,
#                          save_figs = False)
    
    show()