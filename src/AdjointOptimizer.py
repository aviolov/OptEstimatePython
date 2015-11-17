""" 
Created on Aug 15, 2015

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
from matplotlib.font_manager import FontProperties
from scipy import interpolate 

from AdjointSolver import FPAdjointSolver, generateDefaultAdjointSolver
import ext_fpc
from HitTime_MI_Beta_Estimator import SimulationParams


#from TauParticleEnsemble import TauParticleEnsemble

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/AdjointOptimizer/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/AdjointOptimizer/'

import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
        
import time


label_font_size = 32
xlabel_font_size = 40
ABCD_LABEL_SIZE = 28


class AdjointOptimizer():
    'The class encapsulator fot the Optimization'    
    def __init__(self,tau_chars, tau_char_weights, 
                    mu_sigma,   
                    Tf = 1.0,  Tf_opt=None,
                        alpha_bounds = (-2., 2.),
                        visualize=False):
        
        'Construct the backbone - pde Solver:'        
        self.Solver = generateDefaultAdjointSolver(tau_chars, mu_sigma,
                                         alpha_bounds,
                                         Tf, Tf_opt=Tf_opt);
                                         
        if self.Solver._dx >0.01 or self.Solver._dt > 0.005:
            print 'Refining solver grid to minimal refinement required'
            self.Solver.rediscretize(0.01, 0.005,
                                     self.Solver.getTf(), self.Solver.getXmin(), self.Solver.getXthresh());

        'MOdel Parametes:'
        self.tau_chars = tau_chars;
        self.tau_char_weights = tau_char_weights;        
        self.mu_sigma = mu_sigma;
        
        'control constraints:'
        self.alpha_bounds = alpha_bounds;
        
        'visualization flag:'
        self.visualize = visualize;
        
                                         
    'generate initial alpha control:'                
    def  getInitialControl(self, ts, initial_ts_cs):
        if (None == initial_ts_cs):
            #            initial_control = alpha_max/2.0*ones_like(ts)
            return zeros_like(ts)
        else:
            return interp(ts, initial_ts_cs[0], initial_ts_cs[1]);

    'Push alpha step_size in direction, up to constraints:'
    def incrementAlpha(self, alpha_current, 
                       step_size,
                       direction,
                       optimization_inds):
        
        alpha_min, alpha_max = self.alpha_bounds[0], self.alpha_bounds[1];
        e = ones_like(alpha_current);
                
        'copy current solution:'
        alpha_next = deepcopy(alpha_current);
                
        'push alpha over the perturbable inds:'                
        alpha_next[optimization_inds] = alpha_next[optimization_inds] +\
                                            step_size * direction[optimization_inds];
        'bound from below:'                                    
        alpha_bounded_below = amax(c_[alpha_min*e, alpha_next], axis=1);
    
        'bound from above and return'
        return amin(c_[alpha_max*e, alpha_bounded_below], axis=1)
   
    'return those nodes where the control can be perturbed:'
    def getActiveNodes(self, alpha_current, grad_H, opt_inds):
        active_nodes = ( (alpha_current[opt_inds] > self.alpha_bounds[0]) | (grad_H[opt_inds]>0) ) & \
                       ( (alpha_current[opt_inds] < self.alpha_bounds[1]) | (grad_H[opt_inds]<0) )
        if self.visualize:
            print 'active_nodes count = %d'%(len(active_nodes))
        return active_nodes;
     
    'given direction, make a step incrementing the objective:'
    def makeSingleStepIncrement(self,
                                curriedObjectiveFunc,
                                J_current,  alpha_current,
                                 direction, opt_inds,                                
                                 initial_step_size, 
                                 K_singlestep_max=10,
                                 step_size_reduce_factor=0.5):
        single_step_succeeded = False;        
        
        step_size = initial_step_size;
        for k_ss in xrange(K_singlestep_max):
            #generate proposed control
            alpha_next = self.incrementAlpha(alpha_current,
                                             step_size,
                                             direction,
                                             opt_inds);
             
            if self.visualize:
                print '\t%d:alpha diff norm = %.4f'%(k_ss, sum(abs(alpha_next-alpha_current)))

                            
            'Inner MAIN CALL:'
            fs, J_next = curriedObjectiveFunc(alpha_next);
                                                  
#           #Sufficient decrease?
            print '\tk_ss=%d, step_size=%.4f, J=%.4f '%(k_ss,
                 step_size, J_next);
             
            
#            sufficent_decrease = J_current - c1_wolfe*step_size * active_grad_norm*active_grad_norm
            sufficent_decrease = J_current;
            wolfe_1_condition = (J_next <= sufficent_decrease);
            
            if (wolfe_1_condition):
                print '\t sufficient decrease: %.6f < %.6f breaking' %(J_next,
                                                                       sufficent_decrease);
                step_size /= step_size_reduce_factor;
                
                single_step_succeeded=True;
                break;
             
            'reduce step_size and try again:'
            step_size *=step_size_reduce_factor
            ###Single step loop  
            
        return single_step_succeeded, J_next, alpha_next, step_size 
            
            
    
    'A basic gradient ascent optimizer'
    def basic_gradDescent(self,
                          initial_ts_cs = None,
                          step_size_base = 10.,
                          grad_norm_tol = 1e-5,
                          obj_diff_tol  = 1e-4,
                          soln_diff_tol = 1e-4,                          
                          K_max = 16,
                          K_singlestep_max = 10):
        
        S = self.Solver;
        'Get time-steps (optimizable/vs not):'                                 
        ts = S._ts;
        opt_ts = S.getOptTs();
        opt_inds = S.getOptTs_indices();
        
        'rip a_0'
        alpha_current = self.getInitialControl(ts, initial_ts_cs );        
        if self.visualize:
            figure(); plot(ts, alpha_current); title('initial control');            
            gradHfig = figure();
            
        'The return iteration-lists:'
        alpha_iterations = []; J_iterations = [];
        'the return objective variables:'    
        fs = None; ps = None;
           
        'the incremental differences for convergence purposes:'
        J_previous = nan; alpha_previous = nan*ones_like(alpha_current);
        
        'the increment step-size (delta alpha)'
        step_size = step_size_base;
        
        '''Main Incremental Loop (ascent):''' 
        for k in xrange(K_max):                         
            'Store current (control,objective) solutions:'            
            alpha_iterations.append(alpha_current);
            '''First MAIN CALL:'''
            _, ts, fs, ps, J_current, grad_H =\
                S.solve(self.tau_chars, self.tau_char_weights, self.mu_sigma, 
                             alpha_current);
            J_iterations.append(J_current);
                          
            'rip active nodes - those nodes where control-perturbations  are allowed:'                                           
            active_nodes = self.getActiveNodes(alpha_current, grad_H, opt_inds);
        
            'compute the current grad-norm:'
            active_grad_norm = sqrt(dot(grad_H[active_nodes], grad_H[active_nodes]));
            effective_grad_tol = grad_norm_tol * len(alpha_current[active_nodes]);
            print 'k=%d, J_k=%.4f, ||g_k||_active=%.4f, g_tol_effective=%.4f,'%(k,
                         J_current, active_grad_norm, effective_grad_tol)
        
            'Diagnostics:'
            if self.visualize:
                for tdx in xrange(2):
                    print 'conservation integral: %.2f:%.3f'%(tau_chars[tdx], -sum(diff(sum(fs[tdx,:,:], axis=0)*S._dx)))
                gradHfig.add_subplot(K_max, 2, 2*k+1);
                plot(opt_ts, grad_H); title('k=%d, GradH'%k);
                gradHfig.add_subplot(K_max, 2, 2*k+2); 
                plot(opt_ts[active_nodes], grad_H[active_nodes], '.');
                title('GradH_{active}');        
        
            'Grad convergence?'
            if active_grad_norm <= effective_grad_tol:                
                print 'active grad_norm = %.6f < %.6f => CONVERGENCE!'%(active_grad_norm,
                                                                  effective_grad_tol);
                break
            
            
            'Single Step Increment:'
            curriedObjectiveFunc = lambda alpha_incremented : self.Solver.solveObjectiveOnly(self.tau_chars,
                                                                                             self.tau_char_weights,
                                                                                             self.mu_sigma,
                                                                                             alpha_incremented);
            success_flag, J_current, alpha_current, step_size = \
                    self.makeSingleStepIncrement(curriedObjectiveFunc,
                                                 J_current,
                                                 alpha_current,
                                                 grad_H, opt_inds,
                                                 step_size, K_singlestep_max);
            if False==success_flag:
                print 'Single Step Increment Failed! Bailing...'
                break
            
            'Enforce a max step-size:'
            step_size = min([10*step_size_base, step_size]);   
            
            'objective diff convergence?'
            objective_diff_norm = abs(J_current-J_previous);
            if objective_diff_norm < obj_diff_tol:
                print 'objective diff norm = %.6f < %.6f => CONVERGENCE!'%(objective_diff_norm, 
                                                                           obj_diff_tol);
                break
            
            'Update previous obj. value'
            J_previous = J_current;      
            
            'Did NOT Converge?'
            if (K_max-k)==1:
                print('WARNING: DID NOT CONVERGE in %d iterations'%K_max)            
            #//Main Ascent Loop
    
        'Post Analysis:'
        if self.visualize:   
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
            for (tdx,tau) in enumerate(tau_chars):
                lf = S.hittime_distn_via_conservation(fs[tdx,:,:]);
                plot(ts, lf, label='tau=%.2f'%tau)
            legend()
            
        return  S, fs, ps, alpha_iterations, J_iterations
  
################################################################################
'A sponge data structure to contain the solution of a Optimization Roll:'
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

'Main Interface Routine to compute the optimal control for a set of Input Parameters'    
def FBKDriver(tau_chars, tau_char_weights,
              mu_sigma, 
              Tf= 10., Tf_opt = None,
              alpha_bounds= (-2,2) ,                
              save_soln = True, 
              soln_save_name = None,
              visualize_decsent = True,             
              visualize_summary=True,
              init_ts = None, init_cs = None):
    if len(tau_chars) < 2:
        raise RuntimeError('FBKDriver::expected more than 1 tau-chars')
    print 'FBKDriver::', tau_chars, tau_char_weights, mu_sigma, Tf
    
    if None == init_ts:
        init_ts = linspace(0, Tf, 1000);
        
    if None == init_cs:
        init_cs =  ( ( alpha_bounds[1] - alpha_bounds[0]) *  init_ts/Tf_opt ) + alpha_bounds[0];

    'Enforce max-control past the point of optimization'
    init_cs[where(init_ts>Tf_opt)] = alpha_bounds[1];

    
    'Construct Optimizer:'
    lOptimizer = AdjointOptimizer(tau_chars, tau_char_weights, mu_sigma,
                                  Tf, Tf_opt, 
                                    alpha_bounds= alpha_bounds,
                                    visualize=visualize_decsent);
                                    
    'Main Optimization Call:'
    lSolver, fs, ps, cs_iterates, J_iterates = \
                lOptimizer.basic_gradDescent(initial_ts_cs = [init_ts, init_cs],
                                             step_size_base = 10.0,
                                             K_max=10);
                                             
    'Save results?'
    if save_soln:
        (FBKSolution(tau_chars, tau_char_weights,
                     mu_sigma, alpha_bounds,
                     lSolver,
                     fs, ps,
                     cs_iterates, J_iterates)).save(soln_save_name)
                     
    'Visualize?'                 
    if visualize_summary:
        visualizeRegimes(tau_chars, mu_sigma, Tf)
                

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
    '''Visualize a single Gradient Ascent Evolution (intial->final)
    Plot Controls for Start vs Optimal
    Plot Corresponding Hitting TImes Densities
    Plot Objective Increment
    '''
    
    'load soln:'
    if None == soln_name:
        fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = mu_sigma + [Tf] + [len(tau_chars)])
    else:
        fbkSoln = FBKSolution.load(file_name = soln_name);    

    'create figure:'
    solnfig = figure(figsize=(17,16))
    subplots_adjust(hspace = 0.6, left=.15, right=.975 )
    
    '''Load Results''' 
    ts, Tfopt, cs_init, cs_opt, Js, fs = fbkSoln._Solver._ts,  fbkSoln._Solver.Tf_optimization,\
                         fbkSoln._cs_iterates[0], fbkSoln._cs_iterates[-1],\
                         fbkSoln._J_iterates,\
                         fbkSoln._fs; 
                        
    cmin, cmax = [x for x in fbkSoln._alpha_bounds]
    
    '''CONTROL EVOLUTION FIGURE'''
    axc = solnfig.add_subplot(4, 1, 1)
    axc.hold(True);
    axc.plot(ts, cs_init, 'b', linewidth=3)
    axc.plot(ts, cs_opt, 'r', linewidth = 3)
    axc.hlines(0, ts[0], ts[-1], linestyles='--')
    axc.legend(['Init Guess', 'Opt Soln (%d iterations)'%len(fbkSoln._cs_iterates) ],
               loc='upper left')
    
    axc.vlines(Tfopt, cmin ,cmax, linestyles='--')
    axc.set_xlim(ts[0], ts[-1]);
    axc.set_ylim(cmin ,cmax);
    time_ticks = [ts[-1] /3. , Tfopt  ,ts[-1] ]
    axc.set_xticks(time_ticks)
    xtick_labels = ['$%.1f$'%x for x in time_ticks];
    xtick_labels[1] = '$t_{opt}$';
    axc.set_xticklabels(xtick_labels,  fontsize = label_font_size) 
 
    ticks = [cmin,   .0 ,cmax]
    axc.set_yticks(ticks)
#    axc.set_xlabel('$t$', fontsize = xlabel_font_size);
    axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    axc.set_title('Control Iteration', fontsize=xlabel_font_size) 
    axc.set_yticklabels(('$%.1f$'%cmin, '$0$','$%.1f$'%cmax),
                         fontsize = label_font_size) 
    
   
    '''hitting time density (INITIAL)'''
    axg = solnfig.add_subplot(4, 1, 2); hold(True)
    gtickmax = 0.8    
    for (tdx,tau_char) in enumerate(tau_chars):
            lg = fbkSoln._Solver.solve_hittime_distn_per_parameter(tau_char,
                                                                   [fbkSoln._mu, fbkSoln._sigma],cs_init)
            axg.plot(ts, lg, label = r'$\tau_c=%.2g$'%tau_char,
                     linewidth = 2)
            
            print 'init: tau_char =%f : int(g) = %f'%(tau_char, sum(lg)*fbkSoln._Solver._dt)
    axg.legend( loc='upper left')
#    axg.set_xlabel('$t$', fontsize = xlabel_font_size);
    axg.set_ylabel(r'$g (t| \tau_c)$', fontsize = xlabel_font_size);
    axg.set_title('Initial Guess Hitting Time Densities', fontsize=xlabel_font_size)
    
    axg.set_xlim(ts[0], ts[-1]); 
    axg.set_xticks(time_ticks)
    axg.set_xticklabels(xtick_labels,  fontsize = label_font_size) 
    axg.set_ylim(0, gtickmax);  
    axg.set_yticks([0, gtickmax])
    axg.set_yticklabels(('$0$','$%.1f$'%gtickmax),
                         fontsize = label_font_size) 
    
    '''hitting time density (OPTIMAL)'''
    axg = solnfig.add_subplot(4, 1, 3); hold(True)
    D = fbkSoln._sigma**2/2.0
     
    for (tdx,tau_char) in enumerate(tau_chars):
            lg = -diff(sum(fs[tdx,:,:], axis=0)*fbkSoln._Solver._dx)
            
            axg.plot(ts[:-1], lg/fbkSoln._Solver._dt, label = r'$\tau_c=%.2g$'%tau_char,
                     linewidth = 2)
            
            print 'opt: tau_char =%f : int(g) = %f'%(tau_char, sum(lg))
    axg.legend(loc='upper left')
    axg.set_xlabel('$t$', fontsize = xlabel_font_size);
    axg.set_ylabel(r'$g (t| \tau_c)$', fontsize = xlabel_font_size);
    axg.set_title('Optimal Control Hitting Time Densities', fontsize=xlabel_font_size) 
    axg.set_xticks(time_ticks)
    axg.set_xticklabels(xtick_labels,  fontsize = label_font_size) 
    axg.set_xlim(ts[0], ts[-1]); 
    axg.set_ylim(0, gtickmax);  
    axg.set_yticks([0, gtickmax])
    axg.set_yticklabels(('$0$','$%.1f$'%gtickmax),
                         fontsize = label_font_size) 
        
    '''Objective Progress'''
    axj = solnfig.add_subplot(4, 1, 4)
#    axj.hold(True);
    axj.plot(arange(1, len(Js)+1), -array(Js), 'bx-', linewidth=3)
    axj.set_title('Gradient Ascent of the Objective $I$', fontsize=xlabel_font_size )
    axj.set_ylabel(r'$I_k$',fontsize= xlabel_font_size); 
    axj.set_xlabel('$k$', fontsize= xlabel_font_size);  
    axj.set_ylim(0, max(-array(Js))) 
    ticks = arange(1, len(Js)+1).astype(int) 
    axj.set_xticks(ticks)
    axj.set_yticks([0, -Js[-1]])
    axj.set_yticklabels(('$0$','$%.1f$'%-Js[-1]),
                         fontsize = label_font_size) 
        
    for tick in axj.xaxis.get_major_ticks():
                tick.label.set_fontsize(label_font_size)

    '''save?'''    
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '.pdf')
        print 'saving to ', lfig_name
        solnfig.savefig(lfig_name, dpi=300)       


def sweepSwitchpoint(tau_chars,
                     mu_sigma, 
                     Tf,    
                     recompute_sweep = True,
                     fig_name = None,
                     alpha_bounds = [-2,2]             ):
    
    'load soln:'
    save_soln_name = os.path.join(RESULTS_DIR, 'sweepSwitchpoint.npy')
    if recompute_sweep:
        tswitches = linspace(-1, Tf, ceil(Tf));
        
        'get default solver'
        S = generateDefaultAdjointSolver(tau_chars, mu_sigma, alpha_bounds   );
    
        ts = S._ts;
#        alpha_min, alpha_max = alpha_bounds[0], alpha_bounds[1]
        Js = empty_like(tswitches);

        figure(); hold(True)
        for idx, tswitch in enumerate(tswitches):
            '(bang-bang) Control:'
            alphas =  0.99*alpha_bounds[-1]* tanh( 4*(ts - tswitch ) );
            if 0 == idx:
                alphas *=0.0;
            plot(ts, alphas);

            'Densities:'
            fs = S._fsolve( tau_chars, tau_char_weights,
                            mu_sigma, alphas)
            '''Objective:''' 
            Js[idx] = S.calcObjective(tau_char_weights, fs);
        tswitch_Js = transpose(array([tswitches, Js]));
        
        print tswitch_Js
        'save:'
        numpy.save(save_soln_name, tswitch_Js );
    else:
        'load:'
        tswitch_Js = numpy.load(save_soln_name);

    '''#########################
    VISUALIZE:
    ############################'''
    'Create figure:'
    solnfig = figure(figsize=(17,6))
    subplots_adjust( left=.15, right=.9 , bottom = .2)
    
    '''Controls Figure'''
    ts = tswitch_Js[1:, 0]; Js = tswitch_Js[1:, 1];
    
    axj = solnfig.add_subplot(1, 1, 1); 
    plot(ts, Js, 'rx-', linewidth=3); hold(True);
    plot(ts, tswitch_Js[0, 1]*ones_like(ts), 'b', linewidth=3)
    legend(['bang-bang(ts)', 'do nothing (always 0)'])
    xlabel('$t_{switch}$', fontsize = xlabel_font_size); ylabel('$I$', fontsize=xlabel_font_size)
     
    max_J = amax(array(Js) )
    ylim([0, max_J*1.1])
    
    axj.set_title('Mutual Info, $I$, as function of switching time', fontsize=xlabel_font_size )
    axj.set_yticks([0, max_J])
    axj.set_yticklabels(('$0$','$%.1f$'%max_J),
                         fontsize = label_font_size) 
    time_ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
    axj.set_xticks(time_ticks)
    axj.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
    axj.set_xlim([ts[0], ts[-1]])
    
    if None != fig_name:
        fig_name = os.path.join(FIGS_DIR, fig_name+'.pdf');
        print 'saving to ', fig_name
        savefig(fig_name, dpi=300) 
        

    
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
    psolve(in C)       takes 0.066 secs
    psolve (in python) takes 24.164 secs'''    


def NtausBox(mu_sigma, Tf, 
             alpha_bounds = (-2,2),
             resimulate=False, fig_name = 'NumberOfTausEffect'):
    tau_upper =  2 + log(sqrt(3/2))
    
    tau_chars_list = [[0.5, 2],
                      [1./tau_upper, 1., tau_upper ] ]
    
    tau_chars_many = linspace(0.25,4,16);
    tau_mean = mean(tau_chars_many);
    tau_std_dev = std(tau_chars_many);
    print tau_mean, tau_std_dev
    tau_chars_2 = [tau_mean-tau_std_dev, tau_mean+tau_std_dev]
    tau_chars_list = [tau_chars_2, tau_chars_many]

    Tfopt = 2/3*Tf;
    
    optControls = {}; ts = {};
    cmin = -2; cmax = 2;
    for prior_tag, tau_chars in zip(['2pt_prior', '3pt_prior'], tau_chars_list):
        Ntaus = len(tau_chars)
        print prior_tag
        tau_char_weights = ones_like(tau_chars)/Ntaus
     
        if resimulate: 
            FBKDriver(tau_chars, tau_char_weights,
                      mu_sigma,
                      alpha_bounds=alpha_bounds,
                      Tf=Tf,Tf_opt=Tfopt,
                      init_ts = arange(0, Tf, 0.01),
                      init_cs = alpha_bounds[1]*cos(2*pi*arange(0, Tf, 0.01)/Tf_opt),
                      soln_save_name = prior_tag,
                      visualize_decsent = False,
                      visualize_summary = False);
                      
        'Visualize Regimes'
        visualizeRegimes(tau_chars,
                         mu_sigma,
                         Tf,
                         soln_name=prior_tag)  # fig_name= 'GradientAscent_Nt%d'%len(tau_chars) )#,  

        fbkSoln = FBKSolution.load(prior_tag)
        
        ts[prior_tag] = fbkSoln._Solver._ts;
        optControls[prior_tag] = fbkSoln._cs_iterates[-1] 
                        
        cmin, cmax = [x for x in fbkSoln._alpha_bounds]
    
    
    '''CONTROL FIGURE'''
    solnfig = figure(figsize=(17,6))
    subplots_adjust(bottom = 0.2, left=.15, right=.975 )
   
    axc = solnfig.add_subplot(1, 1, 1);     axc.hold(True);
    for prior_tag, tau_chars in zip(['2pt_prior', '3pt_prior'], tau_chars_list):
        axc.plot(ts[prior_tag], optControls[prior_tag]  , linewidth = 3)
        

    ts = ts['2pt_prior'];
    axc.hlines(0, ts[0], ts[-1], linestyles='--')
    axc.legend(['2pt prior', '3pt prior'] ,
               loc='upper left')
    axc.set_xlim(ts[0], ts[-1]);
    axc.set_ylim(cmin ,cmax);
    time_ticks = [ts[-1]/3. , 2.*ts[-1] /3. ,ts[-1] ]
    axc.set_xticks(time_ticks)
    axc.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
 
    ticks = [cmin,   .0 ,cmax]
    axc.set_yticks(ticks)
    axc.set_xlabel('$t$', fontsize = xlabel_font_size);
    axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    axc.set_title('Optimal Controls', fontsize=xlabel_font_size) 
    axc.set_yticklabels(('$%.1f$'%cmin, '$0$','$%.1f$'%cmax),
                         fontsize = label_font_size) 
    
    '''save?'''    
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '.pdf')
        print 'saving to ', lfig_name
        solnfig.savefig(lfig_name, dpi=300)       
                              
     
def PriorSpreadBox(mu_sigma, Tf, Tfopt,
                   alpha_bounds = (-2,2), resimulate=False,
                   fig_name = 'PriorSpread'):
    
    '''Chech whether priors of different width (spread) result in different optimal
    stimulation (perturbation/control). 
    
    CONCLUSION: It looks like it really does!!!
    '''
    
    tau_chars_list = [[.25, 4],
                      [.75, 1/.75 ] ]
   
    priorTags = ['wide prior', 'narrow prior']
    optControls = {}; ts = {};
    cmin = -2; cmax = 2;
    
    for prior_tag, tau_chars in zip(priorTags, tau_chars_list):
        Ntaus = len(tau_chars)
        print tau_chars
        tau_char_weights = ones_like(tau_chars)/Ntaus
     
        soln_name = 'PriorSpread_%s'%prior_tag
        if resimulate: 
            FBKDriver(tau_chars, tau_char_weights,
                      mu_sigma,
                      alpha_bounds=alpha_bounds,
                      Tf=Tf, Tf_opt = Tfopt,
                      visualize_decsent = False,
                      visualize_summary=True,
                      soln_save_name = soln_name);
       
        fbkSoln = FBKSolution.load(soln_name)
        
        ts[prior_tag] = fbkSoln._Solver._ts;
        optControls[prior_tag] =   fbkSoln._cs_iterates[-1] 
                        
        cmin, cmax = [x for x in fbkSoln._alpha_bounds]
    
    '''CONTROL FIGURE'''
    solnfig = figure(figsize=(17,6))
    subplots_adjust(bottom = 0.2, left=.15, right=.975 )
   
    axc = solnfig.add_subplot(1, 1, 1);     axc.hold(True);
    for prior_tag, tau_chars in zip(priorTags, tau_chars_list):
        axc.plot(ts[prior_tag], optControls[prior_tag]  , linewidth = 3)
        

    ts = ts[priorTags[0]];
    axc.hlines(0, ts[0], ts[-1], linestyles='--')
    axc.vlines(Tfopt, cmin, cmax, linestyles='--')
    axc.legend(priorTags ,
               loc='upper left')
    axc.set_xlim(ts[0], ts[-1]);
    axc.set_ylim(cmin ,cmax);
    time_ticks = [ts[-1]/3. , 2.*ts[-1] /3. ,ts[-1] ]
    axc.set_xticks(time_ticks)
    axc.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
 
    ticks = [cmin,   .0 ,cmax]
    axc.set_yticks(ticks)
    axc.set_xlabel('$t$', fontsize = xlabel_font_size);
    axc.set_ylabel(r'$\alpha(t)$', fontsize = xlabel_font_size);
    axc.set_title('Optimal Controls', fontsize=xlabel_font_size) 
    axc.set_yticklabels(('$%.1f$'%cmin, '$0$','$%.1f$'%cmax),
                         fontsize = label_font_size) 
    
    '''save?'''    
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR, fig_name + '.pdf')
        print 'saving to ', lfig_name
        solnfig.savefig(lfig_name, dpi=300)       
           
    

def OptimizerICsBox(mu_sigma, Tf=12, Tf_opt = 8,
                    alpha_bounds = (-2,2),
                    resimulate=True,
                    fig_name='OptimizerICs'):
    '''Check whether starting the optimizer in different ICs result in the same final Opt Control
    '''
    Ntaus = 2
    tau_chars_list = [ linspace(0.25, 4.0, Ntaus), 
                       linspace(0.5, 2.0, Ntaus),
                       linspace(0.8,1.25, Ntaus)]
    prior_tags = [ 'wide_prior', 'medium_prior', 'concentrated_prior'] 
    
    prior_titles = dict(zip(prior_tags, ['Wide Prior', 'Medium Prior', 'Concentrated Prior']))
    
#    tau_chars_list = tau_chars_list[ 2:   ]
#    prior_tags = prior_tags[2:  ] 
    
#    tau_chars_dict = dict(zip(tau_chars_list,  prior_tags));
    alpha_min  = 0.9*alpha_bounds[0];     
    alpha_max  = 0.9*alpha_bounds[-1];  
    init_ts = linspace(0, Tf, 100);
    init_ts_cs_dict = {'zero' : [init_ts, zeros_like(init_ts)],
                       'linear': [init_ts,  (alpha_max-alpha_min)* (init_ts/Tf_opt ) + alpha_min ],
                       'cos': [init_ts,  alpha_max*cos(2*pi*init_ts/Tf_opt)],
                       'max': [init_ts, alpha_max*ones_like(init_ts)]} 
    
    'MAIN LOOP:'
    for tdx, (tau_chars, prior_tag) in enumerate(zip(tau_chars_list, 
                                                     prior_tags)):
        
        print tdx,':', prior_tag, '\t', tau_chars[0], tau_chars[-1]
        tau_char_weights = ones_like(tau_chars)/len(tau_chars); 
        
        resultsDict = {};
        
        for init_tag, init_ts_cs in init_ts_cs_dict.iteritems():
            soln_tag = prior_tag + init_tag;
             
            if resimulate:
                'Solve:'
                FBKDriver(tau_chars, tau_char_weights, mu_sigma,
                           Tf, Tf_opt, alpha_bounds,
                            save_soln=True, soln_save_name = soln_tag,
                            visualize_decsent=False, visualize_summary=False,
                             init_ts= init_ts_cs[0], init_cs=init_ts_cs[1])
            visualizeRegimes(tau_chars, mu_sigma, Tf, soln_name= soln_tag )
            
            resultsDict[init_tag] =  FBKSolution.load( soln_tag )
        
        'Visualize per prior:'
        solnfig = figure(figsize=(17,12))
        for init_tag, fbkSoln in resultsDict.iteritems():
            subplot(311); hold(True)
            plot(fbkSoln._Solver._ts,
                  fbkSoln._cs_iterates[0],
                  linewidth=3, label=init_tag)
            legend(loc='lower right');
            title('Initial Guesses for the Control', fontsize=xlabel_font_size);
            
            subplot(312); hold(True)
            plot(fbkSoln._Solver._ts,
                  fbkSoln._cs_iterates[-1],
                  linewidth=3, label=init_tag)
            title('Optimal Solution for the Control', fontsize=xlabel_font_size);
#            legend(loc='lower right');
            
            subplot(313); hold(True);
            plot(-array(fbkSoln._J_iterates), 
                 'x-', markersize=3, linewidth=2, label=init_tag);
            legend(loc='lower right');
            title('Objective Evolution', fontsize=xlabel_font_size);
#            
        '''save?'''    
        if None != fig_name:
            lfig_name = os.path.join(FIGS_DIR, fig_name + prior_tag + '.pdf')
            print 'saving to ', lfig_name
            solnfig.savefig(lfig_name, dpi=300)       
   

def KnownParametersBox( Tf=12,Tfopt=8,
                        alpha_bounds = (-2,2),
                        resimulate=True):
    '''Check whether varying mu,sigma has a significant impact on the 
    optimal control 
    CONCLUSION: Well, it only works for some mu, sigmas (the closer to 0,1 the better...!!!
    '''
    Ntaus = 2
    tau_chars  = linspace(0.5, 2.0, Ntaus)
    tau_char_weights = ones_like(tau_chars)/len(tau_chars); 
    
    mu_sigma_list = [(-.5, 1), (0.1,1), (1,1),
                     (0,  .3), (0, 0.9), (0,1.5) ]
                       
    param_tags = ['munegative', 'musmallperturb', 'muthresh',
                  'lownoise', 'sigmasmallperturb', 'highnoise']  
    
   
    'MAIN LOOP:'
    resultsDict = {};
    for tdx, (mu_sigma, param_tag) in enumerate(zip(mu_sigma_list, 
                                                     param_tags)):
        
        print tdx,':', param_tag, '\t', mu_sigma[0], mu_sigma[-1]
        soln_tag = param_tag;
        init_ts = linspace(0, Tf, 100);
         
        if resimulate:
            'Solve'
            FBKDriver(tau_chars, tau_char_weights,
                      mu_sigma,
                      alpha_bounds=alpha_bounds,
                      Tf=Tf,Tf_opt=Tfopt,
                      soln_save_name = soln_tag,
                      visualize_decsent = False,
                      visualize_summary = False,
                      init_ts = init_ts,
                      init_cs =   alpha_bounds[1]*cos(2*pi*init_ts/Tf_opt) );
              
             
        visualizeRegimes(tau_chars, mu_sigma, Tf, soln_name = soln_tag)            
        resultsDict[param_tag] =  FBKSolution.load( soln_tag )
            
                     
        
    'Visualize per prior:'
    figure(figsize=(17,12))
    for tag, fbkSoln in resultsDict.iteritems():
        subplot(211); hold(True)
        plot(fbkSoln._Solver._ts,  fbkSoln._cs_iterates[-1], 
             label= tag , linewidth = 3)
        legend(loc='lower right');
        
        subplot(212); hold(True);
        plot(-array(fbkSoln._J_iterates),
              'x-', label=tag, linewidth = 2);
        legend();
        
        print  tag, fbkSoln._J_iterates[0], fbkSoln._J_iterates[-1]
    title('Different Known Params');
    
def Workbox():
    Tf =  18;
    Tfopt = 12;
    
    alpha_bounds = (-2., 2.);
    mu_sigma = [0.0, 1.0]    
    for tau_chars in [linspace(0.5, 2, 2), 
                      linspace(0.25, 4, 2)]:
#    for tau_chars in [linspace(0.5, 2, 2) ]:
#    for tau_chars in [linspace(0.25, 4, 2)]:
        tau_char_weights = ones_like(tau_chars) / len(tau_chars)    
 
        'Solve a full Gradient Descent + (save+visualize)'
        FBKDriver(tau_chars, tau_char_weights,
              mu_sigma,
              alpha_bounds=alpha_bounds,
              Tf=Tf, Tf_opt=Tfopt);
                     
          
if __name__ == '__main__':
    from pylab import *
    Tf = 15; Tf_opt=10;
    alpha_bounds = (-2., 2.);    
    tau_chars = linspace(0.5, 2, 2);
    tau_chars = linspace(0.25, 4, 2);
    tau_char_weights = ones_like(tau_chars) / len(tau_chars)
    
    mu_sigma = [0., 1.] 
#    mu_sigma = [.0, 2.0] 
#    mu_sigma = [.0, 0.5]
 
    
    'Sandbox to play around with and calibrate pieces of the puzzle:'
#    Workbox(); 
 
    'Solve a full Gradient Descent + (save+visualize)'
#    FBKDriver(tau_chars, tau_char_weights,
#              mu_sigma,
#              alpha_bounds=alpha_bounds,
#              Tf=Tf, Tf_opt = Tf_opt );   
    
    'Visualize Regimes'
#    visualizeRegimes(tau_chars,
#                      mu_sigma,
#                       Tf,
#                       fig_name= 'GradientAscent_Nt%d'%len(tau_chars) )#,  
 
    '''Profile the computational bottlenecks:'''
#    OptControlProfiler( recompute = True );
    

    '''Sweep through bang-bang swith-time and compute impact on MutInfo, I(t_switch)'''
#    sweepSwitchpoint(tau_chars,
#                     mu_sigma,
#                       Tf,
#                       recompute_sweep=False,
#                       fig_name='SweepSwitchpoint_wide' )#  
    
   
    '''Is there a difference between the number of taus (shape of the prior, >2 moments)
    Conclusion: No significant difference'''
    NtausBox(mu_sigma, Tf, alpha_bounds=alpha_bounds,
             resimulate=True);
             

    '''Is there a difference between a wide spread prior vs.
       a concentrated prior for the Optimal Stimulation:
      Conclusion: There seems to be some difference!'''
#    PriorSpreadBox(mu_sigma, Tf, Tf_opt,
#                   alpha_bounds, resimulate=False)
 
    
    'Check effect of varying optimizer ICs:'
#    OptimizerICsBox(mu_sigma,resimulate=False)


    '''Check effect of varying mu, sigma'''
#    KnownParametersBox(resimulate=True)


    show();