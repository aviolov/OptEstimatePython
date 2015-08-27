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

from AdjointSolver import FPAdjointSolver
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
                              step_size_base = 10.,
                              initial_ts_cs = None,
                               visualize=False,
                               Kmax = 100):
    #Interface for drivers:
    return gdOptimalControl_Aggressive(tau_chars, tau_char_weights,
                                       mu_sigma,
                                       Tf=Tf,   
                                       alpha_bounds=alpha_bounds,
                                       grad_norm_tol=grad_norm_tol,
                                       obj_diff_tol=obj_diff_tol,
                                       soln_diff_tol=soln_diff_tol,
                                          step_size_base = step_size_base,
                                          initial_ts_cs=initial_ts_cs,
                                          visualize=visualize,
                                          K_max=Kmax)

def gdOptimalControl_Old(params, Tf,
                            energy_eps = .001, alpha_bounds = (-2., 2.),
                            J_tol = 1e-3, gradH_tol = 1e-2, K_max = 100,  
                            alpha_step = 0.05,
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
                                initial_ts_cs = None, 
                                visualize=False):
    
    print 'Aggresive Gradient Descent: TODO: Redefine active nodes to include those at the boundary but pointing inwards'
    S = generateDefaultAdjointSolver(tau_chars, mu_sigma, alpha_bounds   );
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
        print 'k=%d, J_k=%.4f, ||g_k||_active=%.4f, g_tol_effective=%.4f,'%(k,
                                                J_current,
                                                active_grad_norm,
                                                effective_grad_tol)
        
        for tdx in xrange(2):
            print '%.2f:%.3f'%(tau_chars[tdx], -sum(diff(sum(fs[tdx,:,:], axis=0)*S._dx)))
        
        'Grad convergence?:'
        if active_grad_norm <= effective_grad_tol:
            print 'active grad_norm = %.6f < %.6f, convergence!'%(active_grad_norm,
                                                                  effective_grad_tol);
            break
                         
        #Single line minimization: (step_size selection:
         

        #Single step search:
#        step_size /=step_size_reduce_factor;    #try to be a little more aggressive
        alpha_next, J_next = None, None
        single_step_failed = False;
        for k_ss in xrange(K_singlestep_max):
            #generate proposed control
            alpha_next = incrementAlpha(a_k=step_size,
                                        d_k=-minus_grad_H);
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
    
def generateDefaultAdjointSolver(tau_chars,
                                 mu_sigma,
                                 alpha_bounds = [-2,2] ):
    xmin = FPAdjointSolver.calculate_xmin(alpha_bounds, tau_chars , mu_sigma, num_std = 2.0)
    dx = FPAdjointSolver.calculate_dx(alpha_bounds, tau_chars, mu_sigma, xmin, factor=0.5)
    dt = FPAdjointSolver.calculate_dt(alpha_bounds, tau_chars, mu_sigma, dx, xmin, factor = 2.0)
    print 'Solver params: xmin, dx, dt', xmin,dx,dt
    print 'Tf = ',Tf
    #Set up solver
    return FPAdjointSolver(dx, dt, Tf, xmin)
    
def FBKDriver(tau_chars, tau_char_weights,
              mu_sigma, Tf= 10.,
              alpha_bounds= (-2,2) ,                
              save_soln = True,
              visualize_decsent = True,
              soln_save_name = None):
    print tau_chars, tau_char_weights, mu_sigma, Tf
     
    init_ts = linspace(0, Tf, 1000)
#    init_cs  = 0.9*alpha_bounds[-1]*cos(2*pi*init_ts/Tf);
    cut_pt = 2*Tf/3
    init_cs  = 0.95*alpha_bounds[-1]* tanh( 4*(init_ts - cut_pt ) );
    init_cs =  1.0* ( ( alpha_bounds[1] - alpha_bounds[0]) * (init_ts/init_ts[-1]) +alpha_bounds[0])
#    init_cs = 0.*ones_like(init_ts);

    lSolver, fs, ps, cs_iterates, J_iterates = \
        calculateOptimalControl(tau_chars, tau_char_weights,
                                 mu_sigma,
                                  Tf=Tf ,  
                                 alpha_bounds=alpha_bounds, 
                                 initial_ts_cs=[init_ts, init_cs],
                                 visualize=visualize_decsent) 
    
    if save_soln:
        (FBKSolution(tau_chars, tau_char_weights,
                     mu_sigma, alpha_bounds,
                     lSolver,
                     fs, ps,
                     cs_iterates, J_iterates)).save(soln_save_name)
                

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
    ts,xs, cs_init, cs_opt, Js, fs = fbkSoln._Solver._ts, fbkSoln._Solver._xs,\
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
    
    
    axc.set_xlim(ts[0], ts[-1]);
    axc.set_ylim(cmin ,cmax);
    time_ticks = [ts[-1] /3. , 2.*ts[-1] /3. ,ts[-1] ]
    axc.set_xticks(time_ticks)
    axc.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
 
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
            
            print 'tau_char =%f : int(g) = %f'%(tau_char, sum(lg)*fbkSoln._Solver._dt)
    axg.legend( loc='upper left')
#    axg.set_xlabel('$t$', fontsize = xlabel_font_size);
    axg.set_ylabel(r'$g (t| \tau_c)$', fontsize = xlabel_font_size);
    axg.set_title('Initial Guess Hitting Time Densities', fontsize=xlabel_font_size)
    
    axg.set_xlim(ts[0], ts[-1]); 
    axg.set_xticks(time_ticks)
    axg.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
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
            
            print 'tau_char =%f : int(g) = %f'%(tau_char, sum(lg))
    axg.legend(loc='upper left')
    axg.set_xlabel('$t$', fontsize = xlabel_font_size);
    axg.set_ylabel(r'$g (t| \tau_c)$', fontsize = xlabel_font_size);
    axg.set_title('Optimal Control Hitting Time Densities', fontsize=xlabel_font_size) 
    axg.set_xticks(time_ticks)
    axg.set_xticklabels(['$%.1f$'%x for x in time_ticks],  fontsize = label_font_size) 
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
   
    optControls = {}; ts = {};
    cmin = -2; cmax = 2;
    for prior_tag, tau_chars in zip(['2pt prior', '3pt prior'], tau_chars_list):
        Ntaus = len(tau_chars)
        print tau_chars
        tau_char_weights = ones_like(tau_chars)/Ntaus
     
        if resimulate: 
            FBKDriver(tau_chars, tau_char_weights,
                      mu_sigma,
                      alpha_bounds=alpha_bounds,
                      Tf=Tf,
                      visualize_decsent = False);
                      
        'Visualize Regimes'
        visualizeRegimes(tau_chars,
                         mu_sigma,
                         Tf)  # fig_name= 'GradientAscent_Nt%d'%len(tau_chars) )#,  

        fbkSoln = FBKSolution.load(mu_beta_Tf_Ntaus = mu_sigma + [Tf] + [len(tau_chars)])
        
        ts[prior_tag] = fbkSoln._Solver._ts;
        optControls[prior_tag] =   fbkSoln._cs_iterates[-1] 
                        
        cmin, cmax = [x for x in fbkSoln._alpha_bounds]
    
    
    
    '''CONTROL FIGURE'''
    solnfig = figure(figsize=(17,6))
    subplots_adjust(bottom = 0.2, left=.15, right=.975 )
   
    axc = solnfig.add_subplot(1, 1, 1);     axc.hold(True);
    for prior_tag, tau_chars in zip(['2pt prior', '3pt prior'], tau_chars_list):
        axc.plot(ts[prior_tag], optControls[prior_tag]  , linewidth = 3)
        

    ts = ts['2pt prior'];
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
                              
     
def PriorSpreadBox(mu_sigma, Tf, 
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
                      Tf=Tf,
                      visualize_decsent = False,
                      soln_save_name=soln_name);
                      
        'Visualize Regimes'
        visualizeRegimes(tau_chars,
                         mu_sigma,
                         Tf,
                         soln_name = soln_name)#                       fig_name= 'GradientAscent_Nt%d'%len(tau_chars) )#,  

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
           
    

def OptimizerICsBox(mu_sigma, Tf=12,
                  alpha_bounds = (-2,2),
                   resimulate=True):
    
    '''Check whether starting the optiizer in different ICs result in the same final Opt Control
    CONCLUSION: Yep...!!!
    '''
    Ntaus = 2
    tau_chars_list = [ linspace(0.25, 4.0, Ntaus), 
                       linspace(0.5, 2.0, Ntaus),
                       linspace(0.8,1.25, Ntaus)]
    prior_tags = [ 'wide_prior', 'medium_prior', 'concentrated_prior'] 
    
#    tau_chars_list = tau_chars_list[ 2:   ]
#    prior_tags = prior_tags[2:  ] 
    
#    tau_chars_dict = dict(zip(tau_chars_list,  prior_tags));
    alpha_min  = 0.9*alpha_bounds[0];     
    alpha_max  = 0.9*alpha_bounds[-1];  
    init_ts = linspace(0, Tf, 100);
    init_ts_cs_dict = {'zero' : [init_ts, zeros_like(init_ts)],
                       'linear': [init_ts,  (alpha_max-alpha_min)* (init_ts/Tf ) + alpha_min ], #'zero': [init_ts, zeros_like(init_ts)], #'zero': [init_ts, zeros_like(init_ts)],
                       'cos': [init_ts,  alpha_max*cos(2*pi*init_ts/Tf)]} 
    
    'MAIN LOOP:'
    for tdx, (tau_chars, prior_tag) in enumerate(zip(tau_chars_list, 
                                                     prior_tags)):
        
        print tdx,':', prior_tag, '\t', tau_chars[0], tau_chars[-1]
        tau_char_weights = ones_like(tau_chars)/len(tau_chars); 
        
        resultsDict = {};
        
        for init_tag, init_ts_cs in init_ts_cs_dict.iteritems():
            soln_tag = prior_tag + init_tag;
             
            if resimulate:
                'Solve'
                lSolver, fs, ps, cs_iterates, J_iterates = \
                    calculateOptimalControl(tau_chars, tau_char_weights,
                                            mu_sigma,
                                            Tf=Tf ,  
                                            alpha_bounds=alpha_bounds,
                                            initial_ts_cs = init_ts_cs, 
                                            visualize=False)
    
                'Save'
                (FBKSolution(tau_chars, tau_char_weights,
                         mu_sigma, alpha_bounds,
                         lSolver,
                         fs, ps,
                         cs_iterates, J_iterates)).save( soln_tag )

                
                         
            visualizeRegimes(tau_chars, mu_sigma, Tf, soln_name = soln_tag)            
            resultsDict[init_tag] =  FBKSolution.load( soln_tag )
        
        'Visualize per prior:'
        figure(figsize=(17,12))
        for init_tag, fbkSoln in resultsDict.iteritems():
            subplot(211); hold(True)
            plot(fbkSoln._Solver._ts,
                  fbkSoln._cs_iterates[-1], label=init_tag)
            legend(loc='lower right');
            subplot(212); hold(True);
            plot(-array(fbkSoln._J_iterates), 'x-', label=init_tag);
            legend();
            print fbkSoln._J_iterates, init_tag
        title(prior_tag);
        
     

def KnownParametersBox( Tf=12,
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
    
    alpha_min  = 0.9*alpha_bounds[0];     
    alpha_max  = 0.9*alpha_bounds[-1];  
    init_ts_cs = [linspace(0, Tf, 100),
                   (alpha_max-alpha_min)* (linspace(0, Tf, 100)/Tf ) + alpha_min ] 
    
    'MAIN LOOP:'
    resultsDict = {};
    for tdx, (mu_sigma, param_tag) in enumerate(zip(mu_sigma_list, 
                                                     param_tags)):
        
        print tdx,':', param_tag, '\t', mu_sigma[0], mu_sigma[-1]
        
         
        soln_tag = param_tag ;
         
        if resimulate:
            'Solve'
            lSolver, fs, ps, cs_iterates, J_iterates = \
                calculateOptimalControl(tau_chars, tau_char_weights,
                                        mu_sigma,
                                        Tf=Tf ,  
                                        alpha_bounds=alpha_bounds,
                                        initial_ts_cs = init_ts_cs, 
                                        visualize=False)

            'Save'
            (FBKSolution(tau_chars, tau_char_weights,
                     mu_sigma, alpha_bounds,
                     lSolver,
                     fs, ps,
                     cs_iterates, J_iterates)).save( soln_tag )

#        visualizeRegimes(tau_chars, mu_sigma, Tf, soln_name = soln_tag)            
        resultsDict[param_tag] =  FBKSolution.load( soln_tag )
            
                     
        
    'Visualize per prior:'
    figure(figsize=(17,12))
    for tag, fbkSoln in resultsDict.iteritems():
        subplot(211); hold(True)
        plot(fbkSoln._Solver._ts,
              fbkSoln._cs_iterates[-1], label= tag)
        legend(loc='lower right');
        subplot(212); hold(True);
        plot(-array(fbkSoln._J_iterates), 'x-', label=tag);
        legend();
        
        print  tag, fbkSoln._J_iterates[0], fbkSoln._J_iterates[-1]
    title('Different Known Params');
        
          
if __name__ == '__main__':
    from pylab import *
    Tf =  15;
    alpha_bounds = (-2., 2.);
    
#    tau_chars = linspace(.25, 4, 2);
    tau_chars = linspace(0.25, 4, 2);
#    tau_chars = [2.5];
    tau_char_weights = ones_like(tau_chars) / len(tau_chars)
    
    mu_sigma = [0., 1.] 
#    mu_sigma = [0.0, 1.0] 
 
    'Solve a full Gradient Descent + (save+visualize)'
#    FBKDriver(tau_chars, tau_char_weights,
#              mu_sigma,
#              alpha_bounds=alpha_bounds,
#              Tf=Tf );   
    
    'Visualize Regimes (more details...?)'
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
    
   
    ''' sweep over the number of taus (shape of the prior)'''
#    NtausBox(mu_sigma, Tf, alpha_bounds=alpha_bounds,
#             resimulate=False);
             

    ''' check if there is a difference between a wide spread prior vs.
     a concentrated prior for the Optimal Stimulation:
      Conclusion: There is!'''
#    PriorSpreadBox(mu_sigma, Tf, alpha_bounds, resimulate=False)
 
    
    'Check effect of varying optimizer ICs make a difference:'
#    OptimizerICsBox(mu_sigma,resimulate=False)


    '''Check effect of varying mu, sigma'''
    KnownParametersBox(resimulate=False)
    
    

    show();