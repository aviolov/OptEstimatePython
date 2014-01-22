# -*- coding:utf-8 -*-
"""
Created on Nov 11, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand
from numpy.polynomial import hermite
from OUML import RESULTS_DIR as OUML_RESULTS_DIR
from OUML import estimateParamsBeta, dt

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/HermiteBox/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/HermiteBox/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

#STRUCTURAL PARAMATER VALUES - HELD FIXED
tau = 20. #ms;
beta = 1/tau;
mu = -60.0; 
sigma = .1;
 
def hermiteSandpile():
    beta_mean = .0902
    beta_std  = .0562

    const = lambda x: 1.*ones_like(xs);
    quart = lambda x: x*x*x*x;
    
#    mean_f = lambda x: sqrt(2)*beta_std * x + beta_mean
#    var_f = lambda x: (sqrt(2)*beta_std * x)*(sqrt(2)*beta_std * x)
    mean_f = lambda bs: bs
    var_f = lambda bs: (bs - beta_mean)*(bs - beta_mean)
#    
    for deg in [5, 21,43]:
        print deg , ':'
        xs, ws = hermite.hermgauss(deg)
#        print sqrt(2)*beta_std * xs + beta_mean

        bs = sqrt(2)*beta_std * xs + beta_mean
        choose_index = (bs>0)
        bs = bs[choose_index]
        ws = ws[choose_index]
        factor = sqrt(pi); #sum(ws)
#        print factor
        print 'mean = %.5f'% (dot(mean_f(bs), ws) / factor)
        print 'std = %.5f' %sqrt( (dot(var_f(bs), ws)) / factor)
    
    
    
def visualizeHermitePtsWts(deg=9, delta =1e-1):
    beta_mean = .0902
    beta_std  = .0562   
    Tf = 50.;
    alpha_applied = .0;
    file_name = os.path.join(OUML_RESULTS_DIR,
                            'OU_Xs.a=%.3f_N=%d.npy'%(alpha_applied,
                                                     num_samples));
    from numpy import load, save
    trajectoryBank = load(file_name)
    
    n_thin = int(delta / dt);
    N_f    = int(Tf/dt)
    idx    = 3 #!!! 
    ts, Xs = trajectoryBank[:N_f,0], trajectoryBank[:N_f,idx]
    ts     = ts[::n_thin];
    Xs     = Xs[::n_thin];
    
    Xn = Xs[1:]
    Xm = Xs[:-1]
    N = len(Xn)
    def mu_estimate(beta):
        exp_beta_delta = exp(-beta*delta);
        mu_hat = sum( Xn - exp_beta_delta*Xm - (alpha_applied / beta)* (1 - exp_beta_delta) ) / \
                     (N * (1 - exp_beta_delta))
        return mu_hat;
        
    def sigma_estimate(beta, mu):
        square_term = (Xn - \
                       exp(-beta*delta)* Xm - \
                       (alpha_applied/beta + mu)*(1-exp(-beta*delta) ) ) **2
        sigma_hat = sqrt( 2.* sum (square_term) * beta / \
                           N / (1-exp(-2*beta*delta))) 
    
        return sigma_hat

    def _prior(beta):
        const_term = 1./ (beta_std * sqrt(2.*pi));
        exp_term =  exp(- .5*(beta - beta_mean) * (beta - beta_mean) / \
                             (beta_std*beta_std) );
        return const_term * exp_term;
    beta_a, beta_b  = .001,  beta_mean + 2.*beta_std;
    norm_factor = 0.921014901458; #romberg(_prior, beta_a, beta_b, tol=1e-6);
#    print norm_factor
    def prior(beta):
        const_term = 1./ (beta_std * sqrt(2.*pi)) / norm_factor;
        exp_term =  exp(- .5*(beta - beta_mean) * (beta - beta_mean) / \
                             (beta_std*beta_std) );
        return const_term * exp_term;                      
     
    x_0 = Xs[-1]; print 'xstart, xlast,' ,Xs[0], x_0
    
    forward_delta = 50*delta; #Tf/ 10. #10.*delta
    print 'forward_delta = ', forward_delta
    
    def X_likelihood(x, beta, alpha):
        mu_est = mu_estimate(beta);
        sigma_est = sigma_estimate(beta, mu_est);
        X_std  = sigma_est*sqrt(1.- exp(-2.*beta*forward_delta));
        X_mean = (alpha / beta + mu_est) + \
                     exp(-beta*forward_delta) * (x_0 - alpha / beta - mu_est)
#        print beta, alpha, mu_est, sigma_est
#        print '\t',     forward_delta, exp(-beta*forward_delta), X_std, x_0, X_mean  

        const_term = 1./ (X_std * sqrt(2.*pi));
        exp_term =  exp(- .5*(x - X_mean)*(x - X_mean) / \
                             (X_std*X_std));
        return const_term * exp_term;

    ys, ws = hermite.hermgauss(deg)
    bs = sqrt(2)*beta_std * ys + beta_mean    
    choose_index = (bs>0)
    first_idx_above_zero = len(bs[bs<=0]);
    bs = bs[choose_index]
    b_ws = ws[choose_index]
    print b_ws
    bs_xs_nodes_dict = {}
    for idx, b in enumerate(bs):
        mu_est = mu_estimate(b);
        sigma_est = sigma_estimate(b, mu_est);
        X_std  = sigma_est*sqrt(1.- exp(-2.*b*forward_delta));
        X_mean = (alpha / b + mu_est) + \
            exp(-b*forward_delta) * (x_0 - alpha / b - mu_est)
        
        x_deg = 0;
        lidx = idx + first_idx_above_zero;
        if (lidx < deg / 2):
            x_deg = 1 + 2*lidx
        else:
            x_deg = 2*(deg-lidx) - 1
        ys, x_ws = hermite.hermgauss(x_deg)
        xs = sqrt(2)*X_std * ys + X_mean;
        bs_xs_nodes_dict[b] = (xs, x_ws);
#        print lidx, b, x_deg, len(xs)
        
    ### done writing out the Data Struct:
    figure()
    for b, bw in zip(bs, b_ws ):
        xs, x_ws = bs_xs_nodes_dict[b][:]
        for x, xw in zip(xs, x_ws):
#            plot(x,b, 'k.', markersize = 2100*log(1+xw*bw))
            plot(x,b, 'k.', markersize = 4)
    ylim((0, amax(bs)+.1))
    title(r'deg of $\beta$ = %d'%deg)
    filename = os.path.join(FIGS_DIR, 'nodes_spread.pdf');
    print filename
    savefig(filename);
    
    figure()
    for b, bw in zip(bs, b_ws ):
        xs, x_ws = bs_xs_nodes_dict[b][:]
        for x, xw in zip(xs, x_ws):
#            print b, x, xw*bw
            plot(x,b, 'k.', markersize = int(100*xw*bw))
    ylim((0, amax(bs)+.1))
    title(r'deg of $\beta$ = %d'%deg)
    filename = os.path.join(FIGS_DIR, 'weighted_nodes_spread.pdf');
    print filename
    savefig(filename);
    
    
    
            

def calculateMI(num_samples=10, delta = 1e-1):
    ''' Calculate the integral of the likelihood of the next transition 
        wrt to the prior of \beta '''
    beta_mean = .0902
    beta_std  = .0562   
#    beta_std  = .0562*.1#.0562
    Tf = 50.;
    alpha_applied = .0;
    file_name = os.path.join(OUML_RESULTS_DIR,
                            'OU_Xs.a=%.3f_N=%d.npy'%(alpha_applied,
                                                     num_samples));
    from numpy import load, save
    trajectoryBank = load(file_name)
    
    n_thin = int(delta / dt);
    N_f    = int(Tf/dt)
    idx    = 3 #!!! 
    ts, Xs = trajectoryBank[:N_f,0], trajectoryBank[:N_f,idx]
    ts     = ts[::n_thin];
    Xs     = Xs[::n_thin];
    
    Xn = Xs[1:]
    Xm = Xs[:-1]
    N = len(Xn)
    def mu_estimate(beta):
        exp_beta_delta = exp(-beta*delta);
        mu_hat = sum( Xn - exp_beta_delta*Xm - (alpha_applied / beta)* (1 - exp_beta_delta) ) / \
                     (N * (1 - exp_beta_delta))
        return mu_hat;
        
    def sigma_estimate(beta, mu):
        square_term = (Xn - \
                       exp(-beta*delta)* Xm - \
                       (alpha_applied/beta + mu)*(1-exp(-beta*delta) ) ) **2
        sigma_hat = sqrt( 2.* sum (square_term) * beta / \
                           N / (1-exp(-2*beta*delta))) 
    
        return sigma_hat

    x_0 = Xs[-1]; print 'xstart, xlast,' ,Xs[0], x_0
    
    forward_delta = 50*delta; #Tf/ 10. #10.*delta
    print 'forward_delta = ', forward_delta
    
    def X_likelihood(x, beta, alpha):
        mu_est = mu_estimate(beta);
        sigma_est = sigma_estimate(beta, mu_est);
        X_std  = sigma_est*sqrt(1.- exp(-2.*beta*forward_delta));
        X_mean = (alpha / beta + mu_est) + \
                     exp(-beta*forward_delta) * (x_0 - alpha / beta - mu_est)
#        print beta, alpha, mu_est, sigma_est
#        print '\t',     forward_delta, exp(-beta*forward_delta), X_std, x_0, X_mean  

        const_term = 1./ (X_std * sqrt(2.*pi));
        exp_term =  exp(- .5*(x - X_mean)*(x - X_mean) / \
                             (X_std*X_std));
        return const_term * exp_term;

    proposed_alphas = linspace(-.25,.25, 3);
   
    #The GH Procedure:
    def getQuadSchema(b_deg, alpha, base_x_deg=1):
        ys, ws = hermite.hermgauss(b_deg)
        bs = sqrt(2)*beta_std * ys + beta_mean    
        choose_index = (bs>0)
        first_idx_above_zero = len(bs[bs<=0]);
        bs = bs[choose_index]
        b_ws = ws[choose_index]
        norm_factor =  sum(b_ws)#sqrt(pi) #sum(b_ws)#
#        print b_ws
        bs_xs_nodes_dict = {}
        for idx, b in enumerate(bs):
            mu_est = mu_estimate(b);
            sigma_est = sigma_estimate(b, mu_est);
            X_std  = sigma_est*sqrt(1.- exp(-2.*b*forward_delta));
            X_mean = (alpha / b + mu_est) + \
                exp(-b*forward_delta) * (x_0 - alpha / b - mu_est)
            
            x_deg = 0;
            lidx = idx + first_idx_above_zero;
            if (lidx < b_deg / 2):
                x_deg = base_x_deg + 2*lidx
            else:
                x_deg = base_x_deg+ 2*(b_deg-lidx-1)
            ys, x_ws = hermite.hermgauss(x_deg)
            xs = sqrt(2)*X_std * ys + X_mean;
            bs_xs_nodes_dict[b] = (xs, x_ws);
#            print lidx, b, x_deg, len(xs)
        return bs, b_ws, bs_xs_nodes_dict, norm_factor
    
#    #Visualize Quad Scheme:
#    b_deg = 43
#    for alpha in proposed_alphas:
#        bs, b_ws, bs_xs_nodes_dict,norm_factor = getQuadSchema(b_deg=b_deg,
#                                                                alpha=alpha);
#        figure()
#        subplot(211)
#        for b, bw in zip(bs, b_ws ):
#            xs, x_ws = bs_xs_nodes_dict[b][:]
#            for x, xw in zip(xs, x_ws):
#    #            plot(x,b, 'k.', markersize = 2100*log(1+xw*bw))
#                plot(x,b, 'k.', markersize = 4)
#        ylim((0, amax(bs)+.1))
#        title(r'deg of $\beta$ = %d, alpha =%.2f'%(b_deg, alpha))
#        print bs[0:2]
#        subplot(212)
#        for b, bw in zip(bs, b_ws ):
#            xs, x_ws = bs_xs_nodes_dict[b][:]
#            for x, xw in zip(xs, x_ws):
#    #            print b, x, xw*bw
#                plot(x,b, 'k.', markersize = int(100*xw*bw))
#        ylim((0, amax(bs)+.1))
#    return 
                                                                
    
    ######Exploring the marginal of x (the normalizing factor);
#    b_deg = 11 #13
#    num_xs = 128#128
#    for b_deg, num_xs in zip([9],#    for num_xs in [32, 256]:
#                             [64]):
#        figure()
#    #    x_0 +=1.0 
#        for idx, alpha in enumerate(proposed_alphas[0::]):
#            bs, b_ws, bs_xs_nodes_dict,norm_factor = getQuadSchema(b_deg=b_deg,
#                                                                    alpha=alpha)
#            
#            mu_est = mu_estimate(beta_mean);
#            xguess_mean = (alpha / beta_mean + mu_estimate(beta_mean)) + \
#                            exp(-beta_mean*forward_delta) * (x_0 - alpha / beta_mean - mu_est)
#            xs = linspace(xguess_mean - 3,
#                          xguess_mean + 3, num_xs);
#            pxs = empty_like(xs);
#            for idx, x in enumerate(xs):
#                dpxs = array([px_integrand(b, x, alpha) for b in bs]);
#                pxs[idx] = dot(dpxs, b_ws) / norm_factor
#            
#    #        dpxs = px_integrand(bs, x, alpha)
#    #        pxs = array( [dot(px_integrand(bs, x, alpha), b_ws) / norm_factor for x in xs ])
#            plot(xs, pxs, label='alpha=%.2f'%alpha);
#            print 'alpha=%.2f'%alpha, ': px integral = ' , sum(pxs * (xs[2]-xs[1]))
#            title(r'$\Delta_f = %.1f$ #xs = %d, b_deg = %d'%(forward_delta,
#                                                             num_xs,
#                                                             b_deg));
#            
#        legend()
#        stem(Xs[::20], ones_like(Xs[::20]));
#        stem([x_0], [2], 'r')
#        xlim((-62,-58));
#        file_name = os.path.join(FIGS_DIR,
#                            'x_marginal_gauss_hermite_bx=%d_%d.pdf'%(b_deg, num_xs))
#        print 'saving to ', file_name
#        savefig(file_name)
#    return
    ##################################
    def prior(beta):
        norm_factor = 0.921014901458; #romberg(_prior, beta_a, beta_b, tol=1e-6);
        const_term = 1./ (beta_std * sqrt(2.*pi) ) / norm_factor;
        exp_term =  exp(- .5*(beta - beta_mean) * (beta - beta_mean) / \
                             (beta_std*beta_std) );
        return const_term * exp_term;                      
    def px_integrand(beta, x, alpha):
        return X_likelihood(x, beta, alpha) * prior(beta);
    
    from scipy.integrate import quad
    def MI_gauss_hermite_integrand(b, x, alpha,
                                    gh_bs, gh_b_ws, gh_norm_factor):
        '''this calculates the MI integrand for a fixed x, b'''
        x_likelihood = X_likelihood(x, b, alpha)
        if x_likelihood < 1e-8:
            return .0;
        
        dpxs = array([X_likelihood(x, b, alpha) for b in gh_bs]);
        bayes_factor =  dot(dpxs, gh_b_ws) / gh_norm_factor
        
#        beta_a, beta_b  = .001,  beta_mean + 2.*beta_std;
#        bayes_factor_quad, err = quad(px_integrand, beta_a,beta_b , args=(x, alpha),
#                                      epsabs = 1e-1);
##        print 'x:%.3f, %.4f'%(x, (bayes_factor - bayes_factor_quad));
#        bayes_factor = bayes_factor_quad                        
                                
                                
        return log( x_likelihood / bayes_factor );
    
#    def MI(alpha, b_deg = 9, base_x_deg = 1):
#        bs, b_ws, bs_xs_nodes_dict,norm_factor = getQuadSchema(b_deg=b_deg,
#                                                                alpha=alpha,
#                                                                base_x_deg=base_x_deg);
#        
#        quad_val = .0;
#        for bi, b_w in zip(bs,
#                           b_ws):
#            xs, x_ws = bs_xs_nodes_dict[bi][:];
#            for xj, x_w in zip(xs,
#                               x_ws):
#                f_ij = MI_gauss_hermite_integrand(bi, xj,
#                                                   alpha,
#                                                    bs, b_ws, norm_factor);
##                SANITY CHECKS:
##                f_ij = (bi-beta_mean)**2                                    
##                f_ij = bi
#                quad_val += b_w*x_w*f_ij
#        quad_val /= ( sqrt(pi)*norm_factor )
#        return quad_val
    
    def MI(alpha, b_deg = 9, base_x_deg = 5, quad_epsabs = 1e-1):
        ys, ws = hermite.hermgauss(b_deg)
        bs = sqrt(2)*beta_std * ys + beta_mean    
        choose_index = (bs>0)
        first_idx_above_zero = len(bs[bs<=0]);
        bs = bs[choose_index]
        b_ws = ws[choose_index]
        norm_factor =  sum(b_ws)#sqrt(pi) #sum(b_ws)#
        
        ys, x_ws = hermite.hermgauss(base_x_deg)
        def MI_b_integrand(b, alpha):
            mu_est = mu_estimate(b);
            sigma_est = sigma_estimate(b, mu_est);
            X_std  = sigma_est*sqrt(1.- exp(-2.*b*forward_delta));
            X_mean = (alpha / b + mu_est) + \
                        exp(-b*forward_delta) * (x_0 - alpha / b - mu_est)
        
            xs = sqrt(2)*X_std * ys + X_mean;
            
            dxs =  array([MI_gauss_hermite_integrand(b, xj,
                                                     alpha,
                                                     bs, b_ws, norm_factor) for xj in xs]);
            
            return dot(dxs, x_ws) / sqrt(pi) * prior(b)
            
        beta_a, beta_b  = .001,  beta_mean + 2.*beta_std;
        quad_val, err = quad(MI_b_integrand, beta_a, beta_b,
                             args=(alpha),
                             epsabs = quad_epsabs)
           
        #NOTE: I don't know why dividing by \sqrt(pi) here is necessary!!!
        return quad_val / sqrt(pi) 
    
#    print 'dI_dxdbeta = ', MI_xb_integrand(beta_mean, x_0, .0)
#    print 'xa, xb' ,[calculate_xa_xb(a) for a in proposed_alphas]
#    proposed_alphas = proposed_alphas[0:1]
#    xabs = [calculate_xa_xb(a) for a in proposed_alphas]
#    print xabs
#    print 'dI_dx lb = ', [MI_x_integrand(xa, a) for (xa, xb), a in zip(xabs,
#                                                                       proposed_alphas)]
#    print 'dI_dx ub = ', [MI_x_integrand(xb, a) for (xa, xb), a in zip(xabs,
#                                                                        proposed_alphas)]
    base_x_deg = 1
    for b_deg in [7,13]: 
#    b_deg = 9;
#    for base_x_deg in [1, 3, 7, 11]:
        start = time.clock()
        MIs =  array([MI(a, b_deg = b_deg,
                          base_x_deg=base_x_deg) for a in proposed_alphas[0::2] ])
        end = time.clock();
        print 'Total calculation in %f s'%(end-start)
#    print proposed_alphas
        mi_string = ['%.4f'%mi for mi in MIs]
        print b_deg,'&', base_x_deg, '&', mi_string, r'\\';
#    save(os.path.join(RESULTS_DIR,
#                      'alpha_MIs_forward_delta'), c_[proposed_alphas, MIs])
#    
#    figure()
#    plot(proposed_alphas, MIs)
#    ylim(( amin(MIs) -.1,
#           amax(MIs) -.1 ))
#    filename = os.path.join(FIGS_DIR,
#                         'alpha_MIs.pdf')
#    print filename
#    savefig(filename)
    
#    figure(); plot(xs, Lxs); 
#    import time
#    start = time.clock()
#    int_quad, abs_err = quad(px_integrand, beta_a, beta_b,
#                              args=(x_0, alpha))
#    end = time.clock(); print 'time;', end-start
#    print int_quad


#    start = time.clock()
#    int_quadrature, abs_err = quadrature(integrand, a, b, args= ([alpha]),
#                                         vec_func=False)
#    end = time.clock(); print 'time;', end-start
#    
#    print int_quadrature
#    
#    start = time.clock();
#    int_romberg = romberg(integrand, a, b, args= ([alpha]))
#    end = time.clock(); print 'time;', end-start
#    print int_romberg
    
    
#The main function pattern in Python:
if __name__ == '__main__':
    from pylab import *
    alpha = 0.0 / tau;
    num_samples = 10;
    
    #Estimate params:
#    start = time.clock();
#    estimateBootstrap(alpha       = alpha,
#                      delta       = 1e-1,
#                      num_samples = num_samples)
    
#    hermiteSandpile()
#    visualizeHermitePtsWts(deg = 11) 

    calculateMI()
    show()