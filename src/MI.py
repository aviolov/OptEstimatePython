# -*- coding:utf-8 -*-
"""
Created on Nov 11, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand
from OUML import RESULTS_DIR as OUML_RESULTS_DIR
from OUML import estimateParamsBeta, dt

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/MI/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/MI/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

xlabel_fontsize = 32

#STRUCTURAL PARAMATER VALUES - HELD FIXED
tau = 20. #ms;
beta = 1/tau;
mu = -60.0; 
sigma = .1;
 
def estimateBootstrap( delta = 1e-1,
                       alpha = .0,
                       num_samples=10 ):
    #Load all the simulated trajectories
    Tf = 50.;
    
    file_name = os.path.join(OUML_RESULTS_DIR,
                              'OU_Xs.a=%.3f_N=%d.npy'%(alpha,
                                                       num_samples));
    from numpy import load
    trajectoryBank = load(file_name)
    
    #Select an arbitrary trajectory: (here the 2nd)
    figure(); hold(True);
    n_thin = int(delta / dt); print n_thin
    N_bootstrap = 8;
    N_f = int(Tf/dt)
     
    for idx in xrange(num_samples):
        ts, Xs = trajectoryBank[:N_f,0], trajectoryBank[:N_f,idx]
        ts = ts[::n_thin];
        Xs = Xs[::n_thin];
        try:
            est_params = estimateParamsBeta(Xs, delta, alpha)
        except:
            continue
        
        beta_mean = est_params[1];  
        betas = zeros(N_bootstrap);
        for start in xrange(N_bootstrap):
            delta_bootstrap = delta*N_bootstrap
            ts_ss = ts[start::N_bootstrap]
            Xs_ss = Xs[start::N_bootstrap]
            #Obtain estimator
            try:
                est_params = estimateParamsBeta(Xs_ss, delta_bootstrap)            
                betas[start] = est_params[1];
            except ValueError:
                betas[start] = NaN
                  
            
        betas = betas[betas != NaN]
        print '%d: beta bootstrap:'\
              ' mean, std = %.4f, %.4f, or %.4f, %.4f'%(idx,
                                                        beta_mean,
                                                       norm(betas-beta_mean),
                                                       mean(betas),
                                                       std(betas))

def calculateMI(Xs, alphas_applied, delta,
                  beta_mean, beta_std,
                    proposed_alphas, forward_delta):
    ''' Calculate the integral of the likelihood of the next transition 
        wrt to the prior of \beta '''
  
    Xn = Xs[1:]
    Xm = Xs[:-1]
    alphas_applied = alphas_applied[:-1]
    N = len(Xn)
    def mu_estimate(beta):
        exp_beta_delta = exp(-beta*delta);
        mu_hat = sum( Xn - exp_beta_delta*Xm - (alphas_applied / beta)* (1 - exp_beta_delta) ) / \
                     (N * (1 - exp_beta_delta))
        return mu_hat;
        
    def sigma_estimate(beta, mu):
        square_term = (Xn - \
                       exp(-beta*delta)* Xm - \
                       (alphas_applied/beta + mu)*(1-exp(-beta*delta) ) ) **2
        sigma_hat = sqrt( 2.* sum (square_term) * beta / \
                           N / (1-exp(-2*beta*delta))) 
    
        return sigma_hat

    def _prior(beta):
        const_term = 1./ (beta_std * sqrt(2.*pi));
        exp_term =  exp(- .5*(beta - beta_mean) * (beta - beta_mean) / \
                             (beta_std*beta_std) );
        return const_term * exp_term;
    from scipy.integrate import quad, romberg, quadrature
    beta_a, beta_b  = .001,  beta_mean + 2.*beta_std;
    norm_factor = romberg(_prior, beta_a, beta_b, tol=1e-4);
    def prior(beta):
        const_term = 1./ (beta_std * sqrt(2.*pi)) / norm_factor;
        exp_term =  exp(- .5*(beta - beta_mean) * (beta - beta_mean) / \
                             (beta_std*beta_std) );
        return const_term * exp_term;                      
    
#    bs = linspace(0.01, beta_mean + 2*beta_std, 31);
#    p_bs = prior(bs);
#    ms = array([mu_estimate(b) for b in bs]);
#    ss = array([sigma_estimate(b,m) for b,m in zip(bs,ms)]);
#    figure()
#    subplot(311)
#    plot(bs, p_bs); title(r'prior: ($\rho(\beta)$)'); xlabel(r'$\beta$')
#    subplot(312)
#    plot(bs, ms); title (r'$\hat \mu (\beta)$');
#    subplot(313);
#    plot(bs, ss); title(r'$\hat \sigma (\beta)$'); ylim((amax(ss)-.02, amax(ss)+.02))
#    file_name = os.path.join(FIGS_DIR,
#                             'prior_example.pdf')
#    print file_name
#    savefig(file_name)
#    
#    figure()
#    plot(ts[:-1], Xm); xlabel(r'$t$'); ylabel(r'$X_t$');
#    file_name = os.path.join(FIGS_DIR,
#                             'prior_example_path.pdf')
#    print file_name
#    savefig(file_name)
#    print 'early return'
#    return;
    
    x_0 = Xs[-1]; 
#    print 'xstart, xlast,' ,Xs[0], x_0

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

    xs = linspace(amin(Xs)-1., amax(Xs)+1., 256)
#######Exploring the conditional distribution of x given alpha, beta
#    figure() 
#    for idx, alpha in enumerate([-.1,0,.1]):
#        pxs = X_likelihood(xs, beta_mean, alpha)
#        subplot(3,1,idx+1)
#        plot(xs, pxs); title('alpha = %.2f'%alpha)   
    def px_integrand(beta, x, alpha):
        return X_likelihood(x, beta, alpha) * prior(beta);
    

    ######Exploring the marginal of x (the normalizing factor);
#    figure()
#    x_0 +=1.0 
#    for idx, alpha in enumerate(proposed_alphas):
#        
#        mu_est = mu_estimate(beta_mean);
#        xguess_mean = (alpha / beta_mean + mu_estimate(beta_mean)) + \
#                        exp(-beta_mean*forward_delta) * (x_0 - alpha / beta_mean - mu_est)
#                     
#        xs = linspace(xguess_mean - 3,
#                       xguess_mean + 3,
#                        128);
#        pxs = array([quad(px_integrand, beta_a, beta_b,
#                          args=(x, alpha))[0] for x in xs ])
#        plot(xs, pxs, label='alpha=%.2f'%alpha);
#        print 'alpha=%.2f'%alpha, ': px integral = ' , sum(pxs * (xs[2]-xs[1]))
#        Pxs = cumsum(pxs); #print 'marginal calcs: x_lb, x_ub: Total mass calculated = ', Pxs[-1]
#        Pxs /= Pxs[-1]
#        print alpha, xs[Pxs<.025][-1], xs[Pxs>.975][0], xs[Pxs>.975][0] - xs[Pxs<.025][-1]
#    
#    legend()
#    stem(Xs[::20], ones_like(Xs[::20]));
#    stem([x_0], [2], 'r')
#    xlim((-62,-58));
#    title(r'$\Delta_f = %.1f$'%forward_delta);
#    file_name = os.path.join(FIGS_DIR,
#                            'x_marginal_example_x0_shifted_right.pdf')
#    print 'saving to ', file_name
#    savefig(file_name)
#    return
    
    def calculate_xa_xb(alpha):
        '''this calculates where the bulk of the forward density lies, given a 
        value of alpha and return the .95 quantiles, with a hard-coded granularity 
        of about dx = .1''' 
        mu_est = mu_estimate(beta_mean);
        xguess_mean = (alpha / beta_mean + mu_estimate(beta_mean)) + \
                        exp(-beta_mean*forward_delta) * (x_0 - alpha / beta_mean - mu_est)
                     
        xs = linspace(xguess_mean - 10,
                       xguess_mean + 10,  200);
        pxs = array([quad(px_integrand, beta_a, beta_b,
                      args=(x, alpha),
                      epsabs = 1e-3)[0] for x in xs ])
        Pxs = cumsum(pxs); #print 'marginal calcs: x_lb, x_ub: Total mass calculated = ', Pxs[-1]
        Pxs /= Pxs[-1]
        return xs[Pxs<.025][-1], xs[Pxs>.975][0]
    
    def MI_xb_integrand(beta, x, alpha):
        '''this calculates the MI integrand for a fixed x, b'''
        x_likelihood = X_likelihood(x, beta, alpha)
        if x_likelihood < 1e-8:
            return .0;
        norm_factor, err = quad(px_integrand, beta_a,beta_b , args=(x, alpha),
                                epsabs = 1e-1);
        beta_prior = prior(beta)
#        with warnings.catch_warnings(record=True) as w:
        ret_val =   log( x_likelihood / norm_factor ) * x_likelihood * beta_prior;
#            warnings.simplefilter("always")
#            if len(w)>0:
#                print 'norm_fact = %3f, x_like = %3f'%(norm_factor,
#                                                   x_likelihood)
        return ret_val 
        
    def MI_x_integrand(x, alpha):
        '''this integrates the MI integrand over beta for fixed x
        '''
        quad_val, err =  quad(MI_xb_integrand, beta_a, beta_b,
                               args = (x, alpha),
                                epsabs = 1e-1)
#        print '\t', x, quad_val
        return quad_val
    
    def MI(alpha):
        x_a , x_b = calculate_xa_xb(alpha);
#        print '%.3f'%alpha, ':%.3f, %.3f'%(x_a, x_b)
        assert(x_a < x_b)
#        quad_val, err = romberg(MI_x_integrand,
#                                 x_a, x_b, args =([alpha]))
        quad_val = romberg(MI_x_integrand,
                           x_a, x_b, args =([alpha]),
                           tol= 2e-2)
        return quad_val
    
#    print 'dI_dxdbeta = ', MI_xb_integrand(beta_mean, x_0, .0)
#    print 'xa, xb' ,[calculate_xa_xb(a) for a in proposed_alphas]
#    proposed_alphas = proposed_alphas[0:1]
#    xabs = [calculate_xa_xb(a) for a in proposed_alphas]
#    print xabs
#    print 'dI_dx lb = ', [MI_x_integrand(xa, a) for (xa, xb), a in zip(xabs,
#                                                                       proposed_alphas)]
#    print 'dI_dx ub = ', [MI_x_integrand(xb, a) for (xa, xb), a in zip(xabs,
#                                                                        proposed_alphas)]
    
    return array([MI(a) for a in proposed_alphas[:] ])
   

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
    

def MIDriver(num_samples=10, delta = 1e-1):
    beta_mean = .1502#.0902
    beta_std  = .0562 *1.5#.0562
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
    alphas_applied = zeros_like(Xs);
    
    forward_delta = 50*delta; #Tf/ 10. #10.*delta
#    print 'forward_delta = ', forward_delta
    
#    proposed_alphas = linspace(-.25,.25, 3);
    proposed_alphas = [-.25, .25];

#    forward_deltas = array([1, 10, 25, 50, 100])* delta
    forward_deltas = array([50])* delta
#    forward_deltas = array([1, 10, 25, 50, 100, 200, 300, 400])* delta
#    #MAIN CALL:
#    MIs = []
#    for forward_delta in forward_deltas:
#        MI = calculateMI(Xs, alphas_applied, delta, beta_mean, beta_std, proposed_alphas, forward_delta)
#        MIs.append(MI[0])
##        print(r"%.2f & %.3f \\"%(forward_delta, MI[0]))
#    print list(forward_deltas), MIs
#    plotDsMs(forward_deltas, MIs)
    start = time.clock()
    #MAIN CALL:
    MIs = calculateMI(Xs, alphas_applied, delta,
                       beta_mean, beta_std,
                        proposed_alphas, forward_delta)
    end = time.clock();
    print proposed_alphas
    print MIs;
    print 'Total calculation in %f s'%(end-start)
    
def plotDsMs(ds, Ms):
    figure()
    plot(ds,Ms,
         'o', markersize = 12);
    ylabel(r'$I(\alpha)$', fontsize = xlabel_fontsize)
    xlabel(r'$\Delta_f$', fontsize = xlabel_fontsize)
    file_name = os.path.join(FIGS_DIR, 'forward_deltas_vs_MIs.pdf')
    print file_name
    savefig(file_name)
    
    
    
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
    
    MIDriver()
#    plotDsMs([  0.1 ,  0.5,   1.  ,  2.5  , 5.  , 10.  , 25.],
#            [0.016389533850681978, 0.055767621574534507, 0.0437351800259232, 0.27308813069223303, 0.69945869821900508, 1.3285098511908362, 2.1140491630282576])

    show()