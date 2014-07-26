# -*- coding:utf-8 -*-
"""
Created on Nov 11, 2013

@author: alex
"""
from __future__ import division #always cast integers to floats when dividing

from numpy import *
from numpy.random import randn, rand, seed
#ploting functionality:
from pylab import plot, figure, hold, savefig, show, hist, title, xlabel, ylabel, mpl
from matplotlib.pyplot import legend
from Doublewell_FB_Stationary import xlabel_font_size

#Utility parameters:
RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/Simulator/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/Simulator/'
import os, time

for D in [RESULTS_DIR, FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

def get_ts_dWs(dt, tf):
    #Obtain the increments of a Wiener Process realization
    #returns: ts - the sampled time points of the realization
    #         dWs - the forward incrments of W_t at ts[:-1]
    #the square root of dt:
    sqrt_dt = sqrt(dt)
    #the time points:
    ts = r_[arange(.0, tf, dt), tf];
    #the Gaussian incrments:
    dWs = randn(len(ts)-1)* sqrt_dt
    
    return ts, dWs

def simulateDoublewell(dt=1e-3, Tf=5.0,
                       alphafun= lambda x: 0,
                       X_0=0,
                       A = 3.8,  c = 0.3, sigma = .1,
                       visualize=False):
    #Simulates a path of the OU process using the Millstein scheme:
    #returns: ts - the time points of the simulation
    #         Xs - the simulated values of the process at ts
    
    #Obtain a Wiener realization:
    ts,dWs = get_ts_dWs(dt, Tf)
    #allocate memory for soln:
    Xs = empty_like(ts)
    #ICs:
    Xs[0]  =  X_0; 
    #drift function:
    Ufun = lambda xs: -( 4*xs**3 - A*xs*exp(-1/2*(xs/c)**2 ) / c**2 - 4*xs  )
#    Ufun = lambda xs: .0*xs
    Vfun = lambda xs:    xs**4 + A*exp(-1/2*(xs/c)**2) - 2*xs**2
    
    if visualize:
        xs = linspace(-2,2,101);
        dx = diff(xs[:2])[0]
        fx = Ufun(xs) + alphafun(xs);
        figure(); hold(True)
        subplot(211)
        plot(xs, Vfun(xs));
        V_tilted = Vfun(xs[0]) -(cumsum(fx) - fx[0])*dx 
        plot(xs, V_tilted, 'rx')
        hlines(0,-2,2 )
    #    xlim((-2,2)); ylim((-1,8))
        subplot(212)
        plot(xs, Ufun(xs), 'b.');
        plot(xs, fx, 'r.'); 
        plot((xs[1:]+xs[:-1])/2, -diff(Vfun(xs))/ (dx), 'bx')
        plot((xs[1:]+xs[:-1])/2, -diff(V_tilted)/ (dx), 'rx')
        xlim((-2,2)); ylim((-10,10))
        hlines(0, -2,2)
    
    #evolve in time:
#    print dt, mean(dWs**2)
    for idx,t in enumerate(ts[:-1]):
        x = Xs[idx]
        #the drift:
        f = Ufun(x) + alphafun(x)
        #the volatility:
        g = sigma
                      
        Xs[idx+1] = x + f*dt + g*dWs[idx]  
    
    return ts, Xs, Ufun, Vfun


def DoublewellHarness(amax = 6.):
    alpha_null = lambda x: .0*x;
    alpha_bangbang = lambda x: -amax*sign(x)
    alpha_atan = lambda x: -arctan(5*x) / (pi/ 2.) * amax
    def alpha_paper(x):
        limits = [.1, .35,.8]
        vals   = [0,   2, 4]
        if isscalar(x):
            alpha = amax
            for lim, val in zip(limits, vals):
                if abs(x) < lim:
                    alpha = val;
                    break;
            return -alpha*sign(x)
        #else x is array-like
        alphas = -amax*sign(x)
        for limit, val in zip(limits[::-1],
                                vals[::-1]):
            indxs = (abs(x) < limit) 
        
            alphas[indxs] = -val*sign(x[indxs])
        return alphas
    
    def alpha_trap(x):
        return amax*(-sign(x)*(abs(x)>1.) + sign(x)*(abs(x)<=1.))
    
    x_min,x_max = -5,5 
    xs = linspace(x_min,x_max,100);
    
#    figure(); hold(True)
#    for col, fun in zip(['k', 'b', 'r', 'g'],
#                        [alpha_null, 
#                         alpha_bangbang,
#                          alpha_atan,
#                           alpha_paper]):
#        plot(xs, fun(xs), col+'o')
#        ylim((-amax-1, amax+1))
#    Tf = 100.0    
#    seed_key = randint(2014);
#    
##    #Simulate different controls:
#    seed(seed_key)
#    ts, null_Xs = simulateDoublewell(dt = 1e-2, alphafun=alpha_null, Tf=Tf);
##    
##    seed(seed_key)
##    ts, bangbang_Xs = simulateDoublewell(dt = 1e-2, alphafun=alpha_bangbang, Tf=Tf);
##
#    seed(seed_key)
#    ts, atan_Xs = simulateDoublewell(dt = 1e-2, alphafun=alpha_atan, Tf=Tf);
#
#    seed(seed_key)
#    ts, paper_Xs = simulateDoublewell(dt = 1e-2,
#                                       alphafun=alpha_paper, Tf=Tf);
#                                       
##    seed(seed_key)
##    ts, paper_Xs = simulateDoublewell(dt = 1e-2,
##                                       alphafun=alpha_trap, Tf=Tf);
#    
#    
#    #plot retults:
#    figure()
#    plot(ts, null_Xs, 'k')
##    plot(ts, bangbang_Xs, 'b')
#    plot(ts, atan_Xs, 'r')
#    plot(ts, paper_Xs, 'g')
#    ylim((-1.2, 1.2))
    
    
#    sigmas = array([.1, .75, 1.0, 1.5]) #arange(.05, 1.5, .2)
#    figure(); hold(True)
#    seed_key = 2016;
#    for idx, sigma in enumerate(sigmas):
#        seed(seed_key)
#        ts, atan_Xs = simulateDoublewell(dt = 1e-2,
#                                          alphafun=alpha_atan, Tf=Tf,
#                                          sigma=sigma);
#        subplot(len(sigmas), 1, 1+idx)
#        plot(ts, atan_Xs, label='s=%.2f'%sigma)
#        ylim((-1.5,1.5))
#        legend()

    Tf = 50.0
    sigma = 1.;
    dt = 1e-3;
    
    seed_key = 2014
    seed(seed_key)
    ts, bang_bang_Xs, Ufun, Vfun = simulateDoublewell(dt = dt, X_0 = .5,
                                          alphafun=alpha_bangbang, Tf=Tf,
                                          sigma=sigma);
    seed(seed_key)                                      
    ts, null_Xs, Ufun, Vfun = simulateDoublewell(dt = dt, X_0 = .5,
                                          alphafun=alpha_null, Tf=Tf,
                                          sigma=sigma);
                                          
    xs = linspace(-2,2,50)                                      
#    visualize:
    fig = figure(figsize=(17,8));
    subplots_adjust(hspace = .3,wspace = .25,
                    left=.1, right=1.,
                    top = .9, bottom = .1)
    
#    ax.annotate( '$A=?$', xytext=(-.25, 0.),
#                 xy=(0, 4), arrowprops=dict(arrowstyle='->',
#                                  linewidth = 4),
#                fontsize = xlabel_font_size)
    ax = fig.add_subplot(221)
    ax.annotate( '$A=?$', xytext=(1.,5.),
                 xy=(0, 2), arrowprops=dict(arrowstyle='->',
                                  linewidth = 2),
                fontsize = xlabel_font_size)
    ax = fig.add_subplot(222)
    ax.annotate( '', xytext=(-3.35,5.25),
                     xy=(0, 2),
                     arrowprops=dict(arrowstyle='->',
                                    linewidth = 2))
    for plot_idx in [1,2]:
        ax = fig.add_subplot(2,2,plot_idx)
        ax.plot(xs, Vfun(xs), 'k',
                linewidth = 3)
        ax.set_title('Double Well Potential', fontsize=xlabel_font_size);
        ax.set_xlabel(r'$X$', fontsize = xlabel_font_size)
        ax.annotate ('', (0., 0.),
                         (0., 3.8),
                   arrowprops={'arrowstyle':'<->', 'linewidth':3})
        ax.hlines(.0, -2,2, linestyles='dashed')
    
    for col, plot_idx, afunc in zip(['b', 'r'],
                                    [3,4],
                                    [alpha_null, alpha_bangbang]):
        ax = fig.add_subplot(2,2,plot_idx)
        ax.plot(xs, afunc(xs),
                 col + '+-',
                 linewidth = 3);
        da = .9
        ax.set_ylim((-amax-da, amax+da));
        ax.set_ylabel(r'$\alpha(x)$', fontsize = xlabel_font_size)
        ax.set_xlabel(r'$x$', fontsize = xlabel_font_size)
        
    
    dv = .3
    for plot_idx, col,\
         Xs, dynamics_tag in zip([1,2],
                                 ['b', 'r'],
                                 [null_Xs, bang_bang_Xs],
                                 ['uncontrolled', 'controlled']):
        ax = fig.add_subplot(2,2,plot_idx); ax.hold(True);
        subXs = Xs[::100]
        ax.plot(subXs, Vfun(subXs)+dv, col + '.',
                markersize = 10,
                 label=dynamics_tag);
        ax.legend(prop={'size':xlabel_font_size},
                  loc = 'upper left')
#        arrow( 0., 0., 0, 3.8, fc="k", ec="k",
#                head_width=0.05, head_length=0.1,
#                shape='full' )
        
    
#        ax.set_ylim((-2,2))
#        ax.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
#        ax.set_xlabel(r'$t$', fontsize = xlabel_font_size)
    
#    for plot_idx, Xs in zip([5,6],
#                               [null_Xs, bang_bang_Xs]):
#        ax = fig.add_subplot(3,2,plot_idx)
#        ax.plot(ts[::10], Xs[::10]);
#        ax.set_ylim((-2,2))
#        ax.set_ylabel(r'$X_t$', fontsize = xlabel_font_size)
#        ax.set_xlabel(r'$t$', fontsize = xlabel_font_size)
                                          
    
    fig_name = os.path.join(FIGS_DIR, 'doublewell_example_trajectory.pdf')
    print 'saving to ', fig_name
    save_ret_val = fig.savefig(fig_name, dpi=300)

#The main function pattern in Python:
if __name__ == '__main__':
    from pylab import *
    DoublewellHarness()
    
    show()