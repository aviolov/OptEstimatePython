'''
Created on Oct 29, 2013

@author: alex
'''
from __future__ import division

from numpy import *
from pylab import *
from numpy.polynomial.legendre import * 


def expLegendre():    
    L_deg = 3
    
    xs = arange(-1., 1.0, .2)
    fs = exp(xs)
    leg_cs = legfit(xs, fs, L_deg)
    leg_fs = legval(xs, leg_cs); 
    
         
    taylors = 1 + xs + xs*xs/2 +  xs*xs*xs/6 
    
    plot(xs, fs, 'b', label='true');
    plot(xs, taylors, 'g', label='taylor');
    plot(xs, leg_fs, 'r', label='legendre');
    legend()
    

def  squarewaveLegendre():
    f = lambda x : -1.  + (x>-.5)*1. +(x>.5)*1.
    xs = arange(-1., 1.0, .2)
    fs = f(xs)
    L_deg = 7;
    leg_cs = legfit(xs, fs, L_deg)
    
    leg_fs = legval(xs, leg_cs); 
    
    plot(xs, fs, 'b');
    plot(xs, leg_fs, 'r');
    
def sineLegendre():    
    L_deg = 5
    
    xs = arange(.0, 2*pi*2, .2)
    fs = sin(xs)
    leg_cs = legfit(xs, fs, L_deg)
    leg_fs = legval(xs, leg_cs); 
    print leg_cs
    
#    taylors = 1 - xs*xs/2. +  xs*xs*xs*xs/12. 
    
    plot(xs, fs, 'b');
#    plot(xs, taylors, 'g');
    plot(xs, leg_fs, 'r');

def randCoefs():
    L_deg = 5
    
    xs = arange(.0, 2*pi*2, .2)
    
    leg_cs = rand(L_deg)*exp(-arange(L_deg)*3)
    print leg_cs
    leg_fs = legval(xs, leg_cs); 
    
    plot(xs, leg_fs, 'r');

def piecewiseConst(vals, ts):
    num_vals = len(vals)
    knots = linspace(ts[0], ts[-1], num_vals+1)
    alphas = empty_like(ts);
    for idx, (a,b) in enumerate(zip(knots[:-1],
                                      knots[1:])):
        alphas[(ts>=a) & (ts<=b)] = vals[idx]
    return alphas 
    
def piecewiseConstSandbox():
    Tf = 15.0
    ts = arange(.0, Tf, .2)
    alpha_ws = rand(5)
    print alpha_ws
    
    alphas = piecewiseConst(alpha_ws, ts)
    
    L_deg = 6;
    leg_cs = legfit(ts, alphas, L_deg)
    leg_alphas = legval(ts, leg_cs); 
    
    plot(ts, alphas, 'b')
    plot(ts, leg_alphas, 'b')
    
    




if __name__ == '__main__':
    piecewiseConstSandbox()
    
#    expLegendre()
#    squarewaveLegendre()
#    sineLegendre()
#    randCoefs()
    
    show()