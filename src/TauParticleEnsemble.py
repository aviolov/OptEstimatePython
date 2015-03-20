'''
Created on Feb 13, 2015

@author: alex
'''
from __future__ import division
from numpy import *
from numpy.random import multinomial, normal


FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/TauParticleEnsemble'

import os
for D in [FIGS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
        
''' The Tau chars particles '''
class TauParticleEnsemble():
    def __init__(self, Nparticles):
        self.taus = empty(Nparticles)
        self.weights = ones(Nparticles)/Nparticles
        
        self._a = 0.98
        self._h = sqrt(1-0.98**2)
        
        self.resample_threshold = 0.5;
        
    def setLocationsUsingRange(self, lb, ub):
        ''' log-uniform between lb and ub'''
        self.taus = exp(linspace(log(lb), log(ub), len(self.taus)));
        
    def ensembleMeanStdString(self):
        return '%.2f pm %.2f'%(self.ensembleMean(), sqrt(self.ensembleVar()));
    def ensembleLogMeanStdString(self):
        m, xi = self.ensembleLogMeanVar();
        return '%.2f pm %.2f'%(m, sqrt(xi));
        
    def ensembleMean(self):
        return dot(self.taus, self.weights);
    def ensembleVar(self):
        return dot(self.taus*self.taus, self. weights) - self.ensembleMean()**2;
        
    def ensembleLogMeanVar(self):
        log_taus = log(self.taus)
        m = dot(log_taus, self.weights);
        Xi = dot(log_taus*log_taus, self. weights) - m*m; 
        return m, Xi;
    
    def copyState(self):
        ''' returns copies of locations, weights'''
        return copy(self.taus), copy(self.weights)    
    
    def updateWeights(self, incremental_multipliers, allow_resampling = True):
        ' w_i -> w_i \cdot l(tau_i)'
        self.weights = self.weights*incremental_multipliers;
        self.weights /= sum(self.weights)
        if allow_resampling:
            self.resampleOptionally()
    
    def resampleOptionally(self):
        N = len(self.taus); 
        effective_dimension = 1.0 / dot(self.weights, self.weights)

        if effective_dimension > self.resample_threshold*N:
            return;
        print 'RESAMPLING'        
        
        old_taus_chosen = multinomial(N, self.weights);
        
        new_taus = array([]);
        
        mu, Xi = self.ensembleLogMeanVar();
        Xi = self._h * sqrt( Xi );
        
        'MAIN LOOP:'
        'Remember we are working with log-taus'
        for idx, Nchosen in enumerate(old_taus_chosen):
            if Nchosen == 0:
                continue
            'get the '
            log_tau_i = log(self.taus[idx]);
            mu_i = self._a*log_tau_i + (1-self._a)*mu;
            
            new_taus_i = normal(loc=mu_i, scale=Xi, size=Nchosen)
            
            new_taus = r_[ new_taus, new_taus_i]
        
        'remember to exponentiate the draws:'
        self.taus = sort( exp(new_taus)) ;
        self.weights = ones_like(self.taus)/self.Ntaus();
        #//End Method
    
    def Ntaus(self):
        return len(self.taus);
                
   
def driverBasicParticleManipulations():         
    pE = TauParticleEnsemble(4);
    pE.setLocationsUsingRange(.25, 4);
    
    print pE.taus
    print pE.weights
    
    pE.updateWeights([ 1,  1,  1, 7]);
    print pE.weights
    
    pE.resampleOptionally()
    print pE.taus
    

def filterPoissonianRate(ts, pE):
    
    meanvar = [];
    for tdx, t in enumerate(ts):
        
        taus = pE.taus;
        
        ls = exp(-t/taus)/taus;
        
        pE.updateWeights(ls);
        
        meanvar.append([pE.ensembleMean(), pE.ensembleVar()]);
    
    return array(meanvar);

def driverPoissonianRateFilter():
    pE = TauParticleEnsemble(50);
    pE.setLocationsUsingRange(.25, 4);
    
    seed(110228)
    
    for Nhits in [1e3]: # [10,100,1000,10000, 1e5, 1e6]:
#    Nhits = 1000
    
        ts     =  exponential(scale = 1., size=Nhits)
        mean_t =  mean(ts);
                
    
#    print ts
        print mean_t
        
        mv = filterPoissonianRate(ts, pE);
        
        figure()
        subplot(211);
        tks = arange(0, Nhits);
        mean_tau = mv[:,0];
        std_tau = sqrt(mv[:,1])
        color = 'r';
        
        plot( tks, mean_tau, color + '-');
        plot( tks, mean_tau  + 2*std_tau, color + '--');
        plot( tks, mean_tau  - 2*std_tau, color + '--');
        
        plot(tks, mean_t*ones_like(tks), 'k-')
        ylim([0,4])
        ylabel(r'$\rho( \tau ) $')
        title('Plot of Poissonian (Inverse) Rate Parameter Filtering')
        
        subplot(212);
        
        ids = int32(  (floor( arange(0, sqrt(Nhits))**2 )) )
        
        plot(tks[ids], std_tau[ids], 'x-');
        title('Ensemble Std-Dev Plot')
        ylabel(r'std-dev($\tau)$')
        xlabel('k'); 
        
        print '%.4f, %.4f, %.4f'%(mean_t, mean_tau[-1], std_tau[-1])
        
        figname = os.path.join(FIGS_DIR, 'poisson_rate_filtering.pdf')
        savefig(figname);
        print figname
        

if __name__ == '__main__':
    from pylab import *
    
    'Call some elementary functions of the Particle Ensemble'
#    driverBasicParticleManipulations()

    'apply the filter to estimate the rate of a POISSON PRocess'
    driverPoissonianRateFilter()
    
    show()