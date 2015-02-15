'''
Created on Feb 13, 2015

@author: alex
'''
from __future__ import division
from numpy import *
from numpy.random import multinomial


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
            
            new_taus_i = normal(loc=mu_i  , scale=Xi, size=Nchosen)
            
            new_taus = r_[ new_taus, new_taus_i]
        
        self.taus = sort( exp(new_taus)) ;
        #//End Method
            
            
            
        
            
    
if __name__ == '__main__':
    from pylab import *
    pE = TauParticleEnsemble(4);
    pE.setLocationsUsingRange(.25, 4);
    
    print pE.taus
    print pE.weights
    
    pE.updateWeights([ 1,  1,  1, 7]);
    print pE.weights
    
    pE.resampleOptionally()
    print pE.taus
    
    show()