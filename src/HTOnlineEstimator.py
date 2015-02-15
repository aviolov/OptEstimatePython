'''
Created on Feb 13, 2015

@author: alex
'''

from numpy import *
from scipy.optimize.zeros import brentq
from copy import deepcopy
from scipy.optimize.optimize import fminbound
from matplotlib.font_manager import FontProperties
from numpy.random import rand
from scipy import interpolate 

#import ext_fpc
from HitTime_MI_Beta_Estimator import SimulationParams

from AdjointSolver import FPAdjointSolver, calculateOptimalControl
from scipy.interpolate.interpolate import interp1d
from TauParticleEnsemble import TauParticleEnsemble

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/HTOnlineEstimator/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/HTOnlineEstimator'


class Stimulator():
    def getStimulation(self, particleEnsemble, Tf):
        pass    
class MIStimulator(Stimulator):
    def __init__(self,  mu_sigma): 
        self.mu_sigma = mu_sigma;
        
        self.current_ts = array([0,1]) ;
        self.current_a_opt = zeros_like(self.current_ts);
        
        self.ts_aopts_Massif = []
    
    def getStimulation(self, particleEnsemble, Tf): 
        S, fs, ps, alpha_iterations, J_iterations = calculateOptimalControl(particleEnsemble.taus, 
                                        particleEnsemble.weights,
                                        self.mu_sigma,
                                        Tf,
                                        initial_ts_cs= [self.current_ts,
                                                        self.current_a_opt])
        
        a_opt = alpha_iterations[-1];
        
        self.current_ts = S._ts;
        self.current_a_opt = a_opt;
        
        'Archive:'
        self.ts_aopts_Massif.append([S._ts, a_opt])
        
        'Clean up:'
        del S, fs, ps, alpha_iterations, J_iterations 
        
        'Return:'
        return interp1d(self.current_ts, self.current_a_opt,
                        bounds_error = False,
                        fill_value = self.current_a_opt[-1]);
                  
 
class RandomStimulator(Stimulator):
    def __init__(self, alpha_bounds= [-2,2]):          
        self.alpha_bounds = [0.0, alpha_bounds[1]] 
        
        self.alphasMassif = []
    
    def getStimulation(self):
        rand_draw = rand(1); 
        a_current = (self.alpha_bounds[1]-self.alpha_bounds[0])*rand_draw + self.alpha_bounds[0];
        
        self.alphasMassif.append(a_current);
        return lambda ts: a_current*ones_like(ts)
        
        
def driverStimulators(savefig = False):
    
    ts = arange(.0, 2., .05)
    
    mu_sigma = [.0,1.0];
    
    mS = MIStimulator(mu_sigma)
    rS = RandomStimulator(mu_sigma);
    
    
    Tf = 1.0;
    
    pE = TauParticleEnsemble(2);
    pE.setLocationsUsingRange(.5, 2);
    
    mF = mS.getStimulation(pE, Tf); 
    rF = rS.getStimulation()
    
    figure()
    plot(ts, mF(ts), 'rx-', label='optimal')
    hold(True)
    plot(ts, rF(ts),  'gx-', label='random')
    legend()
    
    figure();
    plot(mS.current_ts, mS.current_a_opt, 'rx-')
    

class HTOnlineEstimator():
#    STATIC PARAMS:
    def __init__(self, 
                 simParams,
                 Ntrials = 1, Nexperiments = 1,
                 FPSolver= None,
                 init_range = [.25, 4],
                 Ntaus = 32,
                 alpha_bounds = [-2,2],
                 MIStimulationTime = 10.0):
        
        self.simParams = simParams;
        
        self.Ntrials = Ntrials;
        self.Nexperiments = Nexperiments
        
        self.miEnsemble = TauParticleEnsemble(Ntaus);
        self.miEnsemble.setLocationsUsingRange(init_range[0], init_range[1]);
        self.randEnsemble = deepcopy( self.miEnsemble );
        
        self.alpha_bounds = alpha_bounds;
        mu_sigma = [self.simParams.mu, self.simParams.sigma];
        
        ''' the online stimulators'''
        self.miStimulator   = MIStimulator(mu_sigma)
        self.MI_Stimulation_Time = MIStimulationTime;

        self.randStimulator = RandomStimulator(mu_sigma);
    
        
        self.likelihoodSolver = FPSolver;
        if None == self.likelihoodSolver:
            '''default likelihood solver'''
            taus = self.miEnsemble.taus;
            xmin = FPAdjointSolver.calculate_xmin(self.alpha_bounds ,
                                                  taus, 
                                                  mu_sigma, 
                                                  num_std = 1.0)
            dx = 0.25* FPAdjointSolver.calculate_dx(alpha_bounds,
                                                    taus,
                                               mu_sigma, xmin)
            dt = 0.25* FPAdjointSolver.calculate_dt(alpha_bounds, 
                                               taus,
                                               mu_sigma,
                                               dx, xmin )
            print 'Solver params: xmin, dx, dt', xmin,dx,dt
            
            self.likelihoodSolver = FPAdjointSolver(dx, dt, 1.0, xmin)
        
        ''' the archive massifs that keep track of the experiment'''
        self.miTausMassif   = [ [self.miEnsemble.taus,   self.miEnsemble.weights] ]
        self.randTausMassif = [ [self.randEnsemble.taus, self.randEnsemble.weights]]
    
    
    '''Kernel Routine to run a single hitting time + update the particle weights'''
    def runSingleTrial(self, visualize = False):        
        ''' get the stimulations'''
        miAlphaF = self.miStimulator.getStimulation(self.miEnsemble, self.MI_Stimulation_Time  ); 
        randAlphaF = self.randStimulator.getStimulation();
        
        '''run the HT simulation:'''
        thits = self.simulateSinglePath(miAlphaF, randAlphaF,
                                            visualize_paths = visualize)
        print thits
        
        '''update particle ensembles'''
        for th, alphaF, pE, massif in zip(thits, 
                                  [ miAlphaF,  randAlphaF],
                                  [self.miEnsemble, self.randEnsemble],
                                  [self.miTausMassif, self.randTausMassif]):
        
            tau_likelihood = self.calculateLikelihood(th, alphaF, pE)
            
            pE.updateWeights(tau_likelihood);
            
            print pE.weights
            massif.append( [pE.taus ,
                            pE.weights] );
                            
        'visualize?'                    
        if visualize:
            figure();
            for pdx, massif in zip([1,2],
                                   [self.miTausMassif, self.randTausMassif]):    
                prev_taus = massif[-2][0]
                taus = massif[-1][0]
                prev_weights = massif[-2][0]
                weights = massif[-1][1] 
                
                subplot(2,1,pdx)
                hold(True);
                
                stem(prev_taus, prev_weights, 'b-.', 'markerfacecolor', 'b')
                stem(taus, weights, 'r-.', 'markerfacecolor', 'r')
        
    
    def simulateSinglePath(self,  
                         miAlphaF,
                         randAlphaF,
                         Tmax = 25.0,
                         dt = 0.001, 
                         x_0 = .0,
                         visualize_paths = True):
        ''' we make the judgement call to use the same random numbers PER DRAW
         that we really simulate each HT as a single renewal process, even though
         in practice different stimulations will result in different hitting times and thus
         the driving dWs will desynchronise'''
        
        #ASSIGN STRUCT PARAMS:       
        mu, tauchar, sigma = self.simPs.mu, self.simPs.tau_char, self.simPs.sigma;
 
        #The random numbers:
        sqrt_dt = sqrt(dt);
              
        #the time points:   
        N_ts = Tmax/dt;    
        
        ts = arange(0, Tmax, dt)
        dWs =  randn(N_ts-1) * sqrt_dt; 

        #the dynamics RHS:
        def compute_dX(alpha, x, dW):
            return (alpha + (mu - x)/tauchar)*dt + sigma * dW
        X_THRESH = 1.0
        'Main integration loop' 
        def computeHittingTime(x, alphas):
            thit = 0.0; 
            for t, dW, alpha in zip(ts, dWs, alphas):
    #            print '%.3f:%.3f'%(t,alpha)
                x+=compute_dX(alpha, x, dW);
                thit+=dt

                if x>=X_THRESH:
                    break;  
            
            return thit,x
     
        
        thits = empty(2);
        
        'Iterate over the various controls:'
        for adx, alphas in enumerate( [miAlphaF(ts),
                                        randAlphaF(ts)]):            
            thits[adx], xhit = computeHittingTime(0.0, alphas)
            if xhit<X_THRESH:
                raise Exception('xhit<X_THRESH - rethink assumptions');
            
        
        if visualize_paths:
            figure();
            stem(thits[0], [1], 'r-.');
            stem(thits[1], [1], 'g-.'); 
            
        return thits;
    
    ''' The basic loop harness to go through N  driver routines for the ONline Estimation:'''
    def runSingleExperiment(self):
        pass
    
    ''' harness to run a whole batch of experiments'''
    def runBatch(self):
        pass
    


def driverSingleTrial(fig_name = None):
          
    '''harness to try out  a single trial for the estimator'''
    simPs   = SimulationParams(tau_char = 1.)
   
    lEstimator = HTOnlineEstimator(simPs )
    
    lEstimator.runSingleTrial(visualize = True)




def driverSingleExperiment():
    ''' Run a single experiment for the Online Estimator using N trials and
        updating the particle ensemble in the process '''
    pass

def driverBatchEstimate():
    ''' run a set of N experiments and see compare the final experiment for the MI stimulator vs. the random Stimulator'''
    pass
    

if __name__ == '__main__':
    from pylab import *
    seed(2802)
    
    '''test stimulators - ok'''
#    driverStimulators()
    
    ''' Do a basic single trial'''    
    driverSingleTrial()
    
    
    
    
    
        
    show();