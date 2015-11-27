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

from AdjointSolver import FPAdjointSolver, xlabel_font_size
from AdjointOptimizer import AdjointOptimizer
from scipy.interpolate.interpolate import interp1d
from TauParticleEnsemble import TauParticleEnsemble

RESULTS_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/HTOnlineEstimator/'
FIGS_DIR    = '/home/alex/Workspaces/Latex/OptEstimate/Figs/HTOnlineEstimator'

import time
from multiprocessing import Process, Lock
import os
for D in [FIGS_DIR, RESULTS_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)
        
class Stimulator():
    def getStimulation(self, particleEnsemble, Tf):
        pass    
    
class MIStimulator(Stimulator):
    def __init__(self,  mu_sigma, reduce_to_2pt_prior=True, alpha_bounds=[-2,2]): 
        self.mu_sigma = mu_sigma;
        self.alpha_bounds = alpha_bounds;
        self.reduce_to_2pt_prior = reduce_to_2pt_prior;
        
        'Record for all ts,alphas applied:'
        self.ts_aopts_Massif = []
        
    'Default Initial Control Guess for the Optimization Procedure'
    def defaultInitialControl(self, ts, Tf_opt):
        'By default return a '
        alpha_max = self.alpha_bounds[1];
        init_alpha =  alpha_max*cos(2*pi*ts/Tf_opt);
        
        'Punch it past the optimization-interval:'
        init_alpha[where(ts>Tf_opt)] = alpha_max;
        
        return init_alpha;

    'Get 2pt prior with same mean, variance as a thicker prior:'
    def getReducedTausWeights(self, taus,weights):

        if abs(1.0-sum(weights)) > 1e-4:
            raise RuntimeError('getReducedTausWeights:: weights do NOT sum to 1')
        
        tau_mean = dot(taus, weights);
        tau_std_dev = sqrt( dot(taus**2, weights) - tau_mean**2);
        
        'reset to mean \pm std-dev:'
        taus = [tau_mean-tau_std_dev, tau_mean+tau_std_dev]
        
        weights = 0.5*ones_like(taus);
         
        return taus,weights; 
                
    'Calculate the Optimal Control given taus,weights'
    def calculateOptimalControl(self, taus, weights, Tf,
                                 init_ts_alpha= None):
        Tf_opt = Tf*0.66;
        
        'Construct Initial Guess for the Control:'
        if None == init_ts_alpha: 
            if 0 == len(self.ts_aopts_Massif):
                init_ts = arange(0, Tf, 0.01);
                init_alpha = self.defaultInitialControl(init_ts, Tf_opt);
                init_ts_alpha = [init_ts, init_alpha]
            else:
                init_ts_alpha = self.ts_aopts_Massif[-1];
        
        'Reduce to 2-pt?'
        if self.reduce_to_2pt_prior:
            taus,weights = self.getReducedTausWeights(taus,weights);
        
        'Construct Optimizer:'
        lOptimizer = AdjointOptimizer(taus, weights, self.mu_sigma,
                                      Tf, Tf_opt, 
                                      alpha_bounds= self.alpha_bounds);
                                    
        'Main Optimization Call:'
        lSolver, _, _, cs_iterates, _ = \
                lOptimizer.basic_gradDescent(initial_ts_cs = init_ts_alpha,
                                             K_max=20);
        
        'return ts,alpha_opt:'
        return lSolver._ts, cs_iterates[-1]

    
    'Compute the Stimulation given a particleEnsemble:'
    def getStimulation(self, particleEnsemble, Tf, init_ts_alpha = None): 
        ''' main inner call:'''
        ts, a_opt = self.calculateOptimalControl(particleEnsemble.taus, 
                                        particleEnsemble.weights,                                        
                                        Tf, init_ts_alpha=init_ts_alpha)
        
        'Archive:'
        self.ts_aopts_Massif.append([ts, a_opt])
        
        'Return:'
        return interp1d(ts, a_opt,
                        bounds_error = False,
                        fill_value = a_opt[-1]);
                  
 
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
    seed(28022011)

    Tf = 15.0;
    ts = arange(.0, Tf, .01)
    
    mu_sigma = [.0,1.0];
    
    'Stimulator Object:'
    mSfull = MIStimulator(mu_sigma,reduce_to_2pt_prior=False);
    mS2pt= MIStimulator(mu_sigma,reduce_to_2pt_prior=True);
    
    rS = RandomStimulator(mu_sigma);
    
    'Particle Ensembles:'
    pE = TauParticleEnsemble(16);
    pE.setLocationsUsingRange(0.5, 2);
    
    'Get stimulations:'
    mF_full = mSfull.getStimulation(pE, Tf);
    mF_2pt = mS2pt.getStimulation(pE,Tf);
    rF = rS.getStimulation()
    
    figure()
    plot(ts, mF_full(ts), 'bx-', label='optimal (full)')
    hold(True)
    plot(ts, mF_2pt(ts), 'rx-', label='optimal (2pt)')
    plot(ts, rF(ts),  'gx-', label='random')
    legend()
    
    figure(); hold(True)
    for tag, mS in zip(['full opt', '2pt opt'],
                       [mSfull, mS2pt]):
        current_ts = mS.ts_aopts_Massif[0][0];
        current_a_opt = mS.ts_aopts_Massif[0][1]; 
        plot(current_ts,current_a_opt, 'x-', label=tag);
    legend();
    

class HTOnlineEstimator():
#    STATIC PARAMS:
    def __init__(self, 
                 simParams,
                 Ntrials = 1, Nexperiments = 1,
                 FPSolver= None,
                 init_range = [.25, 4],
                 Ntaus = 32,
                 alpha_bounds = [-5,5],
                 MI_stimulation_time  = 15.0,
                 start_MIGD_from_rand_stim  = False,
                 update_stimulation_interval = 1):
        
        self.simParams = simParams;
        
        self.Ntrials = Ntrials;
        self.Nexperiments = Nexperiments
        
        self.miEnsemble = TauParticleEnsemble(Ntaus);
        self.miEnsemble.setLocationsUsingRange(init_range[0], init_range[1]);
        self.randEnsemble = deepcopy( self.miEnsemble );
        self.zeroEnsemble = deepcopy(self.miEnsemble);
        
        mu_sigma = [self.simParams.mu, self.simParams.sigma];
        self.alpha_bounds = alpha_bounds;
        
        ''' the online stimulators'''
        'The MI-opt stimulator:'
        self.miStimulator   = MIStimulator(mu_sigma,
                                            self.alpha_bounds  )        
        self.MI_Stimulation_Time = MI_stimulation_time;
        self.miStimulator.init_ts = linspace(0, self.MI_Stimulation_Time, 1000)
#        self.miStimulator.init_alpha = 0.9*self.alpha_bounds[-1]*cos(2*pi*self.miStimulator.init_ts/self.MI_Stimulation_Time);
        alpha_min  = 0.9*self.alpha_bounds[0];   
        alpha_max  = 0.9*self.alpha_bounds[-1];  
        Tf = MI_stimulation_time;
        init_ts = self.miStimulator.init_ts;
        self.miStimulator.init_alpha = alpha_min*(init_ts<=2*Tf/3) + alpha_max*(init_ts>2*Tf/3);
        'The random stimulator:'
        self.randStimulator = RandomStimulator(mu_sigma);
    
        
        self.likelihoodSolver = FPSolver;
        if None == self.likelihoodSolver:
            '''default likelihood solver'''
            taus = self.miEnsemble.taus;
            xmin = FPAdjointSolver.calculate_xmin(self.alpha_bounds ,
                                                  taus, 
                                                  mu_sigma, 
                                                  num_std = 1.0)
            ''' very fine discretization for the likelihood solver '''
            dx = FPAdjointSolver.calculate_dx(alpha_bounds,
                                                    taus,
                                               mu_sigma, xmin)
            dt = FPAdjointSolver.calculate_dt(alpha_bounds, 
                                               taus,
                                               mu_sigma,
                                               dx, xmin, factor=1.)
            print 'Solver params: xmin, dx, dt', xmin,dx,dt
            
            self.likelihoodSolver = FPAdjointSolver(dx, dt, 1.0, xmin)
        
        'whether to start opt MI grad descent from zero or from the random stim?'
        self.start_MIGD_from_rand_stim =  start_MIGD_from_rand_stim;
        
        'how often to recompute the stimulation:'
        self.update_stimulation_interval = update_stimulation_interval;
        
        ''' the archive massifs that keep track of the experiment'''
        self.miTausMassif   = {'taus':  [self.miEnsemble.taus],
                               'weights':   [self.miEnsemble.weights]}
        self.randTausMassif = {'taus':  [self.randEnsemble.taus],
                               'weights': [self.randEnsemble.weights]}
        self.zeroTausMassif = {'taus':  [self.zeroEnsemble.taus],
                               'weights': [self.zeroEnsemble.weights]}
        
        self.hitting_times = [];

        'The current stimulations:'        
        self.randAlpha = None;
        self.miAlphaF = None;
    
    '''Kernel Routine to run a single hitting time + update the particle weights'''
    def runSingleTrial(self, update_stimulus=True, visualize=False):        
        
        ''' get the stimulations '''
        if update_stimulus:
            print 'Re-calibrating stimulus!'
            'New Random Stimulation:'
            self.randAlphaF = self.randStimulator.getStimulation();
            
            'New Controlled Stimulation:'
            init_ts_alpha=None;
            if self.start_MIGD_from_rand_stim:
                alpha_init = self.randAlphaF(self.miStimulator.init_ts);
                init_ts_alpha = [self.miStimulator.init_ts, alpha_init]
            self.miAlphaF = self.miStimulator.getStimulation(self.miEnsemble,
                                                     self.MI_Stimulation_Time,
                                                     init_ts_alpha=init_ts_alpha );
                                                     
        '''run the Hitting TIme simulation:'''
        thits = self.simulateSinglePath(self.miAlphaF, self.randAlphaF , 
                                        visualize=visualize)

        print 'latest hitting times: ', thits
        
        'Store latest hitting times:'
        self.hitting_times.append(thits);
        
        '''update particle ensembles:'''
        for th, alphaF, pEnsemble, massif in zip(thits, 
                                  [self.miAlphaF,  self.randAlphaF, lambda ts: zeros_like(ts)],
                                  [self.miEnsemble, self.randEnsemble, self.zeroEnsemble],
                                  [self.miTausMassif, self.randTausMassif, self.zeroTausMassif]):
            
            tau_likelihood = self.calculateLikelihood(th, alphaF, pEnsemble,
                                                      visualize=visualize)
            
            pEnsemble.updateWeights(tau_likelihood);
            
            massif['taus'].append( pEnsemble.taus);
            massif['weights'].append( pEnsemble.weights);
        
        
        if visualize:
            figure();
            
            ts = arange(0, self.MI_Stimulation_Time, .1);
            plot(ts, self.miAlphaF(ts), 'r');
            plot(ts, self.randAlphaF(ts));
            
            title('alpha');
            
            
    def calculateLikelihood(self, th, alphaF, pE, visualize=False):
        '''Calculate the LIkelihood of the ensemble taus given a hitting time:'''
        dt = self.likelihoodSolver._dt;
         
        self.likelihoodSolver.setTf(th+0.5*dt);
 
        alphas_for_f = alphaF(self.likelihoodSolver._ts);
        
        taus = pE.taus;
        
        tau_likelihood = empty_like(taus)
        
        mu_sigma  = [self.simParams.mu, self.simParams.sigma];
        'main loop:'
        if visualize:
            figure();hold(True);
            subplot(211)
            
        for tdx, tau  in enumerate(taus):
            'calc likelihood for this tau'
            gs_dx = self.likelihoodSolver.solve_hittime_distn_per_parameter(tau,
                                                          mu_sigma,
                                                          alphas_for_f)
            
            'aggregate:'
            tau_likelihood[tdx] = gs_dx[-1]; 
            
            'visualize?'
            if visualize:
                plot(self.likelihoodSolver._ts, gs_dx,   label ='tau%.2f'%tau);
        
                
        if visualize:
            title('Likelihood');
            legend();
            stem([th], ylim()[1:])
            subplot(212);
            plot(self.likelihoodSolver._ts, alphas_for_f);
            title('Alpha')
        
        'return:'
        return tau_likelihood;
            
           
    def simulateSinglePath(self,  
                         miAlphaF,
                         randAlphaF,
                         Tmax = 50.0,
                         dt = 0.0005, 
                         x_0 = .0,
                         visualize = False):
        ''' returns [mi_thit, rand_thit, zero_thit]
        
         we make the judgement call to use the same random numbers PER DRAW
         that we really simulate each _thit_ as a single renewal process, even though
         in practice different stimulations will result in different hitting times and thus
         the driving dWs will desynchronise'''
        
        #ASSIGN STRUCT PARAMS:       
        mu, tauchar, sigma = self.simParams.mu, self.simParams.tau_char, self.simParams.sigma;
 
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
            xs = [x]; 
            for t, dW, alpha in zip(ts, dWs, alphas):
    #            print '%.3f:%.3f'%(t,alpha)
                x += compute_dX(alpha, x, dW);
                thit += dt
                if visualize:
                    xs.append(x);
                    

                if x>=X_THRESH:
                    break; 
            
            if visualize:
                figure(); subplot(211);
                plot(linspace(0, thit, len(xs)), xs);  
                subplot(212);
                plot(linspace(0, thit, len(xs)), alphas[0:len(xs)]);  
            
            return thit,x
     
        
        thits = empty(3);
        
        'Iterate over the various controls:'
        for adx, alphas in enumerate( [miAlphaF(ts),
                                        randAlphaF(ts),
                                        0*ts]):            
            thits[adx], xhit = computeHittingTime(0.0, alphas)
            
            if xhit<X_THRESH:
                raise Exception('xhit<X_THRESH - rethink assumptions');
            
            
            
        return thits;
    
    ''' The basic loop harness to go through N  driver routines for the Online Estimation:'''
    def runSingleExperiment(self, visualize=False):
        print 'Begin Experiment:'
        print 'Start from random stimulation for Grad Descent ICs', self.start_MIGD_from_rand_stim;
        print 'Stimulation update interval', self.update_stimulation_interval;
        print 'Alpha bounds', self.alpha_bounds;
        
        for edx in xrange(self.Ntrials):
            print 'simulating %d / %d trial'%(edx+1, self.Ntrials)
            
            
            visualize_trial = visualize and (0 == edx or (self.Ntrials-1) == edx);
             
            update_stimulus = (0 == mod(edx, self.update_stimulation_interval));
            
            if update_stimulus:
                self.update_stimulation_interval*=2;
            
            self.runSingleTrial(update_stimulus=update_stimulus,
                                visualize=visualize_trial)
            
            print 'MI-Opt log-Ensemble = ' + self.miEnsemble.ensembleLogMeanStdString();
            print 'Rand log-Ensemble = '   + self.randEnsemble.ensembleLogMeanStdString();
            print 'Zero log-Ensemble = '   + self.zeroEnsemble.ensembleLogMeanStdString();

            
#    ''' harness to run a whole batch of experiments '''
#    def runBatch(self):
#        pass
#    
    
    '''load/save utilities'''
    def _default_file_name(self, mu, tau, sigma, Ntaus, Ntrials, Nexperiments):
        return 'HTOnlineEstimator_m=%.1f_t=%.1f_s=%.1f_'%(mu, tau, sigma) + \
                'Nts=%d_Ntrls=%d_Nexps=%d.htoe'%(Ntaus, Ntrials, Nexperiments)
    
                  
    def save(self, file_name=None):
#        path_data = {'path' : self}
        self.miAlphaF = None;
        self.randAlphaF = None;
        
        if None == file_name:
            Ntaus = len(self.miEnsemble.taus);            
            file_name = self._default_file_name(self.simParams.mu,
                                                self.simParams.tau_char,
                                                self.simParams.sigma,
                                                Ntaus,
                                                self.Ntrials,
                                                self.Nexperiments);
             
        print 'saving estimation run to ', file_name
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


def driverMIOptimization():
    raise RuntimeError('DEPRECATD')
#    lEstimator = HTOnlineEstimator.load('single_experiment_example');
#    
#    taus_init = lEstimator.miTausMassif['taus'][0] 
#    weights_init = lEstimator.miTausMassif['weights'][0] 
#
#    taus_final = lEstimator.miTausMassif['taus'][5]
#    weights_final = lEstimator.miTausMassif['weights'][5] 
#    
#    print taus_init 
#    print taus_final
#    
#    print weights_init
#    print weights_final
#    
#    Tf =  10.0;
#    
#    ''' initial taus from zero'''
#    S, fs, ps, alpha_iterations, Js = calculateOptimalControl(taus_init , 
#                                                                weights_init,
#                                                                [0,1],
#                                                                    Tf,
#                                            initial_ts_cs= [ array([0,1]) ,
#                                                             zeros(2) ]); 
#                                                             
#    alpha_init =    alpha_iterations[-1]                                                      
#    figure(figsize=(17,12)) ; hold(True)
#    plot(S._ts,  alpha_init, 'b-', label='INit Taus from AlphaZero')
#   
#   
#    ''' final taus from initial alpha'''
#    S, fs, ps, alpha_iterations, Js = calculateOptimalControl(taus_final , 
#                                                                weights_final,
#                                                                [0,1],
#                                                                Tf,
#                                            initial_ts_cs= [ S._ts, 
#                                                             alpha_init]); 
#                                                             
#                                                             
#    plot(S._ts, alpha_iterations[-1], 'g-', label='Final Taus and AlphaOpt_1') 
#    
#    ''' final taus from zero alpha'''
#    S, fs, ps, alpha_iterations, Js = calculateOptimalControl(taus_final , 
#                                                                weights_final,
#                                                                [0,1],
#                                                                Tf,
#                                                            initial_ts_cs= [ array([0,1]) ,
#                                                                                zeros(2) ] ); 
#                                                             
#                                                             
#    plot(S._ts, alpha_iterations[-1], 'r-', label='Final Taus from AlphaZero')                                                         
#    legend(loc='lower left')
    
    
'Show a single trial - that is a single hitting time with all stimulations:'
def driverSingleTrial(fig_name = None, resimulate = True):          
    '''harness to try out  a single trial for the estimator'''
    simPs   = SimulationParams(tau_char = 1.)
    seed(28021111)
    
    if resimulate:        
        lEstimator = HTOnlineEstimator(simPs, Ntaus=32, alpha_bounds = [-3,3])
        
        start = time.clock()
        lEstimator.runSingleTrial(visualize=True)
        print('Single Trial Time = ', time.clock()-start, 's') 
        
        lEstimator.save('single_trial_example')
        
    
    ''' Now reload and visualize'''
    lEstimator = HTOnlineEstimator.load('single_trial_example')
    
    ''' VISUALIZE '''
    print lEstimator.miTausMassif
    print lEstimator.randTausMassif
    
    ''' Tau Weights Plot'''
    weights_fig = figure(figsize = (17, 8) );
    for pdx, massif, title_tag in zip([1,2,3],
                                      [lEstimator.miTausMassif,
                                        lEstimator.randTausMassif,
                                         lEstimator.zeroTausMassif],
                                      ['Optimal Stimulation Ensemble',
                                        'Random Stimulation Ensemble',
                                        'Zero Stimulation Ensemble' ]):    
        prev_taus = massif['taus'][-2]
        next_taus = massif['taus'][-2]
        
        prev_weights = massif['weights'][-2]
        next_weights = massif['weights'][-1]
        
        subplot(3,1,pdx)
        hold(True);
        
        stem(prev_taus, prev_weights, 'b-.', markerfmt='bo')
        stem(next_taus, next_weights,  'r-.', markerfmt='ro')
        xlim([0, 5])
        title(title_tag)
        legend(['before', 'after'])
    xlabel('Tau Locations'); ylabel('Tau Weights');
    
    ''' Hitting TImes Plot'''
    ht_fig = figure( figsize = (17, 4) );
    hts = array(lEstimator.hitting_times);
    
    for h in lEstimator.hitting_times[0]:    
        print "%.8f"%h 
         
    for sdx, color in zip([0,1, 2], ['r', 'g', 'b']):
        hts_opt = hts[:,sdx];
        stem( cumsum(hts_opt), ones_like(hts_opt), color+'-.', markerfmt = color+'o' );
    xlabel('$t$'); title('Example of hitting times')
    legend( ['OptStim', 'RandStim', 'ZeroStim' ]) 
    
    'controls plot'
    cs_fig = figure( figsize = (17, 4) );
    hold(True);
    ts = lEstimator.miStimulator.ts_aopts_Massif[0][0];
    plot(ts,
         lEstimator.miStimulator.ts_aopts_Massif[0][1], 'r', linewidth=4)
    plot(ts, lEstimator.randStimulator.alphasMassif[0]*ones_like(ts), 'g')
    plot(ts, zeros_like(ts), 'b');
    xlabel('$t$'); title('Sample Trial Controls')
    legend( ['OptStim', 'RandStim', 'ZeroStim' ]) 
    
    if fig_name != None:
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '_weights.pdf');
        print 'saving to ', lfig_name
        weights_fig.savefig(lfig_name)
         
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '_hittimes.pdf');
        print 'saving to ', lfig_name
        ht_fig.savefig(lfig_name); 
        
        

def driverSingleExperiment(fig_name = None, resimulate = True):          
    ''' Run a single experiment for the Online Estimator using N trials and
        updating the particle ensemble in the process '''
    simPs = SimulationParams(tau_char = 1.)
    Ntaus = 32
    Ntrials = 500
    seed(Ntrials)

    experiment_tag ='Nts=%d_Ntrls=%d'%(Ntaus, Ntrials)
    save_file_name = 'single_experiment_example_%s'%experiment_tag
    if resimulate:
        start_time = time.clock();        
        lEstimator = HTOnlineEstimator(simPs,
                                        Ntrials = Ntrials,
                                        Ntaus=Ntaus,
                                        alpha_bounds = [-3,3],
                                        start_MIGD_from_rand_stim=False,
                                        update_stimulation_interval=10)
        
        'main call:'
        lEstimator.runSingleExperiment( visualize=True )
        
        'save output:'        
        lEstimator.save(save_file_name)
        
        print 'Single Experiment Time = ', time.clock()-start_time, 's';
        
    
    ''' Now reload and visualize'''
    lEstimator = HTOnlineEstimator.load(save_file_name)

    print 'MI tau locations:\n', lEstimator.miTausMassif['taus'][-1]
    
    print 'MI tau weights \n', lEstimator.miTausMassif['weights'][-1]
    
    print 'hitting times:\n', array(lEstimator.hitting_times)
    
    'visualize ensembles' 
    visualizeEstimation(lEstimator, simPs, fig_name = fig_name);
   
    'visualize control evolution:'
    tsalphasMassif = lEstimator.miStimulator.ts_aopts_Massif;

    cs_fig = figure(figsize = (17, 12));   
    for idx, tsalphas in enumerate(tsalphasMassif):
        subplot(ceil(len(tsalphasMassif)/2), 2, 1+idx);
        plot(tsalphas[0], tsalphas[1])
        title('Control-Iteration:%d'%idx)
    xlabel(r'$t$', fontsize = xlabel_font_size)
    
    lfig_name = os.path.join(FIGS_DIR,
                              fig_name + '_controls_evolution.pdf');
    print 'saving to ', lfig_name 
    cs_fig.savefig(lfig_name)     
        
   
        
def driverBatchEstimate(resimulate = False,
                        Ntaus = 32,
                         Nexperiments=50,
                          Ntrials = 500,
                           simPs = SimulationParams(),
                           save_base_name = 'single_experiment',
                           multi_process_flag=True,
                           restart_experiment_idx=0):
    
    'Main Computational Function:'
    def singleExperimentProcess(experiment_idx, procLock):
        'set seed'
        seed(experiment_idx+1)
    
        experiment_tag = '%s_id%d'%(save_base_name, experiment_idx);
        
        'parametrize:'
        lEstimator = HTOnlineEstimator(simPs,
                                    Ntrials = Ntrials,
                                    Ntaus=Ntaus,
                                        alpha_bounds = [-3,3],
                                        start_MIGD_from_rand_stim=False,
                                        update_stimulation_interval=25)
        'run:'
        lEstimator.runSingleExperiment(visualize=False)
        
        'save to disk:'
        procLock.acquire();
        lEstimator.save(experiment_tag)
        procLock.release();
        'Return'
     
    'Main (re)simulation:'
    if resimulate:
        start = time.time()
        procLock = Lock();
        if multi_process_flag:
            'Multi-Proc Implementation:'
            Nprocs_at_a_time = 4
            procs  = [];
            procLock = Lock();
            for experiment_idx in xrange(restart_experiment_idx,Nexperiments):
                'Add to Queue:'
                procs.append(Process(target=singleExperimentProcess,
                                     args=(experiment_idx, procLock)) );
                'start sub-process:' 
                procs[-1].start();
                
                if 0 == mod(experiment_idx+1, Nprocs_at_a_time):
                    'Barrier it (flush process queue):'                    
                    for proc in procs:
                        proc.join()
                    'reset procs'
                    procs = [];
        else:
            'Serial Implementation:'
            for experiment_idx in xrange(Nexperiments):
                singleExperimentProcess(experiment_idx, procLock);
        
        'Diagnose: (and return)'
        print 'total batch compute time = ', time.time()-start, 's';
                  
    ''' Now reload and visualize'''
#    for experiment_idx in xrange(Nexperiments):
#        save_file_name = '%s_id%d'%(save_base_name, experiment_idx);
#        lEstimator = HTOnlineEstimator.load(save_file_name)
#        visualizeEstimation(lEstimator, simPs);

def driverNarrowExperiment(fig_name = None, resimulate = True):          
    ''' Run a single experiment for the Online Estimator using N trials and
        updating the particle ensemble in the process '''
    simPs = SimulationParams(tau_char = 1.)
    Ntaus = 32
    Ntrials = 102
    seed(Ntrials)
    
    experiment_tag ='Nts=%d_Ntrls=%d'%(Ntaus, Ntrials)
    save_file_name = 'narrow_prior_experiment_%s'%experiment_tag
    if resimulate:        
        lEstimator = HTOnlineEstimator(simPs,
                                        Ntrials = Ntrials,
                                        Ntaus=Ntaus,
                                        alpha_bounds = [-2,2],
                                        start_MIGD_from_rand_stim=False,
                                        update_stimulation_interval=20,
                                        init_range = [.66, 1.5])
    
        lEstimator.runSingleExperiment( visualize=True )
#        
        lEstimator.save(save_file_name)
        
    
    ''' Now reload and visualize'''
    lEstimator = HTOnlineEstimator.load(save_file_name)

    print 'MI tau locations:\n', lEstimator.miTausMassif['taus'][-1]
    
    print 'MI tau weights \n', lEstimator.miTausMassif['weights'][-1]
    
    print 'hitting times:\n', array(lEstimator.hitting_times)
    
    'visualize ensembles' 
    visualizeEstimation(lEstimator, simPs, fig_name = fig_name);
   
    'visualize control evolution:'
    tsalphasMassif = lEstimator.miStimulator.ts_aopts_Massif;
    
    cs_fig = figure(figsize = (17, 12));   
    
    pdx = 0
    for idx, tsalphas in enumerate(tsalphasMassif ):
        if mod(idx, 4) == 0:
            pdx+=1;
        subplot(2,1,pdx); hold(True);
        plot(tsalphas[0], tsalphas[1], label='%d'%idx)
        legend(loc='lower right');
    xlabel(r'$t$', fontsize = xlabel_font_size)
    
    if None == fig_name:
        return
    lfig_name = os.path.join(FIGS_DIR,
                              fig_name + '_controls_evolution.pdf');
    print 'saving to ', lfig_name 
    cs_fig.savefig(lfig_name)  

def visualizeEstimation(lEstimator, simPs, fig_name=None):  
    print 'MI-Opt log-Ensemble = ' + lEstimator.miEnsemble.ensembleLogMeanStdString();
    print 'Rand log-Ensemble = '  + lEstimator.randEnsemble.ensembleLogMeanStdString();
    print 'Zero log-Ensemble = ' + lEstimator.zeroEnsemble.ensembleLogMeanStdString();
 
    'visualize ensembles'
    ensemble_fig = figure(figsize = (17, 8));   
    hold(True);
     
    for pdx,  massif, title_tag,color in zip([1,2,3],
                                      [lEstimator.miTausMassif, lEstimator.randTausMassif, lEstimator.zeroTausMassif],
                                      ['MI-Opt Stimulation', 'Rand Stimulation', 'Zero Stimulation'],
                                      ['r', 'g', 'b']):
        tks = arange(0, lEstimator.Ntrials+1);
        taus = log( array( massif['taus'] ) );
        ws   =  array( massif['weights'] );
        
        mean_tau = sum(taus*ws, 1);
    #        mean_tau = 1/sum(ws/taus,1);
        
        std_tau = sqrt( sum(taus*taus*ws,1) - mean_tau*mean_tau );
        plot( tks, mean_tau, color + '-', label=title_tag);
        plot( tks, mean_tau  + 2*std_tau, color + '--');
        plot( tks, mean_tau  - 2*std_tau, color + '--');
    
    title(r'Empirical CIs for $\rho(log(\tau))$',
          fontsize = xlabel_font_size);
    xlabel('$k$', fontsize = xlabel_font_size);
    ylabel(r'log mean $\pm$ 2 stdev', fontsize = xlabel_font_size);
    ylim([-2, 2]);
    plot(tks, log( simPs.tau_char*ones_like(tks)), 'k-');
    legend()
    
    'Save Fig:'
    if fig_name == None:
        return
    
    lfig_name = os.path.join(FIGS_DIR,
                              fig_name + '_ensemble_distn_evolution.pdf');
    print 'saving to ', lfig_name 
    ensemble_fig.savefig(lfig_name) 
      
def visualizeAggregatedBatch(fig_name = 'online_updated_prior',
                             Ntaus = 32,
                             Nexperiments=50,
                             Nobs = 500,
                             simPs = SimulationParams(),
                             save_base_name = 'single_experiment'):
    
    
    estimatorsList = [];    
    'Load Resutls:'
    for experiment_idx in xrange(Nexperiments):
        save_file_name =  '%s_id%d'%(save_base_name, experiment_idx);
        estimatorsList.append(HTOnlineEstimator.load(save_file_name));
        
    'Gather Results:'   
    meanstdDict = {};
    mean_taus_Dict = {};
    
    for tag in ['miTausMassif', 'randTausMassif', 'zeroTausMassif']:
        m1_tau = zeros(Nobs);
        m2_tau = zeros(Nobs);
        mean_taus_array= zeros( (Nobs, Nexperiments) );
        for edx,lE in enumerate(estimatorsList):
            massif = getattr(lE, tag); 
            
            taus = array( massif['taus'] )[0:Nobs];
            log_taus = log( taus ) ;
            ws   =  array( massif['weights'][0:Nobs]  );
            
            'Compute Mean:'
            mean_taus_array[:, edx] = sum(taus*ws, 1); 
            m1_tau +=  sum(log_taus*ws, 1);

            m2_tau += sum(log_taus*log_taus*ws,1);
        
        mean_tau = m1_tau / Nexperiments;
                
        std_tau = sqrt( m2_tau/Nexperiments - mean_tau*mean_tau );
        meanstdDict[tag] = [mean_tau, std_tau];
        mean_taus_Dict[tag] = mean_taus_array;
     
     
    'Visualize Ensemble:'
    ensemble_fig = figure(figsize=(17,8)) 
    hold(True)
    tks = arange(1, Nobs+1)  
    for tag, color in zip(['miTausMassif', 'randTausMassif', 'zeroTausMassif'],
                           ['r', 'g', 'b']):
        mean_tau, std_tau  = (x for x in meanstdDict[tag])
        plot( tks, mean_tau, color + 'x-', label=tag);
        plot( tks, mean_tau  + 2*std_tau, color + '--');
        plot( tks, mean_tau  - 2*std_tau, color + '--');
    xlim([0, Nobs])
    title(r'Empirical Confidence Intervals of $\rho(log(\tau))$',
          fontsize = xlabel_font_size);
    xlabel('$k$', fontsize = xlabel_font_size);
    ylabel(r'$log(\tau)$', fontsize = xlabel_font_size);
    ylim([-1.8, 1.8]);
    plot(tks, log( estimatorsList[0].simParams.tau_char*ones_like(tks)), 'k-');
    legend()
     
     
    'Visualize Quantiles:'
    quantile_fig = figure(figsize=(17,8)); hold(True)
    tks = arange(1, Nobs+1)  
    for tag, color in zip(['miTausMassif', 'randTausMassif', 'zeroTausMassif'],
                           ['r', 'g', 'b']):
        
        mean_tau_array = mean_taus_Dict[tag] 
        
        plot( tks, median(mean_tau_array, 1), color + 'x-');
        plot( tks, amin(mean_tau_array, 1), color + '--', label=tag);
        plot( tks, amax(mean_tau_array, 1), color + '--');
    
    xlim([0, Nobs])
    title(r'Min/Median/Max $\hat{\tau}$',
          fontsize = xlabel_font_size);
    xlabel('$k$', fontsize = xlabel_font_size);
    ylabel(r'$\hat{E[\tau]}$', fontsize = xlabel_font_size);
    ylim([0, 3]);
    plot(tks,  estimatorsList[0].simParams.tau_char*ones_like(tks) , 'k-');
    legend()
      
    
    mifig = figure(figsize=(17,8)); hold(True)
    tks = arange(1, Nobs+1)  
    mean_tau_array = mean_taus_Dict['miTausMassif']  
    for cdx in xrange( shape(mean_tau_array)[1] ):
        plot( tks, mean_tau_array[:, cdx], '--'); 
    
    'Save Figs:'
    if None != fig_name:
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '_mean_aggregated_ensemble.pdf');
        print 'saving to ', lfig_name 
        ensemble_fig.savefig(lfig_name)     
        
        lfig_name = os.path.join(FIGS_DIR,
                                  fig_name + '_quantiles_mean_per_experiment.pdf');
        print 'saving to ', lfig_name 
        quantile_fig.savefig(lfig_name)
    else:
        print 'Not Saving Figs!'
       
    
if __name__ == '__main__':
    from pylab import *
    
    '''test the random/mi-optimal stimulators'''
#    driverStimulators()
    
    ''' Do a basic single trial (single hitting-time) '''    
#    driverSingleTrial(fig_name = 'single_trial_example')

    ''' Do a basic single experiment '''
#    driverSingleExperiment(fig_name = 'single_experiment_example')
    
    ''' test the MI Optimization process '''
#    driverMIOptimization();

    ''' batch a bunch of experiments '''
    Nt = 32; Ne=50; Nobs=500;
#    driverBatchEstimate(resimulate=True,Ntaus=Nt, Nexperiments=Ne, Ntrials=Nobs,
#                        multi_process_flag=True)
    
    ''' visualize aggregated belief'''
    visualizeAggregatedBatch(Ntaus=Nt, Nexperiments=Ne, Nobs=Nobs)


    ''' narrow prior estimate'''
#    driverNarrowExperiment()
    
        
    show();