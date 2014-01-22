'''
Created on May 9, 2012

@author: alex
'''
from numpy import iterable

import os
import shutil
TABLES_DIR = '/home/alex/Workspaces/Python/OptEstimate/Results/hd5/'

for D in [TABLES_DIR]:
    if not os.path.exists(D):
        os.mkdir(D)

from tables import *       

class SolutionParams(IsDescription):
    link_id = UInt16Col(pos = 0)  
    name = StringCol(64) #64 char string
    dx   = Float32Col()
    xmin = Float32Col()
    dt   = Float32Col()
    Tf   = Float32Col()
#    class Params(IsDescription):
    mu   = Float32Col()
    beta = Float32Col()
    

def initializeTable(overwrite = False):
    hd5_file_name = os.path.join(TABLES_DIR,
                                 hd5Solution._TABLE_NAME)
    
    if os.path.exists(hd5_file_name) and (not overwrite):
        print 'hd5 File already created, aborting'
        return
    
    hd5_handle = openFile(hd5_file_name, mode = "w")
    #get the root
    root = hd5_handle.root;
    
    #create the solutions group:
    solGrp = hd5_handle.createGroup(root, "Solutions")
    
    hd5_handle.createTable(solGrp, 'params',
                        SolutionParams, 'Solution Parameters')
    
    #close the file (and flush!)
    hd5_handle.close()
    
########################
class hd5Solution():
    _TABLE_NAME = 'ForwardSolutions.h5'
    def __init__(self, params,
                 dx, xmin,
                 dt, Tf,
                 Fs,  
                 alphas,
                 thetas):
        self._dx  = dx;
        self._xmin = xmin;
        self._dt  = dt;
        self._Tf = Tf;
        
        self._mu = params[0]
        self._beta = params[1]
        
        self._thetas = thetas
        
        self._Fs = Fs
        self._alphas = alphas
        
    def save(self, name=''):
#       path_data = {'path' : self}
        hd5_file_name = os.path.join(TABLES_DIR,
                                     self._TABLE_NAME)
        
        hd5_handle = openFile(hd5_file_name, 'r+')
        grp = hd5_handle.root.Solutions;
        
        link_id= grp.params.nrows;
        #save to params
        params = grp.params.row
        params['link_id'] = link_id
        params['name'] = name;
        params['dx'] = self._dx
        params['xmin'] = self._xmin
        params['dt'] = self._dt
        params['Tf'] = self._Tf
        params.append();
        grp.params.flush();
        
        #now store Fs:
        for arr, arr_id in zip([self._Fs, self._thetas, self._alphas],
                               ['Fs', 'thetas', 'alphas']):
            hd5_handle.createArray(grp, str(link_id)+'_%s'%arr_id,
                                   arr, '%s for %s'%(arr_id, name))
        
        
        
        hd5_handle.close() #and flush
    
    @classmethod
    def load(cls, name=None, mu_beta_Tf=None):
        ''' not both args can be None!!!'''
        #Load the hd5 file:
        hd5_file_name = os.path.join(TABLES_DIR,
                                     cls._TABLE_NAME)
        
        hd5_handle = openFile(hd5_file_name, 'r')
        grp = hd5_handle.root.Solutions;
        
        param_record = None;
        select_criteria = '';
        if None != name:
            select_criteria = 'name == "'+ name +'"' #SQL Inject vector
        else:
            select_criteria = 'name == "'+ name +'"' #SQL Inject vector
        
        print 'loading rows where '+ select_criteria
        
        for row in grp.params.where(select_criteria):
            param_record = row;
            #get only the first one (God, this is an awful hack)
            break;
        
        link_id = param_record["link_id"];
        Fs      = hd5_handle.getNode("/Solutions", "%d_Fs"%link_id).read()
        alphas  = hd5_handle.getNode("/Solutions", "%d_alphas"%link_id).read()
        thetas  = hd5_handle.getNode("/Solutions", "%d_thetas"%link_id).read()
        
        soln = hd5Solution([param_record['mu'], param_record['beta'] ], 
                           param_record['dx'], param_record['xmin'], 
                           param_record['dt'], param_record['Tf'],
                            Fs,
                             alphas, 
                              thetas)
        
        #always close the db:-)
        hd5_handle.close();
        
        return soln
################################################

class DataHarvester(object):
    '''
    This collects the data from an estimation run and records it in a PyTable
    '''

    def __init__(self, table_file_name,
                       new_table_file_name=None,
                       overwrite=False):
        '''
        The table_file_name is the hd5 file name where the table is recorded  
        '''
        
        file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
        
        if None != new_table_file_name:
            old_file_name = file_name 
            file_name = os.path.join(TABLES_DIR, new_table_file_name+ '.h5')
            if os.path.exists(old_file_name):
                shutil.copyfile(old_file_name, file_name)
            else:
                print ('Warning: old file name given, but such file DNE! name = ', old_file_name)
        
        if os.path.exists(file_name) and False == overwrite: #Load existing table:
            self._File = openFile(file_name, mode = "r+")
            self._estGroup = self._File.getNode('/', 'Estimates')
            self._sampleGroup = self._File.getNode('/', 'Samples') 
            self._regimeGroup = self._File.getNode("/", 'Regimes') #TODO: Regime group might be unnecessary as a private member =- we never seem to use it:
            self._regimeTable = self._regimeGroup._f_getChild('Regimes')
        
        else: #create it (destructively if necessary):
            self._File = openFile(file_name, mode = "w", title = "Sinusoidal LIF Estimation Report")
            self._estGroup = self._File.createGroup("/", 'Estimates', "Estimates Information")
            self._sampleGroup = self._File.createGroup("/", 'Samples', "Sample Information") 
            self._regimeGroup = self._File.createGroup("/", 'Regimes', "Regime Info")
            self._regimeTable = self._File.createTable(self._regimeGroup, 'Regimes', Regime, "Regimes")
        
        #Associate the table groups with the object members:        
        self._regime = self._regimeTable.row

        self._estTable = None
        self._sampleTable = None
        self._estimate = None
        self._sample = None
        
    
    def flush(self):
        self._File.flush()

    def closeFile(self):
        self._File.close()

    def __del__(self):
        self._File.flush()
        self._File.close()

    def setRegime(self, regime_name, abg_true, Tsim):
        '''first check if regime exists'''
        self.flush()
        regime_idx_array = self._regimeTable.getWhereList('name == "'+ regime_name + '"')
        if 0 != len(regime_idx_array):
            try:
                self._estTable = self._estGroup._f_getChild(regime_name)
            except:
                self._estTable = self._File.createTable(self._estGroup, regime_name, Estimate , "Regime Estimate")
            
            try:
                self._sampleTable = self._sampleGroup._f_getChild(regime_name)
            except:
                self._sampleTable = self._File.createTable(self._sampleGroup, regime_name, Sample , "Regime Samples")
        
        else: #add a new one:
            self._regime['name'] = regime_name
            self._regime['alpha'] = abg_true[0]
            self._regime['beta'] = abg_true[1]
            self._regime['gamma'] = abg_true[2]
            self._regime['Tsim'] = Tsim
            self._regime.append()
            
            self._estTable = self._File.createTable(self._estGroup, regime_name, Estimate , "Regime Estimate")
            
            self._sampleTable = self._File.createTable(self._sampleGroup, regime_name, Sample , "Regime Samples")

        self._estimate = self._estTable.row
        self._sample = self._sampleTable.row
        
    def addEstimate(self, sample_id, method, abg, walltime, warnflag):
            self._estimate['sample_id'] = sample_id
            self._estimate['method'] = method
            self._estimate['alpha'] = abg[0]
            self._estimate['beta'] = abg[1]
            self._estimate['gamma'] = abg[2]
            self._estimate['walltime'] = walltime
            self._estimate['warnflag'] = warnflag
            
            self._estimate.append()
            
    def addSample(self, sample_id, Tf, BinCount, SpikeCount):
            self._sample['sample_id'] = sample_id
            self._sample['Tf'] = Tf
            self._sample['BinCount'] = BinCount
            self._sample['SpikeCount'] = SpikeCount

            self._sample.append()
        
    
    def getTrueParamValues(self, regime_name):
        abg_true = [(row['alpha'], row['beta'], row['gamma']) 
                             for row in self._regimeTable.iterrows()
                                if row['name'] == regime_name]    
        return abg_true[0]
    
    def getEstimates(self, sample_ids, regime, mthd):
#        samplesT = self._sampleGroup._f_getChild(regime)
        estimatesT = self._estGroup._f_getChild(regime)
#        samplesCount = len(samplesT)
        
        if False == iterable(sample_ids):
            if  int == type(sample_ids):
                sample_ids = [sample_ids]
            else:
                raise TypeError('Sample Ids should be a sequence or an int')
        
        abg_estimates  = [(row['alpha'], row['beta'], row['gamma'])
                             for row in estimatesT.iterrows()
                                    if (mthd == row['method']) and 
                                       (row['sample_id'] in sample_ids) ]
        if 0 == len(abg_estimates):
            print 'Warning: no estimates found for  sample_id = ', sample_ids,' in ',regime, ' ,' , mthd
        return abg_estimates
            
    def printme(self):
        print self._File
   
class DataPrinter(object):
    '''
    This prints the data to console faking the interface of a DataHarverster 
    '''

    def __init__(self, table_file_name=None, new_table_file_name=None):
        '''
        The table_file_name is the hd5 file name where the table is recorded  
        '''
        pass
    
    def setRegime(self, regime_name, abg_true, Tsim):
        '''first check if regime exists'''
        print regime_name, abg_true
         
        
    def addEstimate(self, sample_id, method, abg, walltime, warnflag):
        print sample_id, ': ', method, ': %.2f,%.2f,%.2f ' %(abg[0],abg[1],abg[2]), ' in ', walltime, ' : warnflag=', warnflag
             
    def addSample(self, sample_id, Tf, BinCount, SpikeCount):
#        print sample_id, ' '
        pass
    def closeFile(self):
        print 'le fin!!!'
                        

class DataAnalyzer(object):
    '''
    This collects the data from an estimation run and records it in a PyTable
    '''

    def __init__(self, table_file_name):
        '''
        The table_file_name is the hd5 file name where the table is recorded  
        '''
        file_name = os.path.join(TABLES_DIR, table_file_name+ '.h5')
        self._h5file = openFile(file_name, mode = "r")

        self._Estimates = self._h5file.getNode('/', 'Estimates')
        self._Samples =  self._h5file.getNode('/', 'Samples')
        self._regimesTable =  self._h5file.getNode('/', 'Regimes')._f_getChild('Regimes')
        
    def getEstimates(self, sample_ids, regime, mthd):
        samplesT = self._Samples._f_getChild(regime)
        estimatesT = self._Estimates._f_getChild(regime)
#        samplesCount = len(samplesT)
        
        if False == iterable(sample_ids):
            if  int == type(sample_ids):
                sample_ids = [sample_ids]
            else:
                raise TypeError('Sample Ids should be a sequence or an int')
        
#        abg_estimates =[]
#        for row in estimatesT.where('method == "'+ mthd +'" and sample_ids in ' + sample_ids):
#                abg_estimates.append( (row['alpha'], row['beta'], row['gamma']))
        
        abg_estimates  = [(row['alpha'], row['beta'], row['gamma'])
                             for row in estimatesT.iterrows()
                                    if (mthd == row['method']) and 
                                       (row['sample_id'] in sample_ids) ]
        if 0 == len(abg_estimates):
            print 'Warning: no estimates found for  sample_id = ', sample_ids,' in ',regime, ' ,' , mthd
        return abg_estimates

    
    def getAllEstimates(self, regime, mthd):
        estimatesT = self._Estimates._f_getChild(regime)
        abg_estimates  = [(row['alpha'], row['beta'], row['gamma'])
                        for row in estimatesT.iterrows()
                        if row['method'] == mthd ]
        if 0 == len(abg_estimates):
            print 'Warning: no estimates found for  ',regime, ' ,' , mthd
        
        return abg_estimates
           

    def getTrueParamValues(self, regime_name):
        abg_true = [(row['alpha'], row['beta'], row['gamma']) 
                             for row in self._regimesTable.iterrows()
                                if row['name'] == regime_name]    
        return abg_true[0]
    
    def getSampleCount(self, regime):
        samplesT = self._Samples._f_getChild(regime)
#TODO: APparently, it is possible to double include samples in the Samples Table...
#WARNING: DOn't use this method unless you must...
        return len(samplesT)
            
    def getAllRegimes(self):
        return [row['name'] for row in self._regimesTable.iterrows()]
    
    def getWallTimes(self, regime, mthd):
        estimatesT = self._Estimates._f_getChild(regime)
        
        wall_times = [row['walltime']
                            for row in estimatesT.iterrows()
                                if row['method'] == mthd ]
        if 0 == len(wall_times):
            print 'Warning: no estimates found for  ',regime, ' ,' , mthd
        
        return wall_times
        
    def getAllWarnings(self):
        warnings = []
        for regime in self.getAllRegimes():
            estimatesT = self._Estimates._f_getChild(regime)
            warnings += [(regime, row['sample_id'], row['method'], row['warnflag'])
                            for row in estimatesT.iterrows()
                                if row['warnflag'] > 0]
        return warnings
        
        
#def writeWithHarvester():
#    from BinnedSpikeTrain import BinnedSpikeTrain
#    from InitBox import initialize5
#    from Simulator import Path, OUSinusoidalParams
#    import time
#
#    N_phi = 20;
#    print 'N_phi = ', N_phi
#    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
#    
##    D = DataHarvester('test2')
#    D = DataHarvester('test2', 'test3')
#
#    base_name = 'sinusoidal_spike_train_T='
#    for regime_name, T_sim, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
#                                                       [5000 , 20000, 5000, 5000],
#                                                       [4., 32, 16., 16.]): 
#        regime_label = base_name + str(T_sim)+ '_' + regime_name
#            
#        for sample_id in xrange(3,4):
#            file_name = regime_label + '_' + str(sample_id) + '.path'
#            print file_name
#            
#            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
#            ps = binnedTrain._Path._params
#            abg_true = array((ps._alpha, ps._beta, ps._gamma))
#            
#            D.setRegime(regime_name,abg_true, T_sim)
#            
#            phi_omit = None
#            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
#            Tf = binnedTrain.getTf()
#            D.addSample(sample_id, Tf, binnedTrain.getBinCount(), binnedTrain.getSpikeCount())
#            
#            start = time.clock()
#            abg_init = initialize5(binnedTrain)
#            finish = time.clock()
#            D.addEstimate(sample_id, 'Initializer', abg_init, finish-start) 
#                    
#            abg_est = abg_init
#            start = time.clock()
#    #        abg_est = NMEstimator(S, binnedTrain, abg_init)
#            time.sleep(rand())
#            finish = time.clock()
#            D.addEstimate(sample_id, 'Nelder-Mead', abg_est, finish-start) 
#            
#            start = time.clock()
#    #        abg_est = BFGSEstimator(S, binnedTrain, abg_init)
#            time.sleep(rand())
#            finish = time.clock()
#            D.addEstimate(sample_id, 'BFGS', abg_est, finish-start) 
#            
#            start = time.clock()
#    #        abg_est = FortetEstimator(binnedTrain, abg_init)
#            time.sleep(rand())
#            finish = time.clock()
#            D.addEstimate(sample_id, 'Fortet', abg_est, finish-start) 
   

def writeManual():
    '''generate fake data'''
    dx = .25; xmin = -.25;
    dt = .25; Tf = 1.0
    xs = arange(xmin, 1.0+dx/2., dx);
    ts = arange(0.0, Tf + dt/2., dt);
    thetas = array([1,2])
    Fs = tile(xs, (len(ts), 1)) + tile(reshape(ts, (len(ts), 1)), (1, len(xs)))
    tFs = empty((2,) + Fs.shape)
    for idx in xrange(len(thetas)):
        tFs[idx,:,:] = thetas[idx]*Fs
        
    alphas = exp(ts);
    
    hd5Solution([.0, 1.], dx, xmin, dt, Tf,
                 Fs, alphas, thetas).save('manual write')
 
def loadManual():
    name = 'manual write'
    '''generate fake data'''
    hd5Sol = hd5Solution.load(name);
    
    print hd5Sol._mu,hd5Sol._dx, hd5Sol._Tf
    print hd5Sol._Fs;
    print hd5Sol._alphas;
    print hd5Sol._thetas;
    
def load_mu_beta_Tf():
    pass
 
    
#    N_phi = 20;
#    print 'N_phi = ', N_phi
#    phis =  linspace(1/(2.*N_phi), 1. - 1/ (2.*N_phi), N_phi)
#    
#    h5file = openFile("manual_write.h5", mode = "w", title = "Manually Write Estimate Data")
#    grp = h5file.createGroup("/", 'Estimates', "Estimates INformation")
#    
#    base_name = 'sinusoidal_spike_train_T='
#    for regime_name, T_sim, T_thresh in zip(['superT', 'subT', 'crit', 'superSin'],
#                                                       [5000 , 20000, 5000, 5000],
#                                                       [4., 32, 16., 16.]): 
#        
#        estTable = h5file.createTable(grp, regime_name, Estimate , "Regime Estimate")
##        sampleTbl = h5file.createTable(grp, regime_name, Estimate , "Regime Estimate")
#        
#        regime_label = base_name + str(T_sim)+ '_' + regime_name
#            
#        for sample_id in xrange(1,3):
#            file_name = regime_label + '_' + str(sample_id) + '.path'
#            print file_name
#            
#            binnedTrain = BinnedSpikeTrain.initFromFile(file_name, phis)
#            ps = binnedTrain._Path._params
#            abg_true = array((ps._alpha, ps._beta, ps._gamma))
#            if 1 == sample_id:
#                print 'abg_true = ', abg_true
#            
#            phi_omit = None
#            binnedTrain.pruneBins(phi_omit, N_thresh = 64, T_thresh=T_thresh)
#            print 'N_bins = ', len(binnedTrain.bins.keys())
#            
#            Tf = binnedTrain.getTf()
#            print 'Tf = ', Tf
#        
#            #Estimate:
#            estimate = estTable.row
#            
#            start = time.clock()
#            abg_init = initialize5(binnedTrain)
#            finish = time.clock()
#            estimate['method'] = 'Initializer'
#            estimate['sample_id'] = sample_id
#            estimate['alpha'] = abg_init[0]
#            estimate['beta'] = abg_init[1]
#            estimate['gamma'] = abg_init[2]
#            estimate['walltime'] = finish - start
#            estimate.append()
#                                
#            abg_est = abg_init
#            start = time.clock()
#    #        abg_est = NMEstimator(S, binnedTrain, abg_init)
#            time.sleep(rand())
#            print 'Est. time = ', time.clock() - start
#            print 'abg_est = ', abg_est
#                   
#            start = time.clock()
#    #        abg_est = FortetEstimator(binnedTrain, abg_init)
#            time.sleep(2*rand())
#            print 'Est. time = ', time.clock() - start
#            print 'abg_est = ', abg_est
#            
#        estTable.flush()
#         
#    print '#'*44
#    print h5file
#    h5file.close()
    
#def readWithAnalyzer():
#    analyzer = DataAnalyzer('FvsWF_4x16');
#    
#    print analyzer.getEstimates([1,3], 'superSin', 'QuadFortet')
#    print analyzer.getAllEstimates('subT', 'QuadFortet')
#    print analyzer.getEstimates(2, 'crit', 'QuadFortet')
#    print analyzer.getEstimates(2, 'crit', 'Fortet')
#    
#    print analyzer.getTrueParamValues('subT')
#    print analyzer.getSampleCount('crit')
#    
#    print len(analyzer.getAllEstimates('superSin', 'QuadFortet'))
      
    
if __name__ == '__main__':
    from pylab import *
    
#    initializeTable()
#    writeManual()

    loadManual()
    
#    writeWithHarvester()

#    readWithAnalyzer()
#    flagWarnings()

    