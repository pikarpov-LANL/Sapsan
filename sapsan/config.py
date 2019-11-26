from sapsan.Sapsan import Sapsan

pars = {'name':'16','ttrain':[0]}


pars['dim']=16
pars['parameters']=['u_dim128_fm15', 'b_dim128_fm15', 'a_dim128_fm15', 'du0_dim128_fm15', 
                    'du1_dim128_fm15', 'du2_dim128_fm15', 'db0_dim128_fm15', 'db1_dim128_fm15', 'db2_dim128_fm15',
                    'da0_dim128_fm15', 'da1_dim128_fm15', 'da2_dim128_fm15']
pars['target']='tn_dim128_fm15'
pars['fm']=10
pars['targetComp']=2
pars['alpha']=0.001
pars['gamma']=1.778
pars['max_dim']=128
#pars['step']=2
pars['dataset']='mhd'
pars['axis']=3
pars['from3D']=True
pars['savepath']='Figures/'
pars['dt']=2.5e-3
pars['dataType']='h5' #supports hdf5 h5 dat and txt; default h5
pars['path']='/raid1/JHTDB/mhd/max_1024/mhd128_t@.4/'
pars['method'] = "cnn"
pars['experiment_name'] = 'trial'

sps = Sapsan()
model = sps.fit(pars)

#sps.test(model, ttest=[1])


class SapsanConfig(object):
    def __init__(
            self,
            dim: int,
            parameters: list, # TODO: rename to features or something like that :)
            target: str,
            filter_modes: int,
            target_comp: int,
            alpha: float,
            gamma: float,
            max_dim: int,
            dataset: str,
            axis: int,
            from_3d: bool,
            savepath: str,
            dt: float,
            data_type: str,
            path: str,
            method: str,
            experiment_name: str
    ):
        """
        :param dim: dimension of the training set
        :param parameters: parameters (e.g. features) to train on
        :param target: target parameter to train against and to predict
        :param filter_modes: number of modes to filter down to
        :param target_comp:
        :param alpha: will use the default KRR alpha
        :param gamma: will use the default KRR gamma
        :param max_dim: initial dimension of the data file, from which training and testing sets will be extracted
        :param dataset: which data set to use; options are 'iso' for HD turbulence, and 'mhd' for MHD turbulence
        :param axis: dimensionality; either 2 or 3
        :param from_3d: extracts 2D slice from 3D data; only relevant if axis=2
        :param savepath: where to save the data
        :param dt: numeral value for dt to calculate the actual time of each timestep
        :param data_type:
        :param path: path to the data
        :param method: method to use for training; either 'cnn' or 'krr'
        :param experiment_name: name of the experiment for mlflow tracking
        """
        self.dim = dim
        self.parameters = parameters
        self.target = target
        self.filter_modes = filter_modes
        self.target_comp = target_comp
        self.alpha = alpha
        self.gamma = gamma
        self.max_dim = max_dim
        self.dataset = dataset
        self.axis = axis
        self.from_3d = from_3d
        self.savepath = savepath
        self.dt = dt
        self.data_type = data_type
        self.path = path
        self.method = method
        self.experiment_name = experiment_name

