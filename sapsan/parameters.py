class ParameterHandler:
    def __init__(self):    
        self.name='test'
        self.parameters=None
        self.target=None
        self.path=None
        self.dataset='iso'
        self.savepath='Figures'
        self.alpha=None
        self.gamma=None
        self.dim=128
        self.max_dim=512
        self.step=None
        self.ttrain=[0]
        self.dn=1 
        self.dt=1
        self.fm=15 
        self.filtname='spectral'
        self.targetComp=0
        self.axis=2
        self.from3D=False
        self.outliers=False
        self.dataType = 'h5' 
        self.parLabel=None
        self.targetLabel=None
        self.n_epochs=100 
        self.batch_size=12
        self.cube_size=8 
        self.method='krr'
        self.experiment_name='demo'
        self.train_fraction = 1
    
    def LoadParameters(self, pars):
        print('Running parameters: ', pars)

        #import parameters
        for key in pars.keys():
            setattr(self, key, pars[key])

            #>>>make sure the savepaths are generalized and input<<<
            #need to adjust savefile path for correct formatting in my case
            if key=='savepath' and self.savepath!=None:
                self.savepath = self.savepath+'%s/'%self.name
        
        return self.__dict__

parameters = ParameterHandler()

