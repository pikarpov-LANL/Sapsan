from typing import List, Optional

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

