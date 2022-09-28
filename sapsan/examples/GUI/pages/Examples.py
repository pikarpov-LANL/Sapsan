import os
import sys
import inspect
import time
import json
import signal
import inspect
import numpy as np
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import configparser
import webbrowser
from io import BytesIO
from threading import Thread

import torch
import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

#uncomment if cloned from github!
sys.path.append(str(Path.home())+"/Sapsan/")

from sapsan.lib.backends import FakeBackend, MLflowBackend
from sapsan.lib.data import HDF5Dataset, EquidistantSampling, flatten
from sapsan.lib import Train, Evaluate
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3d, CNN3dConfig
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3dModel as model
from sapsan.utils.plot import model_graph, pdf_plot, cdf_plot, slice_plot, plot_params

st.set_page_config(
    page_title="Examples",
    page_icon="🚅",
)

#initialization of defaults
cf = configparser.RawConfigParser()

st.title('Sapsan Configuration')
st.write('This demo is meant to present capabilities of Sapsan. You can configure each part of the experiment at the sidebar. Once ready, you can see the summary of the runtime parameters under _Show configuration_. In addition you can review the model that is being used (in the custom setup, you will also be able to edit it). Lastly click the _Run experiment_ button to train the test the ML model.')    

st.sidebar.markdown("**General Configuration**")

def run_experiment():
    '''
    The interface to setup the estimator, configuration, data loading, etc. is
    nearly identical to a Jupyter Notebook interface for Sapsan. In an ideal case,
    this is the only function you need to edit to set up your own GUI demo.
    '''

    if st.session_state.backend_selection == 'Fake':
        tracking_backend = FakeBackend(st.session_state.experiment_name)

    elif st.session_state.backend_selection == 'MLflow':
        tracking_backend = MLflowBackend(st.session_state.experiment_name, 
        st.session_state.mlflow_host, st.session_state.mlflow_port)

    #Load the data 
    x, y, data_loader = load_data(st.session_state.checkpoints)
    y = flatten(y)
    shape = x.shape
    loaders = data_loader.convert_to_torch([x, y])

    st.write("Dataset loaded...")

    estimator = define_estimator(loaders)
    #graph = model_graph(estimator.model, shape)
    #st.graphviz_chart(graph.build_dot())

    #Set the experiment
    training_experiment = Train(backend=tracking_backend,
                                model=estimator,
                                data_parameters = data_loader,
                                show_log = False)

    #Plot progress            
    progress_slot = st.empty()
    epoch_slot = st.empty()
    trainig_bar = st.progress(0)       

    thread = Thread(target=show_log, args=(progress_slot, epoch_slot,trainig_bar))
    add_script_run_ctx(thread)
    thread.start()

    start = time.time()
    #Train the model
    trained_estimator = training_experiment.run()
    
    st.success(f'Finished training in {(time.time()-start):.2f} s!')

    #--- Test the model ---
    #Load the test data
    x, y, data_loader = load_data(st.session_state.checkpoint_test)
    loaders = data_loader.convert_to_torch([x, y])

    #Set the test experiment
    trained_estimator.loaders = loaders
    evaluation_experiment = Evaluate(backend = tracking_backend,
                                     model = trained_estimator,
                                     data_parameters = data_loader)

    #Test the model
    results = evaluation_experiment.run()

    #Plots metrics
    pdf_cdf_slot = st.empty()
    slice_slot = st.empty()

    pdf_cdf_slot.markdown('Plotting PDF & CDF ...')

    mpl.rcParams.update(plot_params())

    fig = plt.figure(figsize=(14,6), dpi=60)
    (ax0, ax1) = fig.subplots(1,2)
    
    pdf_plot([results['predict'], results['target']], label=['prediction', 'target'], ax=ax0)
    cdf_plot([results['predict'], results['target']], label=['prediction', 'target'], ax=ax1)

    plt.subplots_adjust(left=0.07,right=0.9,wspace=0.3)
    plot_static(pdf_cdf_slot)
    #pdf_cdf_slot.pyplot(fig)

    slice_slot.markdown('Plotting spatial distributions ...')

    plot_label = ['predict','target']
    plot_series = []       
    outdata = evaluation_experiment.split_batch(results['predict'])
    for key, value in outdata.items():
        if key in plot_label:
            plot_series.append(value)
            
    slice_plot(plot_series, label=plot_label, cmap='viridis')
    slice_slot.pyplot(plt)    
        
def define_estimator(loaders):
    estimator = CNN3d(config=CNN3dConfig(n_epochs=st.session_state.n_epochs, 
                                         patience=st.session_state.patience,
                                         min_delta=st.session_state.min_delta),
                      loaders=loaders)    
    
    return estimator
    
def load_data(checkpoints):
    #Load the data      
    features = st.session_state.features.split(',')
    features = [i.strip() for i in features]

    target = st.session_state.target.split(',')
    target = [i.strip() for i in target]     

    data_loader = HDF5Dataset(path=st.session_state.path,
                              features=features,
                              target=target,
                              checkpoints=text_to_list(st.session_state.checkpoints),
                              batch_size=text_to_list(st.session_state.batch_size),
                              input_size=text_to_list(st.session_state.input_size),
                              sampler=sampler,
                              shuffle = False)
    x, y = data_loader.load_numpy()
    return x, y, data_loader    


def show_log(progress_slot, epoch_slot, training_bar):        
    '''
    Show loss vs epoch progress with plotly, dynamically.
    The plot will be updated every 0.1 second
    '''
    log_path = 'logs/logs/train.csv'

    if os.path.exists(log_path):
        os.remove(log_path)

    log_exists = False            
    while log_exists == False:            
        if os.path.exists(log_path):
            log_exists = True            
        time.sleep(0.1)

    plot_data = {'epoch':[], 'train_loss':[]}
    last_epoch = 0
    running = True

    while running:
        data = np.genfromtxt(log_path, delimiter=',', 
                             skip_header=1, dtype=np.float32)

        if len(data.shape)==1: data = np.array([data])

        current_epoch = data[-1, 0]
        train_loss = data[-1, 1]

        if current_epoch == last_epoch:
            pass
        else:     
            epoch_slot.markdown('Epoch:$~$**%d**/**%d** $~~~~~$ Train Loss:$~$**%.4e**'%(current_epoch,st.session_state.n_epochs,train_loss))
            plot_data['epoch'] = data[:, 0]               
            plot_data['train_loss'] = data[:, 1]
            df = pd.DataFrame(plot_data)

            if len(plot_data['epoch']) == 1:
                plotting_routine = px.scatter
            else:
                plotting_routine = px.line

            training_bar.progress(len(plot_data['epoch'])/st.session_state.n_epochs)
            fig = plotting_routine(df, x="epoch", y="train_loss", log_y=True,
                          title='Training Progress', width=700, height=400)
            fig.update_layout(yaxis=dict(exponentformat='e'))
            fig.layout.hovermode = 'x'
            progress_slot.plotly_chart(fig)

            last_epoch = current_epoch            

        if current_epoch == st.session_state.n_epochs: 
            return

        time.sleep(0.1)

def plot_static(slot):
    buf = BytesIO()
    plt.savefig(buf, format="png",  dpi=50)
    slot.image(buf)          

# ----- Backend Widget Functions -----

def load_config(config_file):
    cf.read(config_file)
    config = OrderedDict(cf.items('sapsan_config'))
    str_type = ['path','checkpoints','features','target','input_size','sample_to','batch_size','checkpoint_test']
    for key, value in config.items():
        if key not in str_type:
            try: value = int(value)
            except: 
                try: value = float(value)
                except: pass
        config[key]=value
    return config

def selectbox_params():
    st.session_state.backend_list = ['Fake', 'MLflow']
    st.session_state.backend_selection_index = st.session_state.backend_list.index(st.session_state.backend_selection)

def text_to_list(value):
    to_clean = ['(', ')', '[', ']', ' ']
    for i in to_clean: value = value.translate({ord(i) : None})
    value = list([int(i) for i in value.split(',')])
    return value

#--------- Load Defaults ---------
 
config_file = st.sidebar.text_input('Configuration File', "config.txt", type='default')

if st.sidebar.button('Reload Config'):
    config = load_config(config_file)
    for key, value in config.items():
        setattr(st.session_state, key, value)      
else: 
    config = load_config(config_file)
    for key, value in config.items():
        if key in st.session_state.keys(): pass
        else: setattr(st.session_state, key, value)    
    selectbox_params()     

#------- Define All Widgets -------    
        
st.sidebar.text_input(label='Experiment Name',value=config['experiment_name'],key='experiment_name')

with st.sidebar.expander('Backend'):
    st.selectbox(
        label='What backend to use?',key='backend_selection',
        options=st.session_state.backend_list, index=st.session_state.backend_selection_index)
    
    if st.session_state.backend_selection == 'MLflow':
        st.text_input(label='MLflow host',value=config['mlflow_host'],key='mlflow_host')
        st.number_input(label='MLflow port',value=config['mlflow_port'],key='mlflow_port',
                        min_value=1024,max_value=65535,format='%d')

with st.sidebar.expander('Data: train'):
    st.text_input(label='Path',value=config['path'],key='path')
    st.text_input(label='Checkpoints',value=config['checkpoints'],key='checkpoints')
    st.text_input(label='Input features',value=config['features'],key='features')
    st.text_input(label='Input target',value=config['target'],key='target')
    st.text_input(label='Input Size',value=config['input_size'],key='input_size')
    st.text_input(label='Sample to size',value=config['sample_to'],key='sample_to')
    st.text_input(label='Batch Size',value=config['batch_size'],key='batch_size')

with st.sidebar.expander('Data: test'):
    st.text_input(label='Checkpoint: test',value=config['checkpoint_test'],key='checkpoint_test')    
    
with st.sidebar.expander('Model'):
    st.number_input(label='# of Epochs',value=config['n_epochs'],key='n_epochs',min_value=1,format='%d')
    st.number_input(label='Patience',value=config['patience'],key='patience',min_value=0,step=1,format='%d')    
    st.number_input(label='Min Delta',value=config['min_delta'],key='min_delta',step=config['min_delta']*0.5,format='%.1e')

#sampler_selection = st.sidebar.selectbox('What sampler to use?', ('Equidistant3D', ''), )
if st.session_state.sampler_selection == "Equidistant3D":
    sampler = EquidistantSampling(text_to_list(st.session_state.sample_to))    

show_config = [
    ['experiment name', st.session_state.experiment_name],
    ['data path', st.session_state.path],
    ['features', st.session_state.features],
    ['target', st.session_state.target],
    ['checkpoints', st.session_state.checkpoints],
    ['checkpoint: test', st.session_state.checkpoint_test],        
    ['Reduce each dimension to', st.session_state.sample_to],
    ['Batch size per dimension', st.session_state.batch_size],
    ['number of epochs', st.session_state.n_epochs],    
    ['patience', st.session_state.patience],    
    ['min_delta', st.session_state.min_delta],
    ['backend_selection', st.session_state.backend_selection],    
    ]

if st.session_state.backend_selection=='MLflow': 
    show_config.append(['mlflow_host', st.session_state.mlflow_host])
    show_config.append(['mlflow_port', st.session_state.mlflow_port])

with st.expander("Show configuration"):    
    st.table(pd.DataFrame(show_config, columns=["Key", "Value"]))

with st.expander("Show model graph"):    
    st.write('Please load the data first or enter the data shape manualy, comma separated.')
    st.text_input(label='Data Shape',key='data_shape',value='16,1,8,8,8')

    shape = st.session_state.data_shape
    shape = np.array([int(i) for i in shape.split(',')])
    shape[1] = 1

    #Load the data  
    if st.button('Load Data'):
        x, y, data_loader = load_data(st.session_state.checkpoints)
        y = flatten(y)
        loaders = data_loader.convert_to_torch([x, y])

        shape = x.shape

        #try:
        estimator = define_estimator(loaders)

    #graph = model_graph(estimator.model, shape)
    #st.graphviz_chart(graph.build_dot())
    #except: st.error('ValueError: Incorrect data shape, please edit the shape or load the data.')

with st.expander("Show model code"):            
    st.code(inspect.getsource(model), language='python')    

    st.number_input(label='Edit Port',key='edit_port',value=config['edit_port'],min_value=1024,max_value=65535,format='%d')    

    if st.button('Edit'):
        estimator_path = inspect.getsourcefile(model)
        st.info(f'Location: {estimator_path}')
        
        os.system(f"jupyter notebook {estimator_path} --NotebookApp.password='' --NotebookApp.token='' --no-browser --port={st.session_state.edit_port:n} &")
        webbrowser.open(f"http://localhost:{st.session_state.edit_port:n}/edit/cnn3d_estimator.py", new=2)

st.markdown("---")
if st.button("Run experiment"):
    start = time.time()

    log_path = './logs/log.txt'
    if os.path.exists(log_path): 
        os.remove(log_path)
    else: pass        

    run_experiment()

    st.success(f'Total runtime: {(time.time()-start):.2f} s')

if st.session_state.backend_selection == 'MLflow':
    if st.button("MLflow tracking"):
        webbrowser.open('http://%s:%s'%(st.session_state.mlflow_host, st.session_state.mlflow_port), new=2)

st.sidebar.markdown("---")

with st.sidebar.expander('Super Secret'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('❄️'):
            st.snow()
    with col2:
        if st.button('🎈'):
            st.balloons()