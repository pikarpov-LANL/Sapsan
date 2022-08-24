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
#from st_state_patch import SessionState

import torch
import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

#uncomment if cloned from github!
sys.path.append(str(Path.home())+"/Sapsan/")

from sapsan.lib.backends import FakeBackend, MLflowBackend
from sapsan.lib.data import HDF5Dataset, EquidistantSampling, flatten
from sapsan import Train, Evaluate
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3d, CNN3dConfig
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3dModel as model
from sapsan.utils.plot import model_graph, pdf_plot, cdf_plot, slice_plot, plot_params

st.set_page_config(
    page_title="Examples",
    page_icon="🚅",
)

#initialization of defaults
cf = configparser.RawConfigParser()
widget_values = {}

st.title('Sapsan Configuration')
st.write('This demo is meant to present capabilities of Sapsan. You can configure each part of the experiment at the sidebar. Once you are done, you can see the summary of your runtime parameters under _Show configuration_. In addition you can review the model that is being used (in the custom setup, you will also be able to edit it). Lastly click the _Run experiment_ button to train the test the ML model.')    

st.sidebar.markdown("**General Configuration**")

try:
    cf.read('temp.txt')
    temp = dict(cf.items('config'))
except: pass

def run_experiment():
    '''
    The interface to setup the estimator, configuration, data loading, etc. is
    nearly identical to a Jupyter Notebook interface for Sapsan. In an ideal case,
    this is the only function you need to edit to set up your own GUI demo.
    '''

    if widget_values['backend_selection'] == 'Fake':
        tracking_backend = FakeBackend(widget_values['experiment name'])

    elif widget_values['backend_selection'] == 'MLflow':
        tracking_backend = MLflowBackend(widget_values['experiment name'], 
        widget_values['mlflow_host'],widget_values['mlflow_port'])

    #Load the data 
    x, y, data_loader = load_data(widget_values['checkpoints'])
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

    thread = Thread(target=show_log, args=(progress_slot, epoch_slot,
                                           trainig_bar, widget_values['n_epochs']))
    add_script_run_ctx(thread)
    thread.start()

    start = time.time()
    #Train the model
    trained_estimator = training_experiment.run()

    st.write('Trained in %.2f sec'%((time.time()-start)))
    st.success('Done! Plotting...')

    #--- Test the model ---
    #Load the test data
    x, y, data_loader = load_data(widget_values['checkpoint_test'])
    loaders = data_loader.convert_to_torch([x, y])

    #Set the test experiment
    trained_estimator.loaders = loaders
    evaluation_experiment = Evaluate(backend = tracking_backend,
                                     model = trained_estimator,
                                     data_parameters = data_loader)

    #Test the model
    results = evaluation_experiment.run()

    #Plots metrics
    mpl.rcParams.update(plot_params())

    fig = plt.figure(figsize=(12,6), dpi=60)
    (ax1, ax2) = fig.subplots(1,2)

    pdf_plot([results['predict'], results['target']], label=['prediction', 'target'], ax=ax1)
    cdf_plot([results['predict'], results['target']], label=['prediction', 'target'], ax=ax2)
    plot_static()

    plot_label = ['predict','target']
    plot_series = []       
    outdata = evaluation_experiment.split_batch(results['predict'])
    for key, value in outdata.items():
        if key in plot_label:
            plot_series.append(value)
    slices = slice_plot(plot_series, label=plot_label, cmap='viridis')
    st.pyplot(plt)    
    
def define_estimator(loaders):
    estimator = CNN3d(config=CNN3dConfig(n_epochs=int(widget_values['n_epochs']), 
                                         patience=int(widget_values['patience']), 
                                         min_delta=float(widget_values['min_delta'])),
                      loaders=loaders)    
    
    return estimator
    
def load_data(checkpoints):
    #Load the data      
    features = widget_values['features'].split(',')
    features = [i.strip() for i in features]

    target = widget_values['target'].split(',')
    target = [i.strip() for i in target]     

    checkpoints = np.array([int(i) for i in checkpoints.split(',')])

    data_loader = HDF5Dataset(path=widget_values['path'],
                              features=features,
                              target=target,
                              checkpoints=checkpoints,
                              batch_size=text_to_list(widget_values['batch_size']),
                              input_size=text_to_list(widget_values['input_size']),
                              sampler=sampler,
                              shuffle = False)
    x, y = data_loader.load_numpy()
    return x, y, data_loader    


def show_log(progress_slot, epoch_slot, training_bar, n_epochs):        
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
            epoch_slot.markdown('Epoch:$~$**%d**/**%d** $~~~~~$ Train Loss:$~$**%.4e**'%(current_epoch,n_epochs,train_loss))
            plot_data['epoch'] = data[:, 0]               
            plot_data['train_loss'] = data[:, 1]
            df = pd.DataFrame(plot_data)

            if len(plot_data['epoch']) == 1:
                plotting_routine = px.scatter
            else:
                plotting_routine = px.line

            training_bar.progress(len(plot_data['epoch'])/n_epochs)
            fig = plotting_routine(df, x="epoch", y="train_loss", log_y=True,
                          title='Training Progress', width=700, height=400)
            fig.update_layout(yaxis=dict(exponentformat='e'))
            fig.layout.hovermode = 'x'
            progress_slot.plotly_chart(fig)

            last_epoch = current_epoch            

        if current_epoch == widget_values['n_epochs']: 
            return

        time.sleep(0.1) 


def plot_static():
    buf = BytesIO()
    plt.savefig(buf, format="png",  dpi=50)
    st.image(buf) 

# ----- Backend Widget Functions -----
def make_recording_widget(f):
    """
    Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """
    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper

def load_config(config_file):
    cf.read(config_file)
    config = dict(cf.items('sapsan_config'))
    return config

def selectbox_params():
    widget_values['backend_list'] = ['Fake', 'MLflow']
    widget_values['backend_selection_index'] = widget_values['backend_list'].index(widget_values['backend_selection'])

def text_to_list(value):
    to_clean = ['(', ')', '[', ']', ' ']
    for i in to_clean: value = value.translate({ord(i) : None})
    value = list([int(i) for i in value.split(',')])
    return value


#--- Load Default ---    
#state = SessionState()   
#button = make_recording_widget(st.sidebar.button)
number = make_recording_widget(st.sidebar.number_input)
number_main = make_recording_widget(st.number_input)
text = make_recording_widget(st.sidebar.text_input)
text_main = make_recording_widget(st.text_input)
checkbox = make_recording_widget(st.sidebar.checkbox)
selectbox = make_recording_widget(st.sidebar.selectbox)

config_file = st.sidebar.text_input('Configuration File', "config.txt", type='default')

if st.sidebar.button('reload config'):
    #st.caching.clear_cache()
    config = load_config(config_file)

    for key, value in config.items():
        widget_values[key+'_default'] = value
        widget_values[key] = value
        widget_values[key+'flag'] = None

    selectbox_params()                
    st.sidebar.text('... loaded config %s'%config_file)

else:
    config = load_config(config_file)
    for key, value in config.items():
        if key in widget_values: pass
        else: widget_values[key] = value
    selectbox_params()

st.sidebar.text('>Collapse all sidebar pars to reset<')

widget_values['experiment name'] = st.sidebar.text_input(label='Experiment Name',value=config['experiment name'])

with st.sidebar.expander('Backend'):
    widget_values['backend_selection'] = st.selectbox(
        'What backend to use?',
        widget_values['backend_list'], index=widget_values['backend_selection_index'])
    
    if widget_values['backend_selection'] == 'MLflow':
        widget_values['mlflow_host'] = st.text_input(label='MLflow host',value=config['mlflow_host'])
        widget_values['mlflow_port'] = st.number_input(label='MLflow port',value=int(config['mlflow_port']),
                                                       min_value=1024,max_value=65535,format='%d')

with st.sidebar.expander('Data: train'):
    widget_values['path'] = st.text_input(label='Path',value=config['path'])
    widget_values['checkpoints'] = st.text_input(label='Checkpoints',value=config['checkpoints'])
    widget_values['features'] = st.text_input(label='Input features',value=config['features'])
    widget_values['target'] = st.text_input(label='Input target',value=config['target'])
    widget_values['input_size'] = st.text_input(label='Input Size',value=config['input_size'])
    widget_values['sample_to'] = st.text_input(label='Sample to size',value=config['sample_to'])
    widget_values['batch_size'] = st.text_input(label='Batch Size',value=config['batch_size'])

with st.sidebar.expander('Data: test'):
    widget_values['checkpoint_test'] = st.text_input(label='Checkpoint: test',value=config['checkpoint_test'])    

with st.sidebar.expander('Model'):
    widget_values['n_epochs'] = st.number_input(label='# of Epochs',value=int(config['n_epochs']), 
                                                min_value=1,format='%d')
    
    widget_values['patience'] = st.number_input(label='Patience',value=int(config['patience']), 
                                                min_value=0,format='%d')
    widget_values['min_delta'] = st.number_input(label='Min Delta',value=float(config['min_delta']), 
                                                step=float(config['min_delta'])*0.5,format='%.1e')    

#sampler_selection = st.sidebar.selectbox('What sampler to use?', ('Equidistant3D', ''), )
if widget_values['sampler_selection'] == "Equidistant3D":
    sampler = EquidistantSampling(text_to_list(widget_values['sample_to']))    

show_config = [
    ['experiment name', widget_values["experiment name"]],
    ['data path', widget_values['path']],
    ['checkpoints', widget_values['checkpoints']],
    ['features', widget_values['features']],
    ['target', widget_values['target']],
    ['Reduce each dimension to', widget_values['sample_to']],
    ['Batch size per dimension', widget_values['batch_size']],
    ['number of epochs', widget_values['n_epochs']],
    ['patience', widget_values['patience']],
    ['min_delta', widget_values['min_delta']],
    ['backend_selection', widget_values['backend_selection']],
    ['checkpoint: test', widget_values['checkpoint_test']],
    ]

if widget_values['backend_selection']=='MLflow': 
    show_config.append(['mlflow_host', widget_values['mlflow_host']])
    show_config.append(['mlflow_port', widget_values['mlflow_port']])

with st.expander("Show configuration"):    
    st.table(pd.DataFrame(show_config, columns=["key", "value"]))

with st.expander("Show model graph"):    
    st.write('Please load the data first or enter the data shape manualy, comma separated.')
    widget_values['Data Shape'] = st.text_input(label='Data Shape', value='16,1,8,8,8')

    shape = widget_values['Data Shape']
    shape = np.array([int(i) for i in shape.split(',')])
    shape[1] = 1

    #Load the data  
    if st.button('Load Data'):
        x, y, data_loader = load_data(widget_values['checkpoints'])
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

    widget_values['edit_port'] = st.number_input(label='Edit Port',value=8601,min_value=1024,max_value=65535,format='%d')    

    if st.button('Edit'):
        estimator_path = inspect.getsourcefile(model)
        st.info(f'Location: {estimator_path}')
        
        os.system(f"jupyter notebook {estimator_path} --NotebookApp.password='' --NotebookApp.token='' --no-browser --port={widget_values['edit_port']:n} &")
        webbrowser.open(f"http://localhost:{widget_values['edit_port']:n}/edit/cnn3d_estimator.py", new=2)

st.markdown("---")

if st.button("Run experiment"):
    start = time.time()

    log_path = './logs/log.txt'
    if os.path.exists(log_path): 
        os.remove(log_path)
    else: pass        

    run_experiment()

    st.write('Finished in %.2f sec'%((time.time()-start))) 

if widget_values['backend_selection'] == 'MLflow':
    if st.button("MLflow tracking"):
        webbrowser.open('http://%s:%s'%(widget_values['mlflow_host'], widget_values['mlflow_port']), new=2)


with open('temp.txt', 'w') as file:
    file.write('[config]\n')
    for key, value in widget_values.items():
        file.write('%s = %s\n'%(key, value))