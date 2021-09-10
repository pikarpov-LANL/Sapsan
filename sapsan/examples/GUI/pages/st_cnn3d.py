import os
import sys
import inspect
import time
import json
import signal
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
from streamlit.report_thread import add_report_ctx

#uncomment if cloned from github!
sys.path.append(str(Path.home())+"/Sapsan/")

from sapsan import Train, Evaluate 
from sapsan.lib.backends import FakeBackend, MLflowBackend
from sapsan.lib.data import HDF5Dataset, EquidistantSampling, flatten, torch_splitter
from sapsan.utils.plot import model_graph, pdf_plot, cdf_plot, slice_plot, plot_params

# Load your Estimator, EstimatorConfig, and EstimatorModel
from sapsan import CNN3d as Estimator
from sapsan import CNN3dConfig as EstimatorConfig
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3dModel as EstimatorModel

#initialization of defaults
cf = configparser.RawConfigParser()
widget_values = {}

           
def cnn3d():
    st.title('Sapsan Configuration')
    st.write('This demo is meant to present capabilities of Sapsan. You can configure each part of the experiment at the sidebar. Once you are done, you can see the summary of your runtime parameters under _Show configuration_. In addition you can review the model that is being used (in the custom setup, you will also be able to edit it). Lastly click the _Run experiment_ button to train the test the ML model.')    
    
    st.sidebar.markdown("General Configuration")
    
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
        data_loader = load_data(widget_values['checkpoints'])
        x, y = data_loader.load_numpy()
        y = flatten(y)
        loaders = data_loader.convert_to_torch([x, y])
        
        st.write("Dataset loaded...")
        
        estim = Estimator(config=EstimatorConfig(n_epochs=int(widget_values['n_epochs']), 
                                                 patience=int(widget_values['patience']), 
                                                 min_delta=float(widget_values['min_delta'])),
                          loaders=loaders)
        
        #Set the experiment
        training_experiment = Train(backend=tracking_backend,
                                    model=estim,
                                    data_parameters = data_loader,
                                    show_log = False)
        
        #Plot progress        
        progress_slot = st.empty()
        epoch_slot = st.empty()
        
        thread = Thread(target=show_log, args=(progress_slot, epoch_slot))
        add_report_ctx(thread)
        thread.start()
        
        start = time.time()
        #Train the model
        trained_estimator = training_experiment.run()

        st.write('Trained in %.2f sec'%((time.time()-start)))
        st.success('Done! Plotting...')

        #--- Test the model ---
        #Load the test data
        data_loader = load_data(widget_values['checkpoint_test'])
        x, y = data_loader.load_numpy()
        loaders = data_loader.convert_to_torch([x, y])

        #Set the test experiment
        trained_estimator.loaders = loaders
        evaluation_experiment = Evaluate(backend = tracking_backend,
                                         model = trained_estimator,
                                         data_parameters = data_loader)
        
        #Test the model
        cubes = evaluation_experiment.run()
        
        #Plot PDF, CDF, and slices
        #Similar setup to replot from sapsan.Evaluate()
        mpl.rcParams.update(plot_params())
        
        fig = plt.figure(figsize=(12,6), dpi=60)
        (ax1, ax2) = fig.subplots(1,2)
        
        pdf_plot([cubes['pred_cube'], cubes['target_cube']], 
                 names=['prediction', 'target'], ax=ax1)
        cdf_plot([cubes['pred_cube'], cubes['target_cube']], 
                 names=['prediction', 'target'], ax=ax2)
        plot_static()
        
        slices_cubes = evaluation_experiment.split_batch(cubes['pred_cube'])
        slice_plot([slices_cubes['pred_slice'], slices_cubes['target_slice']], 
                   names=['prediction', 'target'], cmap=evaluation_experiment.cmap)
        st.pyplot(plt)    
    
    
    def load_data(checkpoints):
        #Load the data      
        features = widget_values['features'].split(',')
        features = [i.strip() for i in features]
        
        target = widget_values['target'].split(',')
        target = [i.strip() for i in target]     
        
        checkpoints = np.array([int(i) for i in checkpoints.split(',')])
        
        #print('--ST Shapes--', batch_size, input_size)
        data_loader = HDF5Dataset(path=widget_values['path'],
                                  features=features,
                                  target=target,
                                  checkpoints=checkpoints,
                                  batch_size=text_to_list(widget_values['batch_size']),
                                  input_size=text_to_list(widget_values['input_size']),
                                  sampler=sampler,
                                  shuffle = False)
        return data_loader  
    
    
    def show_log(progress_slot, epoch_slot):        
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
                epoch_slot.markdown('Epoch:$~$**%d** $~~~~~$ Train Loss:$~$**%.4e**'%(current_epoch, train_loss))
                plot_data['epoch'] = data[:, 0]
                plot_data['train_loss'] = data[:, 1]
                df = pd.DataFrame(plot_data)
                
                if len(plot_data['epoch']) == 1:
                    plotting_routine = px.scatter
                else:
                    plotting_routine = px.line
                
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
           
    def widget_history_checkbox(title, params):
        if st.sidebar.checkbox(title):
            widget_history_checked(params)
        else:
            widget_history_unchecked(params)
    
    def widget_history_checked(params):
        widget_type = {number:int, number_main:int, text:str, text_main:str, checkbox:bool}
        for i in range(len(params)):
            label = params[i]['label']
            default = params[i]['default']
            widget = params[i]['widget']

            not_widget_params = ['default', 'widget', 'widget_type']
            additional_params = {key:value for key, value in params[i].items() if key not in not_widget_params}
            try:
                if widget_values[label+'_flag'] == True:
                    widget_values[label+'_flag'] = False
                    try:
                        widget_values[label+'_default'] = widget_type[widget](temp[label])
                        widget(value = widget_type[widget](temp[label]), **additional_params)
                    except: widget(value = widget_values[label+'_default'], **additional_params)
                else:
                    widget(value = widget_values[label+'_default'], **additional_params)
            except: 
                widget_values[label+'_flag'] = False
                widget(value = widget_type[widget](default), **additional_params)
    
    def widget_history_unchecked(params):
        widget_type = {number:int, number_main:int, text:str, text_main:str, checkbox:bool}
        for i in range(len(params)):
            label = params[i]['label']
            default = params[i]['default']
            widget = params[i]['widget']
            
            widget_values[label+'_flag'] = True        
            widget_values[label+'_default'] = widget_type[widget](default)
    
    # ------- Widget Backend End ---------
    
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
    
    config_file = st.sidebar.text_input('Configuration file', "st_config.txt", type='default')
        
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
    
    st.sidebar.text('> Collapse all sidebar pars to reset <')
    
    widget_history_checked([{'label':'experiment name', 'default':config['experiment name'], 'widget':text}])

    if st.sidebar.checkbox('Backend', value=False):        
        widget_values['backend_selection'] = selectbox(
            'What backend to use?',
            widget_values['backend_list'], index=widget_values['backend_selection_index'])
        
        widget_values['backend_selection_index'] = widget_values['backend_list'].index(widget_values['backend_selection'])
        
        
        if widget_values['backend_selection'] == 'MLflow':
            widget_history_checked([{'label':'mlflow_host', 'default':config['mlflow_host'], 'widget':text}])
            widget_history_checked([{'label':'mlflow_port', 'default':config['mlflow_port'], 
                                     'widget':number, 'min_value':1024, 'max_value':65535}])
    else:
        widget_history_unchecked([{'label':'mlflow_host', 'default':config['mlflow_host'], 'widget':text}])
        widget_history_unchecked([{'label':'mlflow_port', 'default':config['mlflow_port'], 'widget':number,
                                                                    'min_value':1024, 'max_value':65535}]) 

    
    widget_history_checkbox('Data: train',[{'label':'path', 'default':config['path'], 'widget':text},
                                    {'label':'checkpoints', 'default':config['checkpoints'],'widget':text},
                                    {'label':'features', 'default':config['features'], 'widget':text},
                                    {'label':'target', 'default':config['target'], 'widget':text},
                                    {'label':'input_size', 'default':config['input_size'], 'widget':text},
                                    {'label':'sample_to', 'default':config['sample_to'], 'widget':text},
                                    {'label':'batch_size', 'default':config['batch_size'], 'widget':text}])
    
    widget_history_checkbox('Data: test',[{'label':'checkpoint_test', 
                                           'default':config['checkpoint_test'],'widget':text}])
    

        
    widget_history_checkbox('Model',[{'label':'n_epochs', 
                                      'default':config['n_epochs'], 'widget':number, 'min_value':1},
                                     {'label':'patience', 'default':config['patience'], 'widget':number, 'min_value':0},
                                     {'label':'min_delta', 'default':config['min_delta'], 'widget':text}])  

    #sampler_selection = st.sidebar.selectbox('What sampler to use?', ('Equidistant3D', ''), )
    if widget_values['sampler_selection'] == "Equidistant3D":
        sampler = EquidistantSampling(text_to_list(widget_values['sample_to']))    
        
    show_config = [
        ['experiment name',  widget_values["experiment name"]],
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
    
    if st.checkbox("Show configuration"):
        st.table(pd.DataFrame(show_config, columns=["key", "value"]))

    if st.checkbox("Show model graph"):
        st.write('Please load the data first or enter the data shape manualy, comma separated.')
        #st.write('Note: the number of features will be changed to 1 in the graph')
        
        widget_history_checked([{'label':'Data Shape', 
                                 'default':'1,1,8,8,8', 'widget':text_main}])

        shape = widget_values['Data Shape']
        shape = np.array([int(i) for i in shape.split(',')])        
        
        #Load the data  
        if st.button('Load Data'):
            data_loader = load_data(widget_values['checkpoints'])
            x, y = data_loader.load_numpy()
            shape = np.array(x.shape)
            st.write('Loaded data shape: ', str(shape))

        estimator = Estimator(loaders=torch_splitter(loaders = [np.ones((1,1,1,1,1)),
                                                                np.ones((1,1,1,1,1))]))        
        if shape[1] != 1: 
            shape[1] = 1
            st.write('Setting number of channels to 1: ', str(shape))

        #try:
        graph = model_graph(model = estimator.model, shape = shape)
        st.graphviz_chart(graph.build_dot())
        #except: st.error('ValueError: Incorrect data shape, please edit the shape or load the data.')

    if st.checkbox("Show code of model"):           
        st.code(inspect.getsource(EstimatorModel), language='python')        
        widget_history_checked([{'label':'edit_port', 'default':8601, 'widget':number_main,
                                                      'min_value':1024, 'max_value':65535}])
        
        if st.button('Edit'):
            os.system('jupyter notebook ../../../sapsan/lib/estimator/cnn/cnn3d_estimator.py --no-browser --port=%d &'%widget_values['edit_port'])
            webbrowser.open('http://localhost:%d'%widget_values['edit_port'], new=2)
            
    else:
        widget_history_unchecked([{'label':'edit_port', 'default':config["edit_port"], 'widget':number_main,
                                              'min_value':1024, 'max_value':65535}])

    st.markdown("---")

    if st.button("Run experiment"):
        start = time.time()
        
        log_path = './logs/log.txt'
        if os.path.exists(log_path): 
            os.remove(log_path)
        else: pass        

        run_experiment()
        
        st.write('Finished in %.2f sec'%((time.time()-start))) 
        
        
    if st.button("MLflow tracking"):
        webbrowser.open('http://%s:%s'%(widget_values['mlflow_host'], widget_values['mlflow_port']), new=2)
        
    
    with open('temp.txt', 'w') as file:
        file.write('[config]\n')
        for key, value in widget_values.items():
            file.write('%s = %s\n'%(key, value))                            