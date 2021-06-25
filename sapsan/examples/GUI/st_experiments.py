import streamlit as st
import os
import sys
import inspect
from pathlib import Path

#uncomment if cloned from github!
sys.path.append(str(Path.home())+"/Sapsan/")

from sapsan.lib.backends.fake import FakeBackend
from sapsan.lib.backends.mlflow import MLflowBackend
from sapsan.lib.data import HDF5Dataset, EquidistantSampling, flatten
from sapsan.lib.estimator import CNN3d, CNN3dConfig
from sapsan.lib.estimator.cnn.spacial_3d_encoder import CNN3dModel
from sapsan.lib.experiments.evaluate import Evaluate
from sapsan.lib.experiments.train import Train
from sapsan.utils.plot import model_graph

import pandas as pd
import torch
import matplotlib.pyplot as plt
import configparser
import webbrowser
import time
import numpy as np
from threading import Thread
from streamlit.ReportThread import add_report_ctx
import json
from collections import OrderedDict
import plotly.express as px
import os
import signal
import sys
from st_state_patch import SessionState
from multiprocessing import Process

#initialization of defaults
cf = configparser.RawConfigParser()
widget_values = {}


def intro():
    st.sidebar.success("Select an experiment above")

    st.markdown(
        
        """
        # Welcome to Sapsan!
        
        ---
        
        Sapsan is a pipeline for Machine Learning (ML) based turbulence modeling. While turbulence 
        is important in a wide range of mediums, the pipeline primarily focuses on astrophysical application. 
        With Sapsan, one can create their own custom models or use either conventional or physics-informed 
        ML approaches for turbulence modeling included with the pipeline ([estimators](https://github.com/pikarpov-LANL/Sapsan/wiki/Estimators)).
        For example, Sapsan features ML models in its set of tools to accurately capture the turbulent nature applicable to Core-Collapse Supernovae.
        
        > ## **Purpose**
        
        > Sapsan takes out all the hard work from data preparation and analysis in turbulence 
        > and astrophysical applications, leaving you focused on ML model design, layer by layer.             

        **👈 Select an experiment from the dropdown on the left** to see what Sapsan can do!
        ### Want to learn more?
        - Check out Sapsan on [Github](https://github.com/pikarpov-LANL/Sapsan)
        - Find the details on the [Wiki] (https://github.com/pikarpov-LANL/Sapsan/wiki)
    """
    )
    
    show_license = st.checkbox('License Information', value=False)
    if show_license:
        st.markdown(
            """
Sapsan has a BSD-style license, as found in the [LICENSE] (https://github.com/pikarpov-LANL/Sapsan/blob/master/LICENSE) file.            
            
© (or copyright) 2019. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
        """
        )
        
            
def cnn3d():
    st.title('Sapsan Configuration')
    st.write('This demo is meant to present capabilities of Sapsan. You can configure each part of the experiment at the sidebar. Once you are done, you can see the summary of your runtime parameters under _Show configuration_. In addition you can review the model that is being used (in the custom setup, you will also be able to edit it). Lastly click the _Run experiment_ button to train the test the ML model.')
    
    st.sidebar.markdown("General Configuration")
    
    try:
        cf.read('temp.txt')
        temp = dict(cf.items('config'))
    except: pass
    
    def make_recording_widget(f):
        """Return a function that wraps a streamlit widget and records the
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
        
    #show loss vs epoch progress with plotly
    def show_log(progress_slot, epoch_slot):        
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
                           shuffle = False,
                           train_fraction = 1)
        x, y = data_loader.load_numpy()
        return x, y, data_loader
    
            
    def run_experiment():
        
        if widget_values['backend_selection'] == 'Fake':
            tracking_backend = FakeBackend(widget_values['experiment name'])
            
        elif widget_values['backend_selection'] == 'MLflow':
            tracking_backend = MLflowBackend(widget_values['experiment name'], 
            widget_values['mlflow_host'],widget_values['mlflow_port'])
        
        #Load the data 
        x, y, data_loader = load_data(widget_values['checkpoints'])
        y = flatten(y)
        loaders = data_loader.convert_to_torch([x, y])
        
        st.write("Dataset loaded...")
        
        #Set the experiment
        training_experiment = Train(backend=tracking_backend,
                                    model=estimator,
                                    loaders = loaders,
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
        training_experiment.run()
        st.write('Trained in %.2f sec'%((time.time()-start)))
        st.success('Done! Plotting...')

        #def evaluate_experiment():
        #--- Test the model ---
        #Load the test data
        x, y, data_loader = load_data(widget_values['checkpoint_test'])
        loaders = [x, y]

        #Set the test experiment
        evaluation_experiment = Evaluate(backend=tracking_backend,
                                         model=training_experiment.model,
                                         loaders = loaders,
                                         data_parameters = data_loader)
        
        #Test the model
        evaluation_experiment.run()


        data = y
        #'data', data
        st.pyplot()
    
    #--- Load Default ---
    
    state = SessionState()
    
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
        sampler = EquidistantSampling(text_to_list(widget_values['input_size']), 
                                      text_to_list(widget_values['sample_to']))
    
    estimator = CNN3d(config=CNN3dConfig(n_epochs=int(widget_values['n_epochs']), 
                                         patience=int(widget_values['patience']), 
                                         min_delta=float(widget_values['min_delta'])))
        
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
                                 'default':'16,1,8,8,8', 'widget':text_main}])

        shape = widget_values['Data Shape']
        shape = np.array([int(i) for i in shape.split(',')])
        shape[1] = 1
        
        #Load the data  
        if st.button('Load Data'):
            x, y, data_loader = load_data(widget_values['checkpoints'])
            shape = x.shape
        
        try:
            graph = model_graph(estimator.model, shape)
            st.graphviz_chart(graph.build_dot())
        except: st.error('ValueError: Incorrect data shape, please edit the shape or load the data.')

    if st.checkbox("Show code of model"):           
        st.code(inspect.getsource(CNN3dModel), language='python')        
        widget_history_checked([{'label':'edit_port', 'default':8601, 'widget':number_main,
                                                      'min_value':1024, 'max_value':65535}])
        
        if st.button('Edit'):
            os.system('jupyter notebook ../../../sapsan/lib/estimator/cnn/spacial_3d_encoder.py --no-browser --port=%d &'%widget_values['edit_port'])
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
        
        #p = Process(target=run_experiment)
        #p.start()
        #state.pid = p.pid

        run_experiment()
        
        st.write('Finished in %.2f sec'%((time.time()-start))) 
    
    #if st.button("Stop experiment"):
            #sys.exit('Experiment stopped')
    #        st.stop
        
        
    if st.button("MLflow tracking"):
        #os.system('cmd.exe /C start http://%s:%s'%(widget_values['mlflow_host'], widget_values['mlflow_port']))
        webbrowser.open('http://%s:%s'%(widget_values['mlflow_host'], widget_values['mlflow_port']), new=2)
        
    #if st.button("Evaluate experiment"):
    #    #st.write("Experiment is running. Please hold on...")
    #    evaluate_experiment()
    
    with open('temp.txt', 'w') as file:
        file.write('[config]\n')
        for key, value in widget_values.items():
            file.write('%s = %s\n'%(key, value))
        
        
def custom():
    st.markdown("# Construction ongoing!")
    
def ccsn():
    import os
    
    st.markdown("# 1D Core-Collapse Supernovae Experiment")
    st.write('Below is an example on our ML implementation within 1D CCSN code developed by Chris Fryer (Los Alamos National Laboratory)')
    
    def run_ccsn():
        os.system("./ChCode/run15f1/st_test")
        st.error("At line 4617 of file 1dburn.f (unit = 60, file = 'tst1')")
    
    if st.button("Run experiment"):
        run_ccsn()
        
        
def config_write(var, file):
    cf.read(config_file)
    cf['sapsan_config'][''] = '1123'
    with open(config_file, 'w') as file:
        cf.write(file)
    
def test(): 
    
    #--- Experiment tracking backend ---
    experiment_name = "CNN experiment"

    #Fake (disable backend)
    tracking_backend = FakeBackend(experiment_name)

    #MLflow
    #launch mlflow with: mlflow ui --port=9000
    #uncomment tracking_backend to use mlflow

    MLFLOW_HOST = "localhost"
    MLFLOW_PORT = 9000
    path = "../data/t{checkpoint:1.0f}/{feature}_dim32_fm15.h5"
    features = ['u']
    target = ['u']

    #dimensionality of the data
    AXIS = 3

    #Dimensionality of your data per axis
    CHECKPOINT_DATA_SIZE = 32

    #Reduce dimensionality of each axis to
    SAMPLE_TO = 16

    #Dimensionality of each axis in a batch
    BATCH_SIZE = 8

    #Sampler to use for reduction
    sampler = EquidistantSampling(CHECKPOINT_DATA_SIZE, SAMPLE_TO, AXIS)
    estimator = CNN3d(
    config=CNN3dConfig(n_epochs=20, batch_dim=BATCH_SIZE, patience=10, min_delta=1e-5)
    )

    #Load the data
    data_loader = HDF5Dataset(path=path,
                       features=features,
                       target=target,
                       checkpoints=[0],
                       batch_size=BATCH_SIZE,
                       input_size=INPUT_SIZE,
                       sampler=sampler)
    x, y = data_loader.load()

    print(x.shape, y.shape)

    #Set the experiment
    training_experiment = Train(name=experiment_name,
                                 backend=tracking_backend,
                                 model=estimator,
                                 inputs=x, targets=y,
                                 data_parameters = data_loader.get_parameters())

    #Train the model
    training_experiment.run()
    #tracking_backend = MLflowBackend(experiment_name, MLFLOW_HOST, MLFLOW_PORT)
    '''
    st.write('----Before----')
    try:
        cf.read('temp.txt')
        temp = dict(cf.items('config'))
        st.write(temp)
    except: st.write('no temp 8(')

    def make_recording_widget(f):
        """Return a function that wraps a streamlit widget and records the
        widget's values to a global dictionary.
        """
        def wrapper(label, *args, **kwargs):
            widget_value = f(label, *args, **kwargs)
            widget_values[label] = widget_value
            return widget_value

        return wrapper

    def widget_history(name, default):
        if checkbox(name+'_checkbox'):
            try:
                if widget_values['flag'] == True:
                    widget_values['flag'] = False
                    try:
                        widget_values[name+'_default'] = int(temp[name])
                        number("recorded number", value = int(temp[name]))
                    except: number(name, value = widget_values[name+'_default'])
                else:
                    number(name, value = widget_values[name+'_default'])
                st.write('I tried and succeded')
            except: 
                widget_values['flag'] = False
                number(name, value = default)  
        else:
            widget_values['flag'] = True        
            widget_values[name+'_default'] = default
    
    
    button = make_recording_widget(st.button)
    number = make_recording_widget(st.number_input)
    checkbox = make_recording_widget(st.checkbox)
    #button("recorded button")
    name = 'recorded number'
    default = 10
    if st.button("reset"):
        widget_values[name+'_default'] = default
        widget_values[name] = default
        widget_values['flag'] = False
        st.write(widget_values[name], widget_values[name+'_default'])
        #widget_history(name, default)
    
    

            
    name = 'recorded number'
    print(widget_history(name, default))
    

        
    st.write('----After----')
    st.write("recorded values: ", widget_values)
    
    with open('temp.txt', 'w') as file:
        file.write('[config]\n')
        for key, value in widget_values.items():
            file.write('%s = %s\n'%(key, value))
    '''
   


'''
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def write_config(config_file, config):
    with open(config_file, 'w') as file:
        st.write('writing to file! ', config['n_epochs'])
        file.write('[sapsan_config]\n')
        for key, value in config.items():
            file.write('%s = %s \n'%(key, value))

def index_history_checked(params):
    widget_type = {number:int, text:str, checkbox:bool, selectbox:int}
    for i in range(len(params)):
        default = params[i]['default']
        widget = params[i]['widget']
        if 'name' in params[i]: name = params[i]['name']
        else: name = params[i]['label']

        if widget == selectbox: params[i]['input']='index'
        else: params[i]['input']='value'

        not_widget_params = ['default', 'widget', 'widget_type', 'name', 'input']
        additional_params = {key:value for key, value in params[i].items() if key not in not_widget_params}
        try:
            if widget_values[name+'_flag'] == True:
                widget_values[name+'_flag'] = False
                try:
                    widget_values[name+'_default'] = widget_type[widget](temp[name])
                    widget(index = widget_type[widget](temp[name]), **additional_params)
                except: widget(index = widget_values[name+'_default'], **additional_params)
            else:
                widget(index = widget_values[name+'_default'], **additional_params)
        except: 
            widget_values[name+'_flag'] = False
            widget(index = widget_type[widget](default), **additional_params)
            
            
            
        
        
        #index_history_checked([{'label':'backend_selection', 'default':1, 
        #                        'widget':selectbox, 
        #                        'options':widget_values['backend_list']}])
        
        #'name':'backend_selection_index'
        
        #widget_values['backend_selection'] = widget_values['backend_list'][widget_values['backend_selection_index']]
        #st.write('backend print', widget_values['backend_selection'])
        
        
                #widget_history_unchecked([{'label':'backend_selection', 'name':'backend_selection_index', 'default':1, 
        #                        'widget':selectbox, 
        #                        'options':widget_values['backend_list']}])
'''
"""
    #show loss vs epoch progress with plotly
    def show_log(progress_slot, epoch_slot):
        from datetime import datetime
        
        #log_path = 'logs/checkpoints/_metrics.json'
        log_path = 'logs/log.txt'
        log_exists = False
        while log_exists == False:
            if os.path.exists(log_path):
                log_exists = True
            time.sleep(0.1)
            
        plot_data = {'epoch':[], 'train_loss':[]}
        last_epoch = 0
        running = True
        
        start_time= datetime.now()
        while running:
            with open(log_path) as file:
                #get the date of the latest event
                lines = list(file)
                latest_time = lines[-4].replace(",",".")
                latest_time = datetime.strptime(latest_time, '[%Y-%m-%d %H:%M:%S.%f] ')

                #check for the newest entry
                if start_time > latest_time:
                    continue
                else:
                    current_epoch = int(lines[-2].split('/')[0])
                    train_loss = float(lines[-2].split('loss=')[-1])
                    valid_loss = float(lines[-1].split('loss=')[-1])

                '''
                #to read a .json file
                data = OrderedDict(json.load(file))
                elem = list(data.keys())

                if 'epoch' in elem[-1]:
                    current_epoch = int(elem[-1].rpartition('_')[-1]) + 1
                else:
                    current_epoch = -1
                '''
            if current_epoch == last_epoch or current_epoch == -1:
                pass
            else:     
                #metrics = data['epoch_%d'%(current_epoch-1)][-1]
                metrics = {'train_loss':train_loss, 'valid_loss':valid_loss}
                epoch_slot.markdown('Epoch:$~$**%d** $~~~~~$ Train Loss:$~$**%.4e**'%(current_epoch, metrics['train_loss']))
                plot_data['epoch'] = np.append(plot_data['epoch'], current_epoch)
                plot_data['train_loss'] = np.append(plot_data['train_loss'], metrics['train_loss'])                
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
"""

