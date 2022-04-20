import mlflow
from threading import Thread
import os
import time
import socket
from contextlib import closing

from sapsan.core.models import ExperimentBackend


class MLflowBackend(ExperimentBackend):
    def __init__(self, name: str = 'experiment',
                       host: str = 'localhost', 
                       port: int = 9000):
        super().__init__(name)
        self.host = host
        self.port = port
        
        self.mlflow_url = "http://{host}:{port}".format(host=host,
                                                        port=port)
        mlflow.set_tracking_uri(self.mlflow_url)
        if self.check_open_port():
            print("%s:%s is busy"%(self.host, self.port))
            self.experiment_id = mlflow.set_experiment(name)
            print("mlflow ui is already running at %s:%s"%(self.host, self.port))
        else:
            print("starting mlflow ui, please wait ...")
            self.start_ui()
            self.experiment_id = mlflow.set_experiment(name)
            print("mlflow ui is running at %s:%s"%(self.host, self.port))
    
    def start_ui(self):
        mlflow_thread = Thread(target=
                       os.system("mlflow ui --host %s --port %s &"%(self.host, self.port)))
        mlflow_thread.start()
        time.sleep(3)
        
    def start(self, run_name: str, nested = False, run_id = None):
        mlflow.start_run(run_name = run_name, nested = nested, run_id = run_id) 
        return mlflow.active_run().info.run_id
    
    def resume(self, run_id, nested = True):
        resumed_run = mlflow.start_run(run_id, nested = nested)
        print(f"Status of MLflow run {run_id} (nested={nested}): {resumed_run.info.status}")
        print("Please don't forget to call 'backend.end()' at the end...")
        
    def log_metric(self, name: str, value: float):
        mlflow.log_metric(name, value)

    def log_parameter(self, name: str, value: str):        
        mlflow.log_param(name, value)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def close_active_run(self):
        while mlflow.active_run()!=None: mlflow.end_run()
        
    def end(self):
        mlflow.end_run()
        
    def check_open_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex((self.host, self.port)) == 0:
                return True
            else: return False