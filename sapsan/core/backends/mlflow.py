import mlflow

from sapsan.core.models import ExperimentBackend


class MlFlowExperimentBackend(ExperimentBackend):
    def __init__(self, name: str, host: str, port: int):
        super().__init__(name)
        self.host = host
        self.port = port
        self.mlflow_url = "http://{host}:{port}".format(host=host,
                                                        port=port)
        mlflow.set_tracking_uri(self.mlflow_url)
        self.experiment_id = mlflow.set_experiment(name)

    def log_metric(self, name: str, value: float):
        mlflow.log_metric(name, value)

    def log_parameter(self, name: str, value: str):
        mlflow.log_param(name, value)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)