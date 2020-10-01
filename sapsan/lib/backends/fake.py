import logging

from sapsan.core.models import ExperimentBackend


class FakeBackend(ExperimentBackend):    
    def start(self, run_name: str):
        pass
    
    def log_parameter(self, name: str, value: str):
        logging.info("Logging experiment '{experiment}' parameter "
                     "{name}: {value}".format(experiment=self.name,
                                              name=name,
                                              value=value))

    def log_artifact(self, path: str):
        logging.info("Logging artifact {path}".format(path=path))

    def log_metric(self, name: str, value: float):
        logging.info("Logging experiment '{experiment}' metric "
                     "{name}: {value}".format(experiment=self.name,
                                              name=name,
                                              value=value))
    def end(self):
        pass