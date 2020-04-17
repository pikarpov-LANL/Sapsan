TEMPLATE = """
from typing import List
    
from sapsan.core.abstrctions.algorithm import Parameter, Artifact, Metric
from sapsan.core.abstrctions.experiment import Experiment
from sapsan.core.abstrctions.tracking import TrackingBackend
from sapsan.core.tracking.logger import LoggingBackend
from {name}.algorithm import {name_upper}Estimator
from {name}.dataset import {name_upper}Dataset
    
    
class {name_upper}Experiment(Experiment):
    def __init__(self,
                 estimator: {name_upper}Estimator,
                 dataset: {name_upper}Dataset,
                 tracking_backend: TrackingBackend = LoggingBackend()):
        super().__init__(tracking_backend)
        self.estimator = estimator
        self.dataset = dataset
    
    def run(self, **kwargs):
        data = self.dataset.load()
        return self.estimator.predict(data)
    
    def test(self, **kwargs):
        data = self.dataset.load()
        return self.estimator.predict(data)
    
    @property
    def parameters(self) -> List[Parameter]:
        return [
            Parameter("estimator", str(self.estimator)),
            Parameter("dataset", str(self.dataset))
        ]
    
    @property
    def metrics(self) -> List[Metric]:
        return []
    
    @property
    def artifacts(self) -> List[Artifact]:
        return []

"""


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
