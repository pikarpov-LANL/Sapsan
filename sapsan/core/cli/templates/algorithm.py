TEMPLATE = """
import json
from typing import List, Union, Optional
    
import numpy as np
    
from sapsan.core.abstrctions.algorithm import Estimator, Metric, Artifact, Parameter
from sapsan.core.abstrctions.dataset import Dataset
    
    
class {name_upper}Estimator(Estimator):
    def __init__(self,
                 multiplier: int):
        self.multiplier = multiplier
    
    def train(self, features: Union[np.array, Dataset], targets: Optional[Union[np.array, Dataset]] = None):
        pass
    
    def predict(self, data: Union[np.array, Dataset]) -> Union[np.array, Dataset]:
        return data * self.multiplier
    
    def save(self, path: str):
        with open(path, "w") as file:
            to_save = dict(("multiplier", self.multiplier))
            json.dump(to_save, file)
    
    @classmethod
    def load(cls, path: str):
        with open(path, "r") as file:
            data = json.load(file)
            return {name_upper}Estimator(data["multiplier"])

"""


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
