import yaml
from typing import List, Optional


class SapsanConfig(object):
    def __init__(
            self,
            name: str,
            ttrain: List[int],
            dim: int,
            parameters: List[str], # TODO: rename to features or something like that :)
            target: str,
            filter_modes: int,
            target_comp: int,
            alpha: float,
            gamma: float,
            max_dim: int,
            dataset: str,
            axis: int,
            from_3d: bool,
            savepath: str,
            dt: float,
            data_type: str,
            path: str,
            method: str,
            experiment_name: str,
            step: Optional[int] = None
    ):
        self.name = name
        self.ttrain = ttrain
        self.step = step
        self.dim = dim
        self.parameters = parameters
        self.target = target
        self.filter_modes = filter_modes
        self.target_comp = target_comp
        self.alpha = alpha
        self.gamma = gamma
        self.max_dim = max_dim
        self.dataset = dataset
        self.axis = axis
        self.from_3d = from_3d
        self.savepath = savepath
        self.dt = dt
        self.data_type = data_type
        self.path = path
        self.method = method
        self.experiment_name = experiment_name

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            return cls(**cfg['config'])

    def __repr__(self):
        return "SapsanConfig(name={0})".format(self.name)
