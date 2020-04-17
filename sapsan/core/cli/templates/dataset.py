TEMPLATE = """
import numpy as np
    
from sapsan.core.abstrctions.dataset import Dataset
    
    
class {name}Dataset(Dataset):
    def load(self):
        return np.random.random((4, 4))

"""


def get_template(name: str):
    return TEMPLATE.format(name=name.capitalize())
