import shutil
import unittest
import os

from .notebook_runner import run_notebook
from sapsan import __path__

class TestExamples(unittest.TestCase):
    """Tests example notebooks."""
    
    def notebooks(self):
        return ['cnn_example.ipynb', 
				'picae_example.ipynb', 
				'krr_example.ipynb',
				'pimlturb_diagonal_example.ipynb']

    def test_examples(self):
        """Tests examples"""       
        path = '{path}/examples/'.format(path=__path__[0])
        for nt in self.notebooks():
            _, errors = run_notebook(notebook_path=path+nt,
                                     resources_path=path) 
            print(errors)
            self.assertEqual(errors, [])
