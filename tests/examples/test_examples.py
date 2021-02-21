import shutil
import unittest
import os

from .notebook_runner import run_notebook


class TestExamples(unittest.TestCase):
    """Tests example notebooks."""

    def notebooks(self):
        return ['cnn_example.ipynb', 'krr_example.ipynb']
    
    def setUp(self) -> None:
        os.mkdir("./runtime_test_resources")
        shutil.copytree("./sapsan/examples/data", "./data")
        
        for nt in self.notebooks():
            shutil.copyfile("./sapsan/examples/"+nt, 
                            "./runtime_test_resources/"+nt)
        
    def tearDown(self) -> None:
        shutil.rmtree("./runtime_test_resources")
        shutil.rmtree("./data")

    def test_cnn_example(self):
        """Tests cnn example."""
        for nt in self.notebooks():
            _, errors = run_notebook(notebook_path="./runtime_test_resources/"+nt,
                                     resources_path="./")
            self.assertEqual(errors, [])
