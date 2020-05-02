import shutil
import unittest

from .notebook_runner import run_notebook


class TestExamples(unittest.TestCase):
    """Tests example notebooks."""

    def setUp(self) -> None:
        shutil.copytree("../examples", "./examples/resources")

    def tearDown(self) -> None:
        shutil.rmtree("./examples/resources")

    def test_cnn_example(self):
        """Tests cnn example."""
        nb, errors = run_notebook(notebook_path="./examples/resources/cnn_example.ipynb",
                                  resources_path="./examples/resources")
        self.assertEqual(errors, [])
