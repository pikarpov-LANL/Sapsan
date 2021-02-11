import shutil
import unittest

from .notebook_runner import run_notebook


class TestExamples(unittest.TestCase):
    """Tests example notebooks."""

    def setUp(self) -> None:
        shutil.copytree("./sapsan/examples", "./runtime_test_resources")

    def tearDown(self) -> None:
        shutil.rmtree("./runtime_test_resources")

    def test_cnn_example(self):
        """Tests cnn example."""
        _, errors = run_notebook(notebook_path="./runtime_test_resources/cnn_example.ipynb",
                                 resources_path="./runtime_test_resources")
        self.assertEqual(errors, [])
