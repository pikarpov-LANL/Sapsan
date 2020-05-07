import os
import shutil
import unittest

from sapsan.lib.estimator import KRR, KRRConfig


class TestKrrEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.resources_path = "./resources"
        os.mkdir(self.resources_path)

    def test_save_and_load(self):
        estimator = KRR(
            KRRConfig(gamma=0.1, alpha=0.2)
        )
        estimator.save(self.resources_path)
        loaded_estimator = KRR.load(self.resources_path)
        self.assertEqual(estimator.config.gamma, loaded_estimator.config.gamma)
        self.assertEqual(estimator.config.alpha, loaded_estimator.config.alpha)

    def tearDown(self) -> None:
        shutil.rmtree(self.resources_path)
