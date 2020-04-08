import os
import shutil
import unittest

from sapsan.lib.estimator import Spacial3dEncoderNetworkEstimator, Spacial3dEncoderNetworkEstimatorConfiguration
from sapsan.lib.estimator import SpacialAutoencoderNetworkEstimator, SpacialAutoencoderNetworkEstimatorConfiguration


class TestCnnEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.resources_path = "./cnn_test_resources"
        os.mkdir(self.resources_path)

    def test_encoder_save_and_load(self):
        estimator = Spacial3dEncoderNetworkEstimator(Spacial3dEncoderNetworkEstimatorConfiguration(1))
        estimator.save(self.resources_path)

        loaded_estimator = Spacial3dEncoderNetworkEstimator.load(self.resources_path)
        self.assertEqual(estimator.config.n_input_channels, loaded_estimator.config.n_input_channels)
        self.assertEqual(estimator.config.grid_dim, loaded_estimator.config.grid_dim)
        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)
        self.assertEqual(estimator.config.n_output_channels, loaded_estimator.config.n_output_channels)

    def test_auto_encoder_save_and_load(self):
        estimator = SpacialAutoencoderNetworkEstimator(SpacialAutoencoderNetworkEstimatorConfiguration(1))
        estimator.save(self.resources_path)

        loaded_estimator = SpacialAutoencoderNetworkEstimator.load(self.resources_path)
        self.assertEqual(estimator.config.n_input_channels, loaded_estimator.config.n_input_channels)
        self.assertEqual(estimator.config.grid_dim, loaded_estimator.config.grid_dim)
        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)
        self.assertEqual(estimator.config.n_output_channels, loaded_estimator.config.n_output_channels)

    def tearDown(self) -> None:
        shutil.rmtree(self.resources_path)