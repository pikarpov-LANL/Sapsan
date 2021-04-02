import os
import shutil
import unittest

from sapsan.lib.estimator import CNN3d, CNN3dConfig, PICAE, PICAEConfig, KRR, KRRConfig

class TestCnnEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.resources_path = "./test_resources"
        os.mkdir(self.resources_path)

    def test_cnn3d_save_and_load(self):
        estimator = CNN3d(CNN3dConfig(n_epochs = 1))
        estimator.save(self.resources_path)

        loaded_estimator = CNN3d.load(self.resources_path, model=CNN3d, config=CNN3dConfig)
        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)

        
    def test_picae_save_and_load(self):
        estimator = PICAE(PICAEConfig())
        estimator.save(self.resources_path)

        loaded_estimator = PICAE.load(self.resources_path, model=PICAE, config=PICAEConfig)
        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)
    
        
    def test_krr_save_and_load(self):
        estimator = KRR(KRRConfig(gamma=0.1, alpha=0.2))
        estimator.save(self.resources_path)
        loaded_estimator = KRR.load(self.resources_path)
        self.assertEqual(estimator.config.gamma, loaded_estimator.config.gamma)
        self.assertEqual(estimator.config.alpha, loaded_estimator.config.alpha)        

        
    def tearDown(self) -> None:
        shutil.rmtree(self.resources_path)
