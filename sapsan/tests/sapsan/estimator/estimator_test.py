import os
import shutil
import unittest
import numpy as np

from sapsan.lib.data.data_functions import torch_splitter
from sapsan.lib.estimator import load_estimator, load_sklearn_estimator
from sapsan.lib.estimator.krr.krr_estimator import KRR, KRRConfig
from sapsan.lib.estimator.picae.picae_estimator import PICAE, PICAEConfig
from sapsan.lib.estimator.cnn.cnn3d_estimator import CNN3d, CNN3dConfig

class TestCnnEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.resources_path = "./test_resources"
        if os.path.exists(self.resources_path): shutil.rmtree(self.resources_path)
        os.mkdir(self.resources_path)

        
    def default_loaders(self, model_type):
        if model_type=='torch':
            x = np.ones((3,16,16,16,16))
            y = np.ones((3,16,16,16,16))
            return torch_splitter(loaders = [x,y])
        if model_type=='sklearn':
            x = np.ones((3,16))
            y = np.ones((3,16))
            return [x,y]
   

    def test_cnn3d_save_and_load(self):
        estimator = CNN3d(config = CNN3dConfig(n_epochs = 1),
                          loaders = self.default_loaders('torch'))
        estimator.model = estimator.train()
        estimator.save(self.resources_path)
        
        loaded_estimator = load_estimator.load(self.resources_path, 
                               estimator=CNN3d(config=CNN3dConfig(),
                                               loaders=self.default_loaders('torch')),
                               load_saved_config=True)
        
        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)
    
    
    def test_picae_save_and_load(self):
        estimator = PICAE(config = PICAEConfig(n_epochs = 1),
                          loaders = self.default_loaders('torch'))
        estimator.model = estimator.train()
        estimator.save(self.resources_path)
        
        loaded_estimator = load_estimator.load(self.resources_path, 
                               estimator=PICAE(config=PICAEConfig(),
                                               loaders=self.default_loaders('torch')),
                               load_saved_config=True)

        self.assertEqual(estimator.config.n_epochs, loaded_estimator.config.n_epochs)
    
    
    def test_krr_save_and_load(self):
        estimator = KRR(config = KRRConfig(gamma=0.1, alpha=0.2),
                        loaders = self.default_loaders('sklearn'))
        estimator.model = estimator.train()
        estimator.save(self.resources_path)

        loaded_estimator = load_sklearn_estimator.load(self.resources_path, 
                               estimator=KRR(config=KRRConfig(),
                                           loaders=self.default_loaders('sklearn')),
                               load_saved_config=True)
        
        self.assertEqual(estimator.config.gamma, loaded_estimator.config.gamma)
        self.assertEqual(estimator.config.alpha, loaded_estimator.config.alpha)         
                
        
    def tearDown(self) -> None:
        shutil.rmtree(self.resources_path)