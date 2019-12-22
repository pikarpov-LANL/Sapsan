import os
import unittest
from sapsan.config.general import SapsanConfig


class TestSapsanConfig(unittest.TestCase):
    def test_config_loader(self):
        path = "{0}/resources/default.yaml".format(os.path.dirname(__file__))
        cfg = SapsanConfig.from_yaml(path)
        self.assertEqual(cfg.name, '16')