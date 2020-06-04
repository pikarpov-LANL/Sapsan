TEMPLATE = """import numpy as np
import unittest

from {name}.algorithm.research_algorithm import {name_upper}Estimator


class Test{name_upper}Estimator(unittest.TestCase):

    def test_predict(self):
        multiplier = 2
        input_data = np.random.random([1, 2])
        expected_result = input_data * multiplier
        estimator = {name_upper}Estimator(multiplier=multiplier)
        result = estimator.predict(input_data)

        self.assertTrue(np.allclose(result, expected_result))
"""


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
