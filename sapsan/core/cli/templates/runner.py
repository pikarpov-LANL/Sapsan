TEMPLATE = """
import os
import argparse
    
from {name}.algorithm.{name}_algorithm import {name_upper}Estimator
from {name}.dataset.{name}_dataset import {name_upper}Dataset
from {name}.experiment.{name}_experiment import {name_upper}Experiment
    
    
def run(multiplier: int):
    dataset = {name_upper}Dataset()
    estimator = {name_upper}Estimator(multiplier=multiplier)
    experiment = {name_upper}Experiment(estimator=estimator, dataset=dataset)
    
    result = experiment.execute()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", required=False, default=4, type=int)
    args = parser.parse_args()
    run(multiplier=args.multiplier)

"""


def get_template(name: str):
    return TEMPLATE.format(name=name.lower(),
                           name_upper=name.capitalize())
