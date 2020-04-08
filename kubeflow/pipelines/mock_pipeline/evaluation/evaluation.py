import argparse
import json
import logging
import sys

from sapsan.lib.mocks import FakeDataset
from sapsan.lib.mocks import FakeInferenceExperiment, FakeTrainingExperiment
from sapsan.lib.mocks import LinearRegressionEstimator, LinearRegressionEstimatorConfiguration

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("get_data.log")])
logger = logging.getLogger(__name__)


def evaluate():
    dataset = FakeDataset()
    estimator = LinearRegressionEstimator(LinearRegressionEstimatorConfiguration())

    training_experiment = FakeTrainingExperiment("training_experiment", estimator, dataset)
    training_experiment.run()

    inference_experiment = FakeInferenceExperiment("evaluation_experiment", dataset, training_experiment.estimator)
    inference_experiment.run()

    mse = inference_experiment.get_report()["metrics"]["mse"]

    metrics = {
        'metrics': [{
            'name': 'mse',
            'numberValue': mse
        }]
    }
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    metadata = {
        'outputs': [{
            'type': 'table',
            'format': 'csv',
            'header': ['metric', 'value'],
            'source': ['mse', mse]
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    return {
        "mse": mse
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.warning(f"Parsed unknown args: {unknown}")
    kwargs = dict(vars(args))

    try:
        logger.info('Evaluation...')
        evaluate()
        logger.info('Done.')
    except Exception as e:
        logger.exception(e)
