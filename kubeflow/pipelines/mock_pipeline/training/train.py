import argparse
import logging
import sys
import time
import json

from sapsan.lib.mocks import FakeDataset
from sapsan.lib.mocks import LinearRegressionEstimator, LinearRegressionEstimatorConfiguration
from sapsan.lib.mocks import FakeTrainingExperiment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("get_data.log")])
logger = logging.getLogger(__name__)


def train():
    dataset = FakeDataset()
    estimator = LinearRegressionEstimator(LinearRegressionEstimatorConfiguration())
    training_experiment = FakeTrainingExperiment("training_experiment", estimator, dataset)

    start_time = time.time()
    logger.info("Training start at {}".format(start_time))
    training_experiment.run()
    end_time = time.time()
    logger.info("Training ends at {}".format(end_time))

    runtime = int(end_time - start_time)

    metrics = {
        'metrics': [{
            'name': 'runtime',
            'numberValue': runtime
        }]
    }
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    metadata = {
        'outputs': [{
            'type': 'table',
            'format': 'csv',
            'header': ['metric', 'value'],
            'source': ['runtime', runtime]
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    return {
        "time": runtime
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.warning(f"Parsed unknown args: {unknown}")
    kwargs = dict(vars(args))

    try:
        logger.info('Training...')
        train()
        logger.info('Done.')
    except Exception as e:
        logger.exception(e)
