import argparse
import json
import os
import logging
import sys
from typing import List, Optional

from sapsan.core.backends.fake import FakeExperimentBackend
from sapsan.core.backends.mlflow import MlFlowExperimentBackend
from sapsan.core.data.jhtdb_dataset import JHTDB128Dataset
from sapsan.core.data.sampling.equidistant_sampler import Equidistance3dSampling
from sapsan.core.estimator.cnn.spacial_3d_encoder import Spacial3dEncoderNetworkEstimator, \
    Spacial3dEncoderNetworkEstimatorConfiguration
from sapsan.core.experiments.evaluation_3d import Evaluation3dExperiment
from sapsan.core.experiments.training import TrainingExperiment

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("get_data.log")])
logger = logging.getLogger(__name__)

os.environ["AWS_ACCESS_KEY_ID"] = "<AWS_ACCESS_KEY_ID>"
os.environ["AWS_SECRET_ACCESS_KEY"] = "<AWS_SECRET_ACCESS_KEY>"


def run(experiment_name: str,
        dataset_root_dir: str,
        checkpoint_data_size: int,
        sample_to: int,
        grid_size: int,
        features: List[str],
        labels: List[str],
        n_epochs: int,
        mlflow_backend_host: Optional[str],
        mlflow_backend_port: Optional[int]):

    if mlflow_backend_host and mlflow_backend_port:
        tracking_backend = MlFlowExperimentBackend(experiment_name, mlflow_backend_host, mlflow_backend_port)
    else:
        tracking_backend = FakeExperimentBackend(experiment_name)

    sampler = Equidistance3dSampling(checkpoint_data_size, sample_to)

    estimator = Spacial3dEncoderNetworkEstimator(
        config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=n_epochs, grid_dim=grid_size)
    )

    x, y = JHTDB128Dataset(path=dataset_root_dir,
                           features=features,
                           labels=labels,
                           checkpoints=[0.0, 0.01, 0.25],
                           grid_size=grid_size,
                           checkpoint_data_size=checkpoint_data_size,
                           sampler=sampler).load()

    training_experiment = TrainingExperiment(name=experiment_name,
                                             backend=tracking_backend,
                                             model=estimator,
                                             inputs=x, targets=y)
    training_experiment.run()

    x, y = JHTDB128Dataset(path=dataset_root_dir,
                           features=features,
                           labels=labels,
                           checkpoints=[0.025],
                           grid_size=grid_size,
                           checkpoint_data_size=checkpoint_data_size,
                           sampler=sampler).load()

    evaluation_experiment = Evaluation3dExperiment(name=experiment_name,
                                                   backend=tracking_backend,
                                                   model=training_experiment.model,
                                                   inputs=x, targets=y,
                                                   n_output_channels=3,
                                                   grid_size=grid_size,
                                                   checkpoint_data_size=sample_to)

    result = evaluation_experiment.run()

    metrics = {
        'metrics': [{'name': metric, 'numberValue': value}
                    for metric, value in evaluation_experiment.get_metrics().items()]
    }
    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    metadata = {
        'outputs': [{
            'type': 'table',
            'format': 'csv',
            'header': ['metric', 'value'],
            'source': [[metric, value]
                       for metric, value in evaluation_experiment.get_metrics().items()]
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for CNN encoder experiment.')
    parser.add_argument('--experiment_name',
                        type=str,
                        default="CNN encoder experiment",
                        required=False,
                        help='Experiment name.')
    parser.add_argument('--dataset_root_dir',
                        type=str,
                        default="/app/dataset",
                        required=False,
                        help='Root directory of dataset.')
    parser.add_argument('--checkpoint_data_size',
                        type=int,
                        default=128,
                        required=False,
                        help='Size of cube of original data file.')
    parser.add_argument('--sample_to',
                        type=int,
                        default=32,
                        required=False,
                        help='Sample dataset to dimension size.')
    parser.add_argument('--grid_size',
                        type=int,
                        default=8,
                        required=False,
                        help='Size of cube that would be used to split data.')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of epochs to train.')
    parser.add_argument('--mlflow_backend_host',
                        type=str,
                        default=None,
                        required=False,
                        help='MlFlow backend host.')
    parser.add_argument('--mlflow_backend_port',
                        type=int,
                        default=None,
                        required=False,
                        help='MlFlow backend port.')

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.warning(f"Parsed unknown args: {unknown}")
    kwargs = dict(vars(args))

    try:
        logger.info('CNN encoder experiment...')

        run(experiment_name=args.experiment_name,
            dataset_root_dir=args.dataset_root_dir,
            checkpoint_data_size=args.checkpoint_data_size,
            sample_to=args.sample_to,
            grid_size=args.grid_size,
            features=['u', 'b', 'a',
                      'du0', 'du1', 'du2',
                      'db0', 'db1', 'db2',
                      'da0', 'da1', 'da2'],
            labels=['tn'],
            n_epochs=args.n_epochs,
            mlflow_backend_host=args.mlflow_backend_host,
            mlflow_backend_port=args.mlflow_backend_port)

        logger.info('Done.')
    except Exception as e:
        logger.exception(e)
