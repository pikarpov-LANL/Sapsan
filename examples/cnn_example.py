import os

from sapsan.general.backends.fake import FakeExperimentBackend
from sapsan.general.backends.mlflow import MlFlowExperimentBackend
from sapsan.general.data.jhtdb_dataset import JHTDB128Dataset
from sapsan.general.data.sampling.equidistant_sampler import Equidistance3dSampling
from sapsan.general.estimator.cnn.spacial_3d_encoder import Spacial3dEncoderNetworkEstimator, Spacial3dEncoderNetworkEstimatorConfiguration
from sapsan.general.experiments.evaluation_3d import Evaluation3dExperiment
from sapsan.general.experiments.training import TrainingExperiment

os.environ["AWS_ACCESS_KEY_ID"] = "<AWS_ACCESS_KEY_ID>"
os.environ["AWS_SECRET_ACCESS_KEY"] = "<AWS_SECRET_ACCESS_KEY>"

def run():
    MLFLOW_BACKEND_HOST = "0.0.0.0"
    MLFLOW_BACKEND_PORT = 9000
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    CHECKPOINT_DATA_SIZE = 128
    SAMPLE_TO = 32
    GRID_SIZE = 8
    features = ['u', 'b', 'a',
                'du0', 'du1', 'du2',
                'db0', 'db1', 'db2',
                'da0', 'da1', 'da2']
    labels = ['tn']

    sampler = Equidistance3dSampling(CHECKPOINT_DATA_SIZE, SAMPLE_TO)

    experiment_name = "CNN experiment"

    estimator = Spacial3dEncoderNetworkEstimator(
        config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=1, grid_dim=GRID_SIZE)
    )

    # tracking_backend = FakeExperimentBackend(experiment_name)
    tracking_backend = MlFlowExperimentBackend(experiment_name, MLFLOW_BACKEND_HOST, MLFLOW_BACKEND_PORT)

    x, y = JHTDB128Dataset(path=dataset_root_dir,
                           features=features,
                           labels=labels,
                           checkpoints=[0.0, 0.01, 0.25],
                           grid_size=GRID_SIZE,
                           checkpoint_data_size=CHECKPOINT_DATA_SIZE,
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
                           grid_size=GRID_SIZE,
                           checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                           sampler=sampler).load()

    evaluation_experiment = Evaluation3dExperiment(name=experiment_name,
                                                   backend=tracking_backend,
                                                   model=training_experiment.model,
                                                   inputs=x, targets=y,
                                                   n_output_channels=3,
                                                   grid_size=GRID_SIZE,
                                                   checkpoint_data_size=SAMPLE_TO)

    evaluation_experiment.run()


if __name__ == '__main__':
    run()
