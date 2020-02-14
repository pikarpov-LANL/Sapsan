from sapsan.general.data.flatten_from_3d import Flatten3dDataset
from sapsan.general.data.jhtdb_dataset import JHTDB128Dataset
from sapsan.general.estimator import KrrEstimator, KrrEstimatorConfiguration
from sapsan.general.experiment import TrainingExperiment, EvaluationExperiment, FakeExperimentBackend


def run():
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    GRID_SIZE = 32
    features = ['u', 'b', 'a',
                'du0', 'du1', 'du2',
                'db0', 'db1', 'db2',
                'da0', 'da1', 'da2']
    labels = ['tn']
    checkpoints = [0.0]

    training_experiment_name = "Training experiment"
    estimator = KrrEstimator(
        config=KrrEstimatorConfiguration.from_yaml()
    )
    print(estimator.config.to_dict())

    x, y = Flatten3dDataset(path=dataset_root_dir,
                           features=features,
                           labels=labels,
                           checkpoints=checkpoints,
                           grid_size=GRID_SIZE).load()

    training_experiment = TrainingExperiment(name=training_experiment_name,
                                             backend=FakeExperimentBackend(training_experiment_name),
                                             model=estimator,
                                             inputs=x, targets=y)
    training_experiment.run()


if __name__ == '__main__':
    run()