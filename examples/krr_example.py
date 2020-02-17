from sapsan.general.backends.fake import FakeExperimentBackend
from sapsan.general.data.flatten_dataset import FlattenFrom3dDataset
from sapsan.general.data.sampling.equidistant_sampler import Equidistance3dSampling
from sapsan.general.estimator import KrrEstimator, KrrEstimatorConfiguration
from sapsan.general.experiments.evaluation_flatten import EvaluationFlattenExperiment
from sapsan.general.experiments.training import TrainingExperiment


def run():
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    CHECKPOINT_DATA_SIZE = 128
    SAMPLE_TO = 16
    features = ['u', 'b', 'a',
                'du0', 'du1', 'du2',
                'db0', 'db1', 'db2',
                'da0', 'da1', 'da2']
    labels = ['tn']
    train_checkpoints = [0.0]

    sampler = Equidistance3dSampling(CHECKPOINT_DATA_SIZE, SAMPLE_TO)

    training_experiment_name = "Training experiment"
    estimator = KrrEstimator(
        config=KrrEstimatorConfiguration().from_yaml()
    )
    x, y = FlattenFrom3dDataset(path=dataset_root_dir,
                                features=features,
                                labels=labels,
                                checkpoints=train_checkpoints,
                                sampler=sampler,
                                label_channels=[0]).load()

    training_experiment = TrainingExperiment(name=training_experiment_name,
                                             backend=FakeExperimentBackend(training_experiment_name),
                                             model=estimator,
                                             inputs=x, targets=y)
    training_experiment.run()

    evaluation_experiment_name = "Evaluation experiment"
    evaluation_experiment = EvaluationFlattenExperiment(name=evaluation_experiment_name,
                                                        backend=FakeExperimentBackend(evaluation_experiment_name),
                                                        model=training_experiment.model,
                                                        inputs=x, targets=y,
                                                        n_output_channels=3,
                                                        checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                                                        grid_size=SAMPLE_TO,
                                                        checkpoints=train_checkpoints)

    evaluation_experiment.run()


if __name__ == '__main__':
    run()