from sapsan.general.data.flatten_from_3d import Flatten3dDataset
from sapsan.general.data.jhtdb_dataset import JHTDB128Dataset, Equidistance3dSampling
from sapsan.general.estimator import KrrEstimator, KrrEstimatorConfiguration
from sapsan.general.estimator.cnn.spacial_3d_encoder import Spacial3dEncoderNetworkEstimator, \
    Spacial3dEncoderNetworkEstimatorConfiguration
from sapsan.general.experiment import TrainingExperiment, EvaluationExperiment, FakeExperimentBackend


def run():
    dataset_root_dir = "/Users/icekhan/Documents/development/myprojects/sapsan/repo/Sapsan/dataset"
    CHECKPOINT_DATA_SIZE = 128
    SAMPLE_TO = 32
    GRID_SIZE = 16
    features = ['u', 'b', 'a',
                'du0', 'du1', 'du2',
                'db0', 'db1', 'db2',
                'da0', 'da1', 'da2']
    labels = ['tn']

    sampler = Equidistance3dSampling(CHECKPOINT_DATA_SIZE, SAMPLE_TO)
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
                           grid_size=GRID_SIZE,
                            checkpoint_data_size=CHECKPOINT_DATA_SIZE,
                            sampler=sampler).load()

    print(x.shape)
    print(y.shape)

    #
    # training_experiment = TrainingExperiment(name=training_experiment_name,
    #                                          backend=FakeExperimentBackend(training_experiment_name),
    #                                          model=estimator,
    #                                          inputs=x, targets=y)
    # training_experiment.run()
    #
    #
    #
    #
    # #%%
    #
    # training_experiment_name = "Training experiment"
    # estimator = Spacial3dEncoderNetworkEstimator(
    #     config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=1, grid_dim=GRID_SIZE)
    # )
    # x, y = JHTDB128Dataset(path=dataset_root_dir,
    #                        features=features,
    #                        labels=labels,
    #                        checkpoints=[0.0],
    #                        grid_size=GRID_SIZE,
    #                        checkpoint_data_size=CHECKPOINT_DATA_SIZE,
    #                        sampler=sampler).load()
    #
    # training_experiment = TrainingExperiment(name=training_experiment_name,
    #                                 backend=FakeExperimentBackend(training_experiment_name),
    #                                 model=estimator,
    #                                 inputs=x, targets=y)
    # training_experiment.run()
    #
    # x, y = JHTDB128Dataset(path=dataset_root_dir,
    #                        features=features,
    #                        labels=labels,
    #                        checkpoints=[0.01],
    #                        grid_size=GRID_SIZE,
    #                        checkpoint_data_size=CHECKPOINT_DATA_SIZE,
    #                        sampler=sampler).load()
    #
    # model = Spacial3dEncoderNetworkEstimator(
    #     config=Spacial3dEncoderNetworkEstimatorConfiguration(n_epochs=1)
    # )
    #
    #
    # evaluation_experiment_name = "Evaluation experiment"
    # evaluation_experiment = EvaluationExperiment(name=evaluation_experiment_name,
    #                                              backend=FakeExperimentBackend(evaluation_experiment_name),
    #                                              model=training_experiment.model,
    #                                              inputs=x, targets=y,
    #                                              n_output_channels=3,
    #                                              grid_size=GRID_SIZE)
    #
    # evaluation_experiment.run()
    #







if __name__ == '__main__':
    run()