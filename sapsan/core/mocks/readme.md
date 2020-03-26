Example of mocked pipeline
====

Note: For kubeflow skeleton

data extraction -> training -> inference evaluation -> plotting

```python
from sapsan.core.mocks.fake_dataset import FakeDataset
from sapsan.core.mocks.regression_estimator import LinearRegressionEstimator, LinearRegressionEstimatorConfiguration
from sapsan.core.mocks.fake_experiment import FakeTrainingExperiment, FakeInferenceExperiment
from sapsan.utils.plot import PlotUtils

dataset = FakeDataset()
estimator = LinearRegressionEstimator(LinearRegressionEstimatorConfiguration())
training_experiment = FakeTrainingExperiment("training_experiment", estimator, dataset)
inference_experiment = FakeInferenceExperiment("inference_experiment", dataset, estimator)

training_experiment.run()
training_experiment.get_report()

result = inference_experiment.run()
inference_experiment.get_report()

plot_result = PlotUtils.plot_pdf(result)
plot_result.show()
```