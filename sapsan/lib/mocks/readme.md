Example of mocked pipeline
====

Note: For kubeflow skeleton

data extraction -> training -> inference evaluation -> plotting

```python
from sapsan.lib.mocks import FakeDataset
from sapsan.lib.mocks import LinearRegressionEstimator, LinearRegressionEstimatorConfiguration
from sapsan.lib.mocks import FakeTrainingExperiment, FakeInferenceExperiment
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