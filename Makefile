CURRENT_DIR = $(shell pwd)
TEMP_DIR = tmp
PROJECT_ROOD_DIR = sapsan
KUBEFLOW_PIPELINE_DIR = kubeflow
DATASET_ROOT_DIRECTORY = dataset

all: sapsan-cnn-encoder

mock: kubeflow-clean-temp-dir kubeflow-create-temp-directory kubeflow-prep-training kubeflow-prep-evaluation kubeflow-prep-plotting kubeflow-clean-temp-dir
sapsan-cnn-encoder: kubeflow-clean-temp-dir kubeflow-create-temp-directory kubeflow-cnn-encoder-pipeline kubeflow-clean-temp-dir


# CNN encoder pipeline
kubeflow-cnn-encoder-pipeline:
	@echo Building CNN ecoder pipeline images
	mkdir ${TEMP_DIR}/cnn_encoder_experiment
	mkdir ${TEMP_DIR}/cnn_encoder_experiment/${PROJECT_ROOD_DIR}
	cp requirements.txt ${TEMP_DIR}/cnn_encoder_experiment
	cp -r ${DATASET_ROOT_DIRECTORY} ${TEMP_DIR}/cnn_encoder_experiment
	cp -r ${PROJECT_ROOD_DIR} ${TEMP_DIR}/cnn_encoder_experiment
	cp -r ${KUBEFLOW_PIPELINE_DIR}/pipelines/cnn_encoder_experiment/* ${TEMP_DIR}/cnn_encoder_experiment
	cd ${TEMP_DIR}/cnn_encoder_experiment && make all


# UTILS
kubeflow-create-temp-directory:
	@echo Creating temp directory
	mkdir ${TEMP_DIR}
	@echo Directory created

kubeflow-clean-temp-dir:
	@echo Deleting temp directory
	if [ -d "${TEMP_DIR}" ]; then rm -Rf ${TEMP_DIR}; fi

# MOCK PIPELINE
kubeflow-prep-training:
	@echo Building training image
	mkdir ${TEMP_DIR}/training
	mkdir ${TEMP_DIR}/training/${PROJECT_ROOD_DIR}
	cp -r ${PROJECT_ROOD_DIR} ${TEMP_DIR}/training
	cp -r ${KUBEFLOW_PIPELINE_DIR}/pipelines/mock_pipeline/training/* ${TEMP_DIR}/training
	cd ${TEMP_DIR}/training && make all

kubeflow-prep-evaluation:
	@echo Building training image
	mkdir ${TEMP_DIR}/evaluation
	mkdir ${TEMP_DIR}/evaluation/${PROJECT_ROOD_DIR}
	cp -r ${PROJECT_ROOD_DIR} ${TEMP_DIR}/evaluation
	cp -r ${KUBEFLOW_PIPELINE_DIR}/pipelines/mock_pipeline/evaluation/* ${TEMP_DIR}/evaluation
	cd ${TEMP_DIR}/evaluation && make all

kubeflow-prep-plotting:
	@echo Building training image
	mkdir ${TEMP_DIR}/plot
	mkdir ${TEMP_DIR}/plot/${PROJECT_ROOD_DIR}
	cp -r ${PROJECT_ROOD_DIR} ${TEMP_DIR}/plot
	cp -r ${KUBEFLOW_PIPELINE_DIR}/pipelines/mock_pipeline/plots/* ${TEMP_DIR}/plot
	cd ${TEMP_DIR}/plot && make all



