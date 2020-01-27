IMAGE_NAME = mock_training
DOCKER_REGISTRY ?= icekhan
TAG ?= latest

CURRENT_DIR = $(shell pwd)

DATA_SIZE = 10

all: build-container clean test-container push-container
release: build-container clean push-container
release-raw: build-container clean push-container

# DOCKER_BUILD_OPTS env var can be used to configure build step
build-container:
	@echo Started building image
	docker build ${DOCKER_BUILD_OPTS} -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

test-container:
	@echo Performing container run
	docker run $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)\
		--data-size $(DATA_SIZE)

push-container:
	@echo Pushing image to the registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

clean:
	@echo Cleaning folder
	rm -rf $(DATA_PATH) *.log