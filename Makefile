# to build and start the container 
# (jupyter notebook will be started at --port==7654)
build-container:
	@docker build . -t sapsan-docker
	@docker run -p 7654:7654 sapsan-docker

# to run existing the container created above
run-container:
	@docker run -p 7654:7654 sapsan-docker:latest
		
