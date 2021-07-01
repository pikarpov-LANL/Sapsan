# to build the container 
build-container:
	@docker build . -t sapsan-docker

# to run existing the container created above
# (jupyter notebook will be started at --port==7654)
run-container:
	@docker run -p 7654:7654 sapsan-docker:latest
		
