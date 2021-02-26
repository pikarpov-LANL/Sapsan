build-container:
	@docker build . -t sapsan-docker
	@docker run -p 7654:7654 sapsan-docker

run-container:
	@docker run -p 7654:7654 sapsan-docker:latest
		
