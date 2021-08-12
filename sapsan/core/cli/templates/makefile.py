TEMPLATE = '''# to build the container 
build-container:
    @docker build . -t {name}-docker

# to run existing the container created above
# (jupyter notebook will be started at --port==7654)
run-container:
    @docker run -p 7654:7654 {name}-docker:latest
'''

def get_template(name: str) -> str:
    return TEMPLATE.format(name=name)