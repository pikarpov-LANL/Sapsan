TEMPLATE = """FROM python:3.8.5-slim

# remember to expose the port your app will run on
EXPOSE 7654

ENV GIT_PYTHON_REFRESH=quiet
RUN pip install -U pip

RUN pip install sapsan=={version}

# copy the notebook and data into a directory of its own (so it isn't in the top-level dir)
COPY {name}_estimator.py {name}_docker/
COPY {name}.ipynb {name}_docker/
COPY ./data/ {name}_docker/data/
WORKDIR /{name}_docker

# run it!
ENTRYPOINT ["jupyter", "notebook", "{name}.ipynb", "--port=7654", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser"]                               
"""

from sapsan import __version__

def get_template(name: str) -> str:
    return TEMPLATE.format(name=name, version=__version__.strip("v"))