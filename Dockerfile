FROM python:3.8.5-slim

# remember to expose the port your app'll be exposed on.
EXPOSE 7654

ENV GIT_PYTHON_REFRESH=quiet
RUN pip install -U pip

#COPY requirements.txt app/requirements.txt
RUN pip install sapsan

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY sapsan/examples/cnn_example.ipynb sapsan_docker_examples/
COPY sapsan/examples/krr_example.ipynb sapsan_docker_examples/
COPY sapsan/examples/GUI sapsan_docker_examples/GUI/
COPY sapsan/examples/data sapsan_docker_examples/data/
WORKDIR /sapsan_docker_examples

# run it!
ENTRYPOINT ["jupyter", "notebook", "cnn_example.ipynb", "--port=7654", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser"]
