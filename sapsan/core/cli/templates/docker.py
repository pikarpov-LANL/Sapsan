TEMPLATE = """
FROM python:3.8.5-slim

# which port to expose
EXPOSE 7777

ENV GIT_PYTHON_REFRESH=quiet
RUN pip install -U pip
RUN pip install sapsan

WORKDIR /app/
COPY ./workdir/ /app/

#install additional packages
#COPY requirements.txt /app/
#RUN pip install -r requirements.txt


# run your notebook!
ENTRYPOINT ["jupyter", "notebook", "example.ipynb", "--port=7777", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser"] 
"""


def get_dockerfile_template(name: str) -> str:
    return TEMPLATE.format(name=name)