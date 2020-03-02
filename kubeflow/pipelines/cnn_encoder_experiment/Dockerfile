FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install numpy~=1.17.0
RUN pip install scikit-learn

RUN mkdir app

COPY ./dataset /app/dataset

COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt

RUN apt-get update
RUN apt-get install -y libgtk2.0-dev
RUN apt install -y git-all

COPY . /app
WORKDIR /app

ENTRYPOINT [ "python", "experiment.py" ]
