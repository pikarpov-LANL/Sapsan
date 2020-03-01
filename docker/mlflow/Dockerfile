FROM continuumio/miniconda3

RUN apt-get update -y && \
    apt-get install -y

RUN pip install mlflow===1.1.0

RUN pip install pysftp
RUN pip install boto3

RUN mkdir /filestore
