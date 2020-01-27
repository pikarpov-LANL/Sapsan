FROM ubuntu:18.04
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install mpich
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install sklearn
RUN pip3 install h5py
RUN pip3 install mpi4py
RUN pip3 install catalyst
RUN pip3 install torch torchvision
RUN pip3 install mlflow
RUN pip3 install opencv-python
RUN apt-get -y install git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /opt/Sapsan
COPY . .

#CMD mlflow ui --port=9999 &
