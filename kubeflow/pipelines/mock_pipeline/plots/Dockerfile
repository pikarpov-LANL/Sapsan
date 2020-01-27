FROM python:3.7-slim

RUN pip install --upgrade pip
RUN pip install numpy~=1.17.0
RUN pip install scikit-learn

RUN mkdir app
COPY . /app
WORKDIR /app

ENTRYPOINT [ "python", "plot.py" ]