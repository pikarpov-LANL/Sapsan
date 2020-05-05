FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
		 git \
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install numpy jupyter

COPY ./requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app
