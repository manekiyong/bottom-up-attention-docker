FROM nvcr.io/nvidia/pytorch:22.01-py3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

RUN pip install ray

ADD /src/detectron2 /src/detectron2

WORKDIR /src/detectron2

RUN pip install -e .

ADD /src/apex /src/apex

WORKDIR /src/apex

RUN python setup.py install

ADD /src /src

WORKDIR /src

RUN python setup.py build develop

