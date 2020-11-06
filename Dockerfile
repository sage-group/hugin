FROM ubuntu:20.04
MAINTAINER Marian Neagul <marian@info.uvt.ro>
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get  install -y wget apt-utils gdal-bin libgdal-dev python3-gdal

ARG username=hugin
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Hugin User" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

RUN apt-get install -y virtualenv
COPY requirements.txt /tmp/requirements.txt
RUN chown -R hugin /home/hugin/ && \
    virtualenv /home/hugin/venv && \
    /home/hugin/venv/bin/pip install -r /tmp/requirements.txt
ENV PATH /home/hugin/venv/bin:$PATH
COPY . /home/hugin/src
RUN chown -R hugin /home/hugin/
USER $USER
#RUN chown -R $USER /opt/hugin/
WORKDIR /home/hugin/src
RUN pwd
RUN whoami
RUN ls -la
RUN /home/hugin/venv/bin/python setup.py develop
RUN cp docker/entrypoint.sh /home/hugin/
RUN chmod +x /home/hugin/entrypoint.sh
ENTRYPOINT /home/hugin/entrypoint.sh
#SHELL [ "/bin/bash", "--login", "-c" ]
