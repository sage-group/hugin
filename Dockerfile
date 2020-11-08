FROM ubuntu:18.04 AS BASE_BUILD
MAINTAINER Marian Neagul <marian@info.uvt.ro>
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install software-properties-common virtualenv && \
    add-apt-repository ppa:graphics-drivers && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list' && \
    bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list' && \
    apt-get update && apt-get  install -y wget apt-utils gdal-bin libgdal-dev python3-gdal && \
    apt-get install -y cuda-10-1 libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

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

FROM BASE_BUILD AS BASE_WITH_REQUIREMENTS

COPY requirements.txt /tmp/requirements.txt
RUN chown -R hugin /home/hugin/ && \
    virtualenv /home/hugin/venv && \
    /home/hugin/venv/bin/pip install -r /tmp/requirements.txt && \
    rm -fr ~/.cache/

FROM BASE_BUILD AS BASE_WITH_SETUP_PY
COPY --from=BASE_WITH_REQUIREMENTS /home/hugin/ /home/hugin/

ENV PATH /home/hugin/venv/bin:$PATH
COPY . /home/hugin/src
WORKDIR /home/hugin/src
RUN /home/hugin/venv/bin/python setup.py develop && \
    chown -R hugin /home/hugin/ && \
    rm -fr ~/.cache/

FROM BASE_BUILD
COPY --from=BASE_WITH_SETUP_PY /home/hugin/ /home/hugin/
ENV PATH /home/hugin/venv/bin:$PATH
USER $USER
WORKDIR /home/hugin/src
RUN cp docker/entrypoint.sh /home/hugin/ && \
    chmod +x /home/hugin/entrypoint.sh && \
    chown -R hugin /home/hugin/ && \
    rm -fr ~/.cache/
RUN apt-cache search cuda
ENTRYPOINT ["/home/hugin/entrypoint.sh"]
#SHELL [ "/bin/bash", "--login", "-c" ]
