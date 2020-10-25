FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER Marian Neagul <marian@info.uvt.ro>
ENV DEBIAN_FRONTEND noninteractive

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


COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get -y install virtualenv apt-utils && \
    apt-get update && apt-get  install -y apt-utils gdal-bin libgdal-dev python3-gdal && \
    rm -rf /var/lib/apt/lists/* && \
    chown -R hugin /home/hugin/ && \
    virtualenv -p python3 /home/hugin/venv && \
    /home/hugin/venv/bin/pip install -r /tmp/requirements.txt && \
    rm -fr /home/hugin/.cache/  /root/.cache/ && \
    chown -R hugin /home/hugin/ && \
    apt-get purge --autoremove -y libgdal-dev apt-utils

COPY --from=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5 /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5
COPY --from=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so.7

ENV PATH /home/hugin/venv/bin:$PATH
COPY . /home/hugin/src
WORKDIR /home/hugin/src
RUN /home/hugin/venv/bin/python setup.py develop && \
    rm -fr /home/hugin/.cache/

#FROM BASE_BUILD
#COPY --from=BASE_WITH_SETUP_PY /home/hugin/ /home/hugin/

ENV PATH /home/hugin/venv/bin:$PATH
WORKDIR /home/hugin/src
RUN cp docker/entrypoint.sh /home/hugin/ && \
    chmod +x /home/hugin/entrypoint.sh && \
    rm -fr /home/hugin/.cache/

ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/:/usr/local/cuda-10.1/compat/
RUN ln -s $(readlink -f /usr/local/cuda-10.1/compat/libcuda.so)  /usr/lib/x86_64-linux-gnu/libcuda.so && \
    ln -s $(readlink -f /usr/local/cuda-10.1/compat/libcuda.so)  /usr/lib/x86_64-linux-gnu/libcuda.so.1
USER $USER
ENTRYPOINT ["/home/hugin/entrypoint.sh"]
CMD ["train"]
#SHELL [ "/bin/bash", "--login", "-c" ]
