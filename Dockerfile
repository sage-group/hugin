FROM continuumio/miniconda3:4.10.3p0
#RUN conda create --name huginenv -y -c conda-forge gdal
COPY docker/conda_amd64.txt /tmp/conda_amd64.txt
RUN conda create --name huginenv -y --file /tmp/conda_amd64.txt && \
    conda clean -a -v -y
ENV CONDA_DEFAULT_ENV="huginenv"
ENV CONDA_PREFIX="/opt/conda/envs/huginenv"
ENV CONDA_PROMPT_MODIFIER="(huginenv) "
ENV GDAL_DATA="/opt/conda/envs/huginenv/share/gdal"
ENV PATH="/opt/conda/envs/huginenv/bin:/opt/conda/condabin:/opt/conda/bin:${PATH}"
ENV PROJ_LIB="/opt/conda/envs/huginenv/share/proj"
