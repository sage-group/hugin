FROM continuumio/miniconda3:22.11.1
COPY docker/conda_amd64.txt /tmp/conda_amd64.txt
RUN conda create --name huginenv -y --file /tmp/conda_amd64.txt && \
    conda clean -a -v -y
ENV HUGIN_VERSION='0.0.0'
ENV CONDA_DEFAULT_ENV="huginenv"
ENV CONDA_PREFIX="/opt/conda/envs/huginenv"
ENV CONDA_PROMPT_MODIFIER="(huginenv) "
ENV GDAL_DATA="/opt/conda/envs/huginenv/share/gdal"
ENV PATH="/opt/conda/envs/huginenv/bin:/opt/conda/condabin:/opt/conda/bin:${PATH}"
ENV PROJ_LIB="/opt/conda/envs/huginenv/share/proj"
RUN pip install hugin==${HUGIN_VERSION} --extra-index-url https://gitlab.dev.info.uvt.ro/api/v4/projects/513/packages/pypi/simple
