# used these as examples:
# https://github.com/kaust-vislab/python-data-science-project/blob/master/docker/Dockerfile
# https://hub.docker.com/r/anibali/pytorch/dockerfile
# https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
 
SHELL [ "/bin/bash", "--login", "-c" ]
 
# install utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        sudo \
        bzip2 \
        libx11-6 \
        git \
        wget \
        libjpeg-dev \
        libpng-dev \
        iproute2 \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
# change your username AND
# change your uid (run id -u to learn it) 
# and gid (run id -g to learn it)
ARG username=ssirak
ARG uid=1005
ARG gid=1005
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
 
RUN addgroup --gid $GID $USER
RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER
 
# switch to that user
# USER $USER
 
# install miniconda
ENV MINICONDA_VERSION latest
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3
# note that the version should exist in the repo https://repo.anaconda.com/miniconda/
 
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
   chmod +x ~/miniconda.sh && \
   ~/miniconda.sh -b -p $CONDA_DIR && \
   rm ~/miniconda.sh
 
# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH
 
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
 
# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash
 
# create a project directory inside user home
# you will login to this point, or jupyter will run from this point
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR
 
 
# build the conda environment
# I named it pytorch, pick another if you want
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
   conda create --name pytorch python=3.8 && \
   conda clean --all --yes
 
# activate the env and install the actual packages you need
# this doesn’t work
RUN conda activate pytorch
RUN conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install pip
# detectron-related stuff
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
RUN pip install opencv-python
RUN pip install albumentations

# why pip? because somehow jupyter's tab completion does not work with conda
RUN pip install --no-cache-dir jupyter jupyterlab
# RUN conda install -c conda-forge jupyter jupyterlab
WORKDIR /storage/ssirak/msc-complementary-labels/src 
# Start a jupyter notebook
# setting this env var helps with autocomplete
ENV SHELL=/bin/bash
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0", "--allow-root" ]
