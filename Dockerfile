# Use an NVIDIA CUDA image as the base
# FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
FROM nvcr.io/nvidia/l4t-cuda:12.2.12-devel

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
# ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
# ENV LANG C.UTF-8

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
# ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV CUDA_HOME="/usr/local/cuda"
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.6 8.7 8.9"

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
    && apt-get update \
    && apt-get install -y apt-transport-https ca-certificates \
    && apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
    && apt-get install -y bash-completion \
    && apt-get install -y guvcview \
    && rm -rf /var/lib/apt/lists/* 
    # && useradd -ms /bin/bash user \
    # && echo "user:user" | chpasswd && adduser user sudo \
    # && echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6 python3-pip  -y

# RUN set -x \
#     && apt-get update \
#     && apt-get install -y software-properties-common \
#     && add-apt-repository ppa:deadsnakes/ppa \
#     && apt-get update \
#     && apt-get install -y python3.11 python3.11-venv python3.11-dev \
#     && apt-get install -y python3.11-tk

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py

WORKDIR /home/user

# RUN git clone https://github.com/NVIDIA-AI-IOT/nanosam && \
#     cd nanosam && \
#     python3 setup.py develop --user
# RUN python3 -m pip install transformers

RUN git clone https://github.com/facebookresearch/segment-anything-2 && \
    cd segment-anything-2 && \
    pip3 install -e . -v && \
    pip3 install -e ".[demo]" && \
    cd checkpoints && ./download_ckpts.sh && cd ..

RUN pip3 install jupyterlab ipywidgets jupyterlab_widgets ipycanvas

# 下载PyTorch的CUDA 11.4版本
# RUN python3 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -c "import torch; print(torch.cuda.is_available());"

# RUN usermod -aG dialout user
# USER user
# STOPSIGNAL SIGTERM
# RUN nvcc --version

#CMD sudo service ssh start && /bin/bash
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

# docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 sam2:latest
# docker run --restart always -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 sam2:latest
# docker build -t sam2 . && docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 sam2:latest python3 -c "import torch; print(torch.cuda.is_available());"
