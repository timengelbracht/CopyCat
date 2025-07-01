# Base Ubuntu 22.04
FROM ubuntu:22.04

# Non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Basic dev tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    cmake

# Install CUDA manually
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y cuda-toolkit-12-2

# Install libraries you need
RUN apt-get install -y \
    libeigen3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    libx11-dev \
    libglfw3-dev \
    libglew-dev \
    libtbb-dev \
    libjsoncpp-dev \
    libspdlog-dev \
    libfmt-dev \
    wget unzip ffmpeg
    
RUN apt-get install -y libc++-dev libc++abi-dev clang

# Install Python3 and pip
RUN apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python  # Optional alias

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip3 install \
    numpy \
    matplotlib \
    opencv-python 
    
RUN pip3 install numpy matplotlib opencv-python
RUN python3 -m pip install projectaria_client_sdk --no-cache-dir
RUN python3 -m pip install moviepy==1.0.3 projectaria_tools
RUN python3 -m pip install aria_studio --no-cache-dir

# Environment variables for Open3D and CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Default command
CMD ["/bin/bash"]





