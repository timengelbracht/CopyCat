FROM osrf/ros:noetic-desktop-full

# =========================================================
# Install Dependencies
# =========================================================

# General Dependencies
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    nano \
    git \
    python3-catkin-tools \
    python3-osrf-pycommon \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-psutil \
    python3-tk \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libopencv-dev \
    libboost-all-dev \
    cmake \
    build-essential \
    gcc \
    g++ \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-rviz \
    && rm -rf /var/lib/apt/lists/*

# Install Ceres Solver
RUN apt-get update && apt-get install -y libceres-dev

# Create a catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Clone OpenVINS
RUN git clone https://github.com/rpng/open_vins.git /catkin_ws/src/open_vins

# Build OpenVINS
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Source the workspace
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]
