# Use official ROS Noetic base image
FROM ros:noetic-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    ros-noetic-ros-base \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-rviz \
    python3-catkin-tools \
    libeigen3-dev \
    libboost-all-dev \
    libblas-dev \
    liblapack-dev \
    libv4l-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN apt-get update && apt-get install -y python3-dev python3-numpy
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools numpy scipy matplotlib opencv-python pyquaternion catkin_pkg

# Create a workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Clone and build Kalibr
# RUN git clone https://github.com/ethz-asl/kalibr.git /catkin_ws/src/kalibr

# Clone and build allan_variance_ros
RUN git clone https://github.com/ori-drs/allan_variance_ros.git /catkin_ws/src/allan_variance_ros

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release" 

# Source the workspace
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["bash"]
