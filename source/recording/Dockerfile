FROM ros:noetic

# dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-rosbag \
    python3-pip \
    git \
    build-essential \
    cmake \
    nano \
    mesa-utils \
    ros-noetic-plotjuggler-ros \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Clone the repository
# IMU driver
RUN git clone --recursive https://github.com/ElettraSciComp/witmotion_IMU_ros.git src/witmotion_ros

# Modify the configuration file
RUN sed -i 's/use_native_orientation: true/use_native_orientation: false/' /catkin_ws/src/witmotion_ros/config/wt901.yml

# Install dependencies
RUN apt-get update && rosdep update
RUN rosdep install --from-paths src --ignore-src -r -y || true

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"


# Set up the environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc



# Default command (Keeps container running for interaction)
CMD ["bash"]