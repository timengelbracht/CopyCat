#!/bin/bash

# Start udev and apply rules
service udev start
udevadm control --reload-rules
udevadm trigger
udevadm settle

# Ensure devices are correctly assigned before starting the main application
sleep 2  # Give time for udev to apply rules

# Print available devices for debugging
ls -l /dev/imu /dev/gripper

# Start the main process
exec "$@"
#exec /bin/bash
