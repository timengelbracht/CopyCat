#!/usr/bin/env bash
#
# Usage: ./record_svo_rosbag.sh <experiment_name>
# Output: /ssd/data/<experiment_name>_<YYYY-MM-DD_HH-MM-SS>.bag  +  .svo

set -euo pipefail

#############################
# 1 – configuration
#############################
TARGET_DIR="/ssd/data"               # <<< permanent storage location
mkdir -p "$TARGET_DIR"               # create if missing

#############################
# 2 – argument parsing
#############################
if [[ $# -lt 1 ]]; then
    echo "[ERROR] Please provide a base name for the recording."
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

NAME="$1"
TIMESTAMP=$(date +%F_%H-%M-%S)
FULL_NAME="${TARGET_DIR}/${NAME}_${TIMESTAMP}"

#############################
# 3 – graceful shutdown
#############################
cleanup() {
    echo -e "\n[INFO] Caught exit signal. Cleaning up…"

    if [[ -n "${ROSBAG_PID:-}" ]]; then
        echo "[INFO] Stopping rosbag (PID $ROSBAG_PID)…"
        kill "$ROSBAG_PID" 2>/dev/null || true
        wait "$ROSBAG_PID" 2>/dev/null || true
    fi

    echo "[INFO] Stopping SVO recording…"
    rosservice call /zedm/zed_node/stop_svo_recording \
        || echo "[WARN] Failed to stop SVO recording."

    echo "[INFO] Cleanup complete."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

#############################
# 4 – start recordings
#############################
echo "[INFO] Saving to: $FULL_NAME.{bag,svo}"

echo "[INFO] Waiting for /zedm/zed_node/start_svo_recording service…"
until rosservice list | grep -q /zedm/zed_node/start_svo_recording; do
    sleep 0.5
done

echo "[INFO] Starting SVO recording…"
rosservice call /zedm/zed_node/start_svo_recording "{svo_filename: '${FULL_NAME}.svo'}" \
    && echo "[✓] SVO recording started." \
    || { echo "[✗] Failed to start SVO recording."; exit 1; }

echo "[INFO] Starting rosbag recording…"
rosbag record -O "${FULL_NAME}.bag" \
    --chunksize=512 \
    --buffsize=0 \
    /digit/left/image_raw \
    /digit/right/image_raw \
    /gripper_force_trigger \
    /zedm/zed_node/imu/data_raw \
    /zedm/zed_node/imu/data \
    /zedm/zed_node/odom \
    /zedm/zed_node/pose \
    /zedm/zed_node/pose_with_covariance \
    /zedm/zed_node/depth/depth_registered \
    /zedm/zed_node/rgb/image_rect_color \
    /zedm/zed_node/rgb/camera_info \
    /zedm/zed_node/left/image_rect_color \
    /zedm/zed_node/right/image_rect_color \
    /zedm/zed_node/left_raw/image_raw_color \
    /zedm/zed_node/right_raw/image_raw_color \
    /tf \
    /joint_states \
    /tf_static &
ROSBAG_PID=$!

wait "$ROSBAG_PID"

