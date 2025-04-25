import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('TkAgg')

# -------- CONFIG --------
FPS = 100  # playback speed
WIN_SCALE = 0.5
WINDOW_TITLE = "Synchronized Sensor Playback"

# Paths to synced data
SYNCED_DIR = Path("/bags/spot-aria-recordings/dlab_recordings/extracted/door_6")
MODALITY_IMAGES = {
    "aria_rgb": SYNCED_DIR / "aria_human_ego" / "camera_rgb",
    "zed_right": SYNCED_DIR / "gripper_right" / "zedm/zed_node/right/image_rect_color",
    "digit": SYNCED_DIR / "gripper_right" / "digit/right/image_raw",
    "iphone_rgb": SYNCED_DIR / "iphone_left" / "camera_rgb",
}

CSV_PATHS = {
    "position": SYNCED_DIR / "gripper_right" / "joint_states" / "data.csv",
    "data": SYNCED_DIR / "gripper_right" / "gripper_force_trigger" / "data.csv",
}

# -------- Load image timestamps --------
all_ts = []
image_buffers = {}
image_data = {}

for name, folder in MODALITY_IMAGES.items():
    image_buffers[name] = {}
    image_data[name] = {}
    ext = ".jpg" if "iphone" in name else ".png"
    for img_path in folder.glob(f"*{ext}"):
        try:
            ts = int(img_path.stem)
            image_buffers[name][ts] = img_path
            all_ts.append(ts)

            # Preload image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 120))  # <<< resize here ONCE
            image_data[name][ts] = img
        except:
            continue

# Load all CSVs
csv_data = {}
for name, csv_path in CSV_PATHS.items():
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp")
    csv_data[name] = df

# Get common sorted timeline
timeline = sorted(set(all_ts))

# -------- Setup Matplotlib grid viewer --------
fig, axs = plt.subplots(3, 2, figsize=(12, 8))  # 4 image views + 2 CSV plots
fig.canvas.manager.set_window_title(WINDOW_TITLE)
plt.subplots_adjust(hspace=0.4)

image_axes = axs[:2, :].flatten()
csv_axes = axs[2, :]

# Initialize image panels
image_names = list(image_buffers.keys())
image_ims = [ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8)) for ax in image_axes]
for ax, name in zip(image_axes, image_names):
    ax.set_title(name)
    ax.axis("off")

# Initialize CSV plots
plot_lines = {}
plot_buffers = {name: [] for name in csv_data}
plot_ts = []
for i, (name, ax) in enumerate(zip(csv_data, csv_axes)):
    ax.set_title(name)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    line, = ax.plot([], [], label=name)
    plot_lines[name] = line
    ax.legend()
    # Fix y-axis based on full range of values
    try:
        col_vals = csv_data[name][name].apply(
            lambda x: float(x.strip("[]")) if isinstance(x, str) and x.startswith("[") else float(x)
        )
        ax.set_ylim(np.min(col_vals) * 0.95, np.max(col_vals) * 1.05)
    except Exception as e:
        print(f"[!] Failed to set y-limits for {name}: {e}")

# -------- Playback loop --------
print("[▶] Starting synchronized playback with Matplotlib dashboard...")
for i, t in enumerate(timeline):
    # Update image views
    for j, name in enumerate(image_names):
        img = image_data[name].get(t, None)
        if img is not None:
            image_ims[j].set_data(img)

# Update CSV plots
plot_ts.append(t)
for name, df in csv_data.items():
    nearest = df.iloc[(df["timestamp"] - t).abs().argsort()[:1]]
    value = nearest[name].values[0]
    if isinstance(value, str) and value.startswith('['):
        try:
            value = float(value.strip('[]'))
        except:
            pass
    try:
        value = float(value)
    except:
        continue
    plot_buffers[name].append(value)
    plot_lines[name].set_data(plot_ts, plot_buffers[name])


    plt.pause(0.001)

plt.close()
