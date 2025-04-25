import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from time_aligner import TimeAligner

class QRCodeDetectorDecoder:
    def __init__(self, frame_dir: Path, ext=".jpg"):
        self.frame_dir = frame_dir
        self.ext = ext
        self.qr = cv2.QRCodeDetector()
        # self.frame_index_at_detection = None
        # self.frame_timestamp_at_detection = None
        # self.qr_timestamp_at_detection = None

    def parse_gopro_qr(self, qr_text: str) -> int:

        if not qr_text.startswith("oT"):
            raise ValueError("Invalid GoPro QR format")

        # Extract timestamp part between the first "oT" and second "oT"
        try:
            timestamp_part = qr_text.split("oT")[1]
        except IndexError:
            raise ValueError("Failed to extract timestamp from QR")

        # Separate the milliseconds
        time_main, millis = timestamp_part.split(".")
        if len(time_main) != 12:
            raise ValueError("Unexpected timestamp length")

        yy = int(time_main[0:2])
        mm = int(time_main[2:4])
        dd = int(time_main[4:6])
        hh = int(time_main[6:8])
        mi = int(time_main[8:10])
        ss = int(time_main[10:12])
        ms = int(millis[:3])

        # Assume 2000-based year (could add handling for other centuries if needed)
        year = 2000 + yy

        # Build datetime object
        dt = datetime(year, mm, dd, hh, mi, ss, ms * 1000)

        # Convert to nanoseconds
        timestamp_ns = int(dt.timestamp() * 1e9)

        return timestamp_ns

    def find_first_valid_qr(self):
        frame_files = sorted(self.frame_dir.glob(f"*{self.ext}"), key=lambda p: int(p.stem))
        for frame_path in frame_files:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            val, points, _ = self.qr.detectAndDecode(img)
            if val:
                try:
                    timestamp_ns = self.parse_gopro_qr(val)
                except ValueError as e:
                    print(f"[!] Failed to parse QR code: {e}")
                    continue

                print(f"[INFO] Found valid QR in: {frame_path.name} → {timestamp_ns} ns")
                #               (device_time, utc_time)
                return (int(frame_path.stem), timestamp_ns)

        raise RuntimeError("No valid QR code found in the stream.")


if __name__ == "__main__":

    
    base_path = Path(f"/bags/spot-aria-recordings/dlab_recordings")
    rec_name = "bottle_6"
    
    

    # iphone (jpg)
    sensor_module_name = "iphone_left"
    label_rgb = f"/camera_rgb"
    frame_dir = base_path / "extracted" / rec_name / sensor_module_name / label_rgb.strip("/")

    qr_iphone = QRCodeDetectorDecoder(frame_dir, ext=".jpg")
    qr_info_iphone = qr_iphone.find_first_valid_qr()

    # zed (png)
    sensor_module_name = "gripper_right"
    label_rgb = "/zedm/zed_node/left/image_rect_color"
    frame_dir = base_path / "extracted" / rec_name / sensor_module_name / label_rgb.strip("/")

    qr_gripper = QRCodeDetectorDecoder(frame_dir, ext=".png")
    qr_info_gripper = qr_gripper.find_first_valid_qr()

    # aria (png)
    sensor_module_name = "aria_human_ego"
    label_rgb = "/camera_rgb"
    frame_dir = base_path / "extracted" / rec_name / sensor_module_name / label_rgb.strip("/")

    qr_aria = QRCodeDetectorDecoder(frame_dir, ext=".png")
    qr_info_aria = qr_aria.find_first_valid_qr()

    grip_align  = TimeAligner(qr_info_aria, qr_info_gripper)
    print(f"[INFO] Gripper to Aria delta: {grip_align.get_delta()} ns")

    iphone_align = TimeAligner(qr_info_aria, qr_info_iphone)
    print(f"[INFO] iPhone to Aria delta: {iphone_align.get_delta()} ns")


    a = 2
