import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from time_aligner import TimeAligner
from typing import List, Tuple
from tqdm import tqdm
from pyzbar.pyzbar import decode, ZBarSymbol
from qreader import QReader
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

class QRCodeDetectorDecoder:

    # code patterns used to split actions within the same recording
    CODES = {
        "yellow": "0I0R.119T5DY.MIXEDREALITY",
        "blue": "0H0R.119T5DY.MIXEDREALITY"
    }
    _qr_reader = QReader(model_size='s', min_confidence=0.4) 

    def __init__(self, frame_dir: Path, ext=".jpg"):
        self.frame_dir = frame_dir
        self.ext = ext
        self.qr = cv2.QRCodeDetector()
        self.logging_tag = f"[{self.__class__.__name__}]"


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

    def find_first_valid_qr(self, stride: int = 1) -> Tuple[int, int]:
        files = sorted(self.frame_dir.glob(f"*{self.ext}"),
                       key=lambda p: int(p.stem))

        hits = 0
        for frame_path in tqdm(files[::stride],
                               desc=f"Scanning {self.frame_dir.name}"):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # payload = decode_qr_opti(img)
            decoded = self._qr_reader.detect_and_decode(img, return_detections=False)
            if not decoded or not decoded[0]:
                continue
            payload = decoded[0]
            # if payload == '':
            #     # 
            #     continue

            try:
                timestamp_ns = self.parse_gopro_qr(payload)
            except ValueError as e:
                print(f"[!] Failed to parse QR: {e}")
                continue

            print(f"[{self.logging_tag}] QR in {frame_path.name} → {timestamp_ns} ns")
            return int(frame_path.stem), timestamp_ns

        print(f"[{self.logging_tag}] No valid QR found in directory.")
        return None, None
    

    def find_all_valid_interaction_qrs(self) -> List[int]:
        """
        Scan frames (every other frame for speed) and return a list of frame
        timestamps (in nanoseconds, taken from filename stems) where a decoded
        QR payload matches one of `self.CODES.values()`.

        """
        all_hits = []

        files = sorted(self.frame_dir.glob(f"*{self.ext}"),
                    key=lambda p: int(p.stem))
        if not files:
            return []

        # Optional: quick sanity read of the first image (not strictly required here)
        first_img = cv2.imread(str(files[0]))
        if first_img is None:
            return [] 

        for frame_path in tqdm(files[::3], desc=f"Scanning {self.frame_dir.name}"):
            t = int(frame_path.stem)

            img = cv2.imread(str(frame_path))

            if img is None:
                continue
            decoded = self._qr_reader.detect_and_decode(img, return_detections=False)
            if not decoded or not decoded[0]:
                continue
            payload = decoded[0]
            
            # payload = decode_qr_opti(img)
            if payload in self.CODES.values():
                all_hits.append(t)
                print(f"[{self.logging_tag}] QR in {frame_path.name} → {t} ns")

        return all_hits

def decode_qr_opti(bgr: np.ndarray) -> str:
    """
    Try to decode a QR in `bgr` using:
        1.  pyzbar (fast, needs sharp edges)
        2.  QReader CNN (robust to blur / low contrast)
    Returns the payload string, or '' if nothing was decoded.
    """
    # ---------------- common light pre‑proc -----------------------------
    # gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # unsharp mask
    # blur  = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    # sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    # # adaptive threshold
    # th = cv2.adaptiveThreshold(
    #     sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY, 31, 2)
    
    # -------------- 3) QReader deep model ------------------------------
    # QReader expects RGB, so convert & upscale to help the model
    # rgb_big = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
    #                      None, fx=2, fy=2,
    #                      interpolation=cv2.INTER_CUBIC)
    decoded = _qr_reader.detect_and_decode(bgr, return_detections=False)
    if decoded and decoded[0]:
        return decoded[0]

    # # -------------- 1) ZBar (pyzbar) -----------------------------------
    # syms = decode(th, symbols=[ZBarSymbol.QRCODE])
    # if syms:
    #     return syms[0].data.decode("utf‑8")

    return ''

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
