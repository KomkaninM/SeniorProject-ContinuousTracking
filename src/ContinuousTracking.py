import os
import json
import time
import shutil
import subprocess
import yaml
import cv2
from pathlib import Path
import logging
from ultralytics import YOLO
from datetime import datetime

try:
    from save_csv import save_csv
except ImportError:
    # Fallback if running from a notebook where paths might be different
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from save_csv import save_csv

class ContinuousTracking:
    """
    A robust video tracking pipeline that processes video in chunks,
    saves progress atomically to Drive, and can resume from crashes
    without data loss.
    """
    def __init__(self, video_path, output_dir, tracker_config_path, model_name = "yolov9e.pt", conf=0.05, iou=0.45, imgsz = 1920, print_current_frame = True):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.tracker_yaml = tracker_config_path # YAML File path
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.print_current_frame = print_current_frame

        self.video_name = Path(video_path).stem
        self.progress_file = os.path.join(output_dir, f"{self.video_name}_progress.json")
        self.tracker_pickle = os.path.join(output_dir, f"{self.video_name}_tracker_state.pkl")
        self.csv_dir = os.path.join(output_dir, "csv") # Create sub-folder, csv

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ensure_dirs()

        self.log_file = os.path.join(output_dir, f"{self.video_name}_run_{current_time}.log")
        self.logger = self._setup_logger()

        # Colors for visualization
        self.colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
            (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128)
        ]


        self._sync_yaml_config()
        self.state = self._load_state()

        self.logger.info("Continuous Tracking Initialized.")
        self.logger.info(f"Progress File Directory: {self.progress_file}")
        self.logger.info(f"Pickle File Directory: {self.tracker_pickle}")
        self.logger.info(f"CSV Directory: {self.csv_dir}")
        self.logger.info(f"Output Directory: {self.output_dir}")

    def _setup_logger(self):
        """Sets up a logger that writes to Console AND Google Drive."""
        logger = logging.getLogger(self.video_name)
        logger.setLevel(logging.INFO)

        # Prevent "Duplicate Logs" when re-running cells in Colab
        if logger.handlers:
            return logger

        # Formatter (Time - Level - Message)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Handler A: File Handler (Writes to Google Drive)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Handler B: Stream Handler (Writes to Colab Output)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _ensure_dirs(self):
        "Checking is the directory is exist, if not create one"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def _sync_yaml_config(self):
        """Forces the tracker YAML to match our pickle path and use the YAML's interval."""
        with open(self.tracker_yaml, 'r') as f:
            config = yaml.safe_load(f)

        # Syncing the pickle file location to match our structure
        # Syncing frame per chunk of JSON to match Pickle
        self.frames_per_chunk = config.get('save_interval', 5000)
        config['state_file'] = self.tracker_pickle
        config['save_interval'] = self.frames_per_chunk

        with open(self.tracker_yaml, 'w') as f:
            yaml.dump(config, f)
        print(f"Synced Config: Chunk Size = {self.frames_per_chunk}")
        print(f"Pickle Path Updated = {self.tracker_pickle}")

    def _load_state(self) :
        """Loads existing progress or creates new."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                self.logger.info("Found previous progress file. Resuming...")
                return json.load(f)
        else :
            self.logger.info("No progress found. Starting fresh.")
            inital_state = {
                "input_video_path": str(self.video_path),
                "fps": None, "width": None, "height": None, "total_frames": None,
                "frames_per_chunk": self.frames_per_chunk,  # adjust if you like
                "next_frame": 0,
                "chunk_index": 0,
                "chunks": [],
                "done": False,
                "last_saved_at": None
            }
            return inital_state

    def _save_state(self):
        self.state["last_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.progress_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _process_chunk(self, cap, model):
        """ Processes ONE single chunk with Atomic Save
            (Save in local storage first, then copy to drive)."""
        idx = self.state["chunk_index"]
        start_frame = self.state["next_frame"]

        chunk_name = f"{self.video_name}_chunk_{idx:03d}.mp4"
        local_path = os.path.join("/content", chunk_name)      # Local Path
        drive_path = os.path.join(self.output_dir, chunk_name) # Drive Path
        csv_name = f"{self.video_name}_chunk_{idx:03d}.csv"
        local_csv_path = os.path.join("/content", csv_name)
        drive_csv_path = os.path.join(self.csv_dir, csv_name)
        # If we are retrying a chunk, ensure we don't append to old garbage files
        if os.path.exists(local_path):
            os.remove(local_path)
        if os.path.exists(local_csv_path):
            os.remove(local_csv_path)

        self.logger.info(f"\nProcessing Chunk {idx:03d} (Frame {start_frame} to {start_frame + self.frames_per_chunk})...")

        # Init Writer (Writing to Local Disk)
        writer = cv2.VideoWriter(local_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.state["fps"], (self.state["width"], self.state["height"]))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        frames_processed = 0

        while frames_processed < self.frames_per_chunk:
            ret, frame = cap.read()
            if not ret:
                self.state["done"] = True # Video Ended naturaly
                break

            # --- DETECTION & TRACKING ---
            results = model.track(frame,
                                  tracker = self.tracker_yaml,
                                  classes = 3,
                                  persist=True,
                                  save = False,
                                  conf = self.conf, iou = self.iou, line_width = 2,
                                  imgsz = self.imgsz)

            # --- DRAWING BB---
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    track_id = int(box[4])
                    score = float(box[5])
                    color = self.colors[track_id % len(self.colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            writer.write(frame)
            current_abs_frame = start_frame + frames_processed
            save_csv(results, current_abs_frame, local_csv_path, save_conf=False)
            if self.print_current_frame:
                print(f"Processed {current_frame} / {self.state['total_frames']} frames")
            current_frame += 1
            frames_processed += 1

        # --- SAVE TO LOCAL ---
        writer.release()

        self.logger.info(f"Uploading Chunk {idx} to Local Drive...")
        if os.path.exists(local_path):
            # Move file to Drive
            shutil.move(local_path, drive_path)

        self.logger.info(f"Uploading CSV Chunk {idx}...")
        if os.path.exists(local_csv_path):
            shutil.move(local_csv_path, drive_csv_path)

        # Update State
        self.state["chunks"].append({"file": chunk_name, "start": start_frame, "end": current_frame - 1})
        self.state["chunk_index"] += 1
        self.state["next_frame"] = current_frame
        self._save_state()
        self.logger.info(f"Chunk {idx} committed successfully.")

    def _combine_videos(self):
        """Merges all chunks using FFmpeg."""
        self.logger.info("\n All chunks done. Combining video...")
        chunk_files = sorted([
            f for f in os.listdir(self.output_dir)
            if f.endswith(".mp4") and f"{self.video_name}_chunk_" in f
        ])

        if not chunk_files:
            self.logger.warning("⚠️ No chunks found to combine.")
            return

        list_path = os.path.join(self.output_dir, "ffmpeg_list.txt")
        final_path = os.path.join(self.output_dir, f"{self.video_name}_combined_output.mp4")

        with open(list_path, "w") as f:
            for cf in chunk_files:
                f.write(f"file '{os.path.join(self.output_dir, cf)}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", final_path, "-y"
        ]
        subprocess.run(cmd, check=True)
        self.logger.info(f"FINAL VIDEO SAVED: {final_path}")

    def run(self):
        """Main Loop."""
        if self.state["done"]:
            self.logger.info("Processing already complete. Running combiner...")
            self._combine_videos()
            return

        self.logger.info("Loading YOLO Model...")
        model = YOLO(self.model_name)

        cap = cv2.VideoCapture(str(self.video_path))

        # Init Video Info if not present
        if self.state["total_frames"] is None:
            self.state["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.state["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.state["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
            self.state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._save_state()

        # Process Chunks Loop
        while not self.state["done"]:
            self._process_chunk(cap, model)

        cap.release()
        self.logger.info("All frames processed!")
        self._combine_videos()
