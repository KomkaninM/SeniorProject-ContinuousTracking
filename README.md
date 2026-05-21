# Continuous Kalman Filter Tracking

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This research project focuses on the design and implementation of a fault-tolerant, stateful multi-object tracking framework developed by 6532028621 Komkanin Maneesaisuwan ME21. The core objective of the pipeline is to enable reliable, long-duration traffic video processing on cloud platforms like Google Colab without data loss. By modifying the underlying Ultralytics library and creating a new custom workflow pipeline, the system segments video processing into iterative intervals while saving and transferring the Kalman filter tracking states across boundaries. This architecture effectively overcomes structural hardware limitations and volatile memory constraints, mitigating tracking failures such as ID switches and fragmented trajectory trails during unexpected system interruptions or network disconnections.

---

## Table of Contents

- [Continuous Kalman Filter Tracking](#continuous-kalman-filter-tracking)
  - [Table of Contents](#table-of-contents)
- [What It Does](#what-it-does)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)

---

# What It Does
This project implements a fault-tolerant, high-performance object tracking pipeline designed to execute long-duration traffic processing without data loss or system crashes.

By modifying the core Ultralytics library using pickle serialization and structuring a new segmented workflow, the framework achieves the following:

1. Ultralytics Library Modification via Pickle: Directly overrides the default BYTETracker class to serialize volatile tracking matrices (Mean, Covariance) and global index counters (BaseTrack._count) into a permanent binary .pkl file.

2. Seamless Kalman State Transfer: Preserves the underlying Kalman filter state across artificial video boundaries. This eliminates standard tracking bugs like tracking trail degradation and identity resets (ID switching) when processing long videos.

3. Iterative Chunk Processing: Subdivides incoming high-resolution video streams into discrete, manageable frame blocks to actively **prevent GPU Out-of-Memory (OOM) fatal memory overloads**.

4. Automated Crash Checkpointing: Periodically commits verified chunks alongside tracking states to Google Drive. If a network disconnect or runtime crash occurs, the pipeline reads the progress history and seamlessly resumes tracking from the last saved interval.


---

## Project Structure

```
SeniorProject-ContinuousTracking/
├── config/                      # Bytetrack Parameters Example
├── src/                         # Core source modules
│   ├── coco_classes.py          # COCO classes import
│   ├── ContinuousTracking.py    # Main Pipeline
│   └── modified_byte_tracker.py # Modified Ultralytics Library File
│   └── save_csv.py              # Save to CSV Function
                
├── SegmentVideoProcessing.ipynb # Interactive notebook for Cont. KF
```


## Getting Started

No local setup required. This project runs entirely in Google Colab.

1. Download SegmentVideoProcessing.ipynb from this repository
2. Open Google Colab and upload the notebook
3. Follow the instructions inside the notebook — all setup steps and usage guidance are documented cell by cell

---

