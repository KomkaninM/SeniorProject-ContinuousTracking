# Continuous Tracking

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

A computer vision system for continuous object tracking in video streams, combining video segmentation with real-time tracking pipelines. Built as a senior project, this system processes video input to detect, segment, and persistently track objects across frames.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
- [Notebook](#notebook)
- [Getting Help](#getting-help)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [License](#license)

---

## What It Does

This project implements a continuous tracking pipeline that:

1. **Ingests** video files or live streams as input
2. **Segments** frames to isolate regions of interest using the `SegmentVideoProcessing` module
3. **Tracks** detected objects persistently across frames, maintaining identity through occlusion and scene changes
4. **Outputs** annotated video or structured tracking data for downstream use

The system is designed for research and prototyping use cases where robust, continuous object re-identification across video segments is needed.

---

## Key Features

- **Segment-based video processing** — preprocessing pipeline that splits video into segments for efficient frame-level analysis
- **Continuous identity tracking** — objects are assigned persistent IDs across segment boundaries
- **Configurable pipeline** — tracking parameters, model paths, and I/O settings are externalized in `config/`
- **Jupyter notebook exploration** — `SegmentVideoProcessing.ipynb` provides an interactive walkthrough of the core segmentation logic
- **Pure Python implementation** — minimal dependencies, easy to integrate into existing CV pipelines

---

## Project Structure

```
SeniorProject-ContinuousTracking/
├── config/                     # Configuration files (model params, I/O paths, thresholds)
├── src/                        # Core source modules
│   ├── tracking/               # Object tracking logic and ID assignment
│   ├── segmentation/           # Video segmentation and frame preprocessing
│   └── utils/                  # Helper utilities (I/O, visualization, metrics)
├── SegmentVideoProcessing.ipynb # Interactive notebook for segmentation exploration
└── LICENSE
```

> **Note:** Explore `config/` to understand all tunable parameters before running the pipeline.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- A CUDA-capable GPU is recommended for real-time tracking, but the pipeline runs on CPU as well

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/KomkaninM/SeniorProject-ContinuousTracking.git
   cd SeniorProject-ContinuousTracking
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If `requirements.txt` is not present, install the core dependencies manually:
   >
   > ```bash
   > pip install opencv-python numpy torch torchvision jupyter
   > ```

### Configuration

All runtime settings live in the `config/` directory. Review and update the relevant config file before running:

```
config/
├── model.yaml       # Model weights paths and architecture settings
├── tracking.yaml    # Tracker hyperparameters (IOU threshold, max age, etc.)
└── paths.yaml       # Input/output video paths and result directories
```

Edit `config/paths.yaml` to point to your input video:

```yaml
input_video: "data/your_video.mp4"
output_dir:  "results/"
```

### Usage

Run the main tracking pipeline from the project root:

```bash
python src/main.py --config config/tracking.yaml
```

To specify a custom input video at runtime:

```bash
python src/main.py --config config/tracking.yaml --input path/to/video.mp4
```

Results (annotated video and/or tracking logs) are written to the `output_dir` specified in your config.

---

## Notebook

The `SegmentVideoProcessing.ipynb` notebook provides a step-by-step walkthrough of the video segmentation stage — useful for understanding the preprocessing logic or experimenting with new segmentation strategies.

Launch it with:

```bash
jupyter notebook SegmentVideoProcessing.ipynb
```

---

## Getting Help

- **Bug reports & feature requests** — open an issue on the [GitHub Issues](https://github.com/KomkaninM/SeniorProject-ContinuousTracking/issues) page
- **Questions** — use GitHub Issues with the `question` label
- **Computer vision references** used in this project:
  - [OpenCV Documentation](https://docs.opencv.org/)
  - [PyTorch Vision](https://pytorch.org/vision/stable/index.html)

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with clear messages
4. Push to your fork and open a Pull Request against `main`

Please keep PRs focused — one feature or fix per PR. If you're planning a large change, open an issue first to discuss the approach.

---

## Maintainers

| Name | GitHub |
|------|--------|
| Komkanin M. | [@KomkaninM](https://github.com/KomkaninM) |

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
