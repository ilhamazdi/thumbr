# Thumbr — Video Thumbnail Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python script that generates visual thumbnail grids from video files. It automatically captures evenly distributed frames from the video and stitches them together into a single image, creating a comprehensive preview with detailed video information and timestamps.

## Features

- **Automated Frame Capture:** Extracts evenly distributed frames across the video's duration.
- **Customizable Grid Layout:** Supports flexible grid sizes (e.g., 3x3, 4x4) for arranging thumbnails.
- **Aspect Ratio Preservation:** Maintains the original aspect ratio of video frames in the thumbnails.
- **Detailed Information Header:** Generates a header with essential video details:
    - Filename
    - Duration
    - Resolution (Width x Height)
    - Frame Rate (FPS)
    - File Size
- **Timestamp Overlays:** Each captured frame is overlaid with its corresponding timestamp.
- **Watermarking:** Adds a customizable watermark to the generated thumbnail.
- **Progress Tracking:** Provides a progress bar during thumbnail generation.

## Example Output

![Example thumbnail grid generated from Big Buck Bunny](output/big_buck_bunny_720p_h264_thumbnail.jpg)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ilhamazdi/thumbr.git
    cd thumbr
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the `main.py` script from your terminal.

**Basic Usage:**

```bash
python main.py "path/to/your/video.mp4"
```
This will generate a thumbnail image (e.g., `your_video_thumbnail.jpg`) in the same directory as the input video.

**Custom Output Path:**

Specify a different output path and filename using the `-o` or `--output` flag:

```bash
python main.py "path/to/your/video.mp4" -o "path/to/desired/output/my_video_preview.png"
```

**Custom Grid Size:**

Change the grid dimensions (rows x columns) using the `-g` or `--grid` flag (default is `3x3`):

```bash
python main.py "path/to/your/video.mp4" -g 4x5
```

## Requirements

-   Python 3.6+
-   `opencv-python` (for video processing and broad codec support)
-   `numpy` (for numerical operations)
-   `Pillow` (PIL) (for image manipulation)
-   `tqdm` (for progress bars)

## License

MIT License

Copyright (c) 2025 ilhamazdi