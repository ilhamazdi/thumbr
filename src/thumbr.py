import cv2
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import signal
import threading
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from src.utils import (
    load_font,
    load_italic_font,
    generate_info_text,
    get_logger,
    validate_video_file,
    discover_videos,
    VIDEO_EXTENSIONS,
)


# Thread-safe event for graceful shutdown
_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    _shutdown_event.set()


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested.

    Returns:
        True if shutdown signal was received, False otherwise.
    """
    return _shutdown_event.is_set()


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@dataclass
class ThumbnailResult:
    """Result of thumbnail generation for a single video."""

    video_path: str
    output_path: Optional[str]
    success: bool
    error: Optional[str] = None
    skipped: bool = False


class Thumbr:
    def __init__(
        self,
        grid_size: Tuple[int, int] = (3, 3),
        logger: Optional[get_logger] = None,
        max_width: int = 1920,
        resize_frames_at_capture: bool = True,
        max_resize_dimension: int = 640,
    ):
        """Initialize the thumbnailer with grid configuration.

        Args:
            grid_size: Tuple of (rows, columns) for the thumbnail grid.
                       Default is (3, 3).
            logger: Optional logger instance. If None, creates a new one.
            max_width: Maximum width of the output image.
            resize_frames_at_capture: If True, resize frames during capture
                                      to reduce memory footprint.
            max_resize_dimension: Maximum dimension when resizing frames.
        """
        self.grid_size = grid_size
        self.total_frames = grid_size[0] * grid_size[1]
        self.max_width = max_width
        self.resize_frames_at_capture = resize_frames_at_capture
        self.max_resize_dimension = max_resize_dimension

        # Initialize fonts with cross-platform fallback
        self.font = load_font()
        self.font_italic = load_italic_font()

        # Set up logger
        self.logger = logger or get_logger("thumbr")

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _retry_on_failure(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic for transient failures.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result of the function call.

        Raises:
            Last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    self.logger.debug(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.debug(f"All {self.max_retries} attempts failed")

        raise last_exception

    def get_video_info_and_frames(
        self, video_path: str, progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[dict, List[np.ndarray]]:
        """Get video information and capture frames in a single video open.

        This method combines get_video_info() and capture_frames() to open
        the video only once, improving performance.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (video_info dict, list of captured frames).
        """
        if progress_callback:
            progress_callback("Opening video")

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = os.path.getsize(video_path)

            video_info = {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "file_size": file_size,
                "filename": os.path.basename(video_path),
            }

            if progress_callback:
                progress_callback("Capturing frames")

            # Capture frames
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = total_frames // (self.total_frames + 1)

            # Calculate target size for frame resizing (if enabled)
            target_width = None
            target_height = None
            if self.resize_frames_at_capture:
                aspect_ratio = width / height
                if width > height:
                    target_width = self.max_resize_dimension
                    target_height = int(target_width / aspect_ratio)
                else:
                    target_height = self.max_resize_dimension
                    target_width = int(target_height * aspect_ratio)

            for i in range(self.total_frames):
                frame_pos = (i + 1) * interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize if enabled
                    if target_width and target_height:
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            (target_width, target_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    frames.append(frame_rgb)

                    # Explicitly delete original frame to free memory
                    del frame

            # Ensure video capture is released
            cap.release()
            cap = None

            return video_info, frames

        finally:
            # Ensure capture is always released
            if cap is not None:
                cap.release()

    def get_video_info(self, video_path: str) -> dict:
        """Get video information including resolution, duration, and file size.

        Note: This method opens the video separately. For better performance,
        use get_video_info_and_frames() which combines both operations.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing video information.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        file_size = os.path.getsize(video_path)

        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "file_size": file_size,
            "filename": os.path.basename(video_path),
        }

    def capture_frames(self, video_path: str) -> List[np.ndarray]:
        """Capture evenly distributed frames from the video.

        Note: This method opens the video separately. For better performance,
        use get_video_info_and_frames() which combines both operations.

        Args:
            video_path: Path to the video file.

        Returns:
            List of captured frames as numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // (self.total_frames + 1)

        for i in range(self.total_frames):
            frame_pos = (i + 1) * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def create_info_header(self, video_info: dict, width: int) -> Image.Image:
        """Create an information header for the thumbnail.

        Args:
            video_info: Dictionary containing video information.
            width: Width of the header image.

        Returns:
            PIL Image containing the header information.
        """
        # Calculate header height based on number of info lines
        num_lines = 5
        line_height = 20
        header_height = num_lines * line_height + 20  # Add padding

        header = Image.new("RGB", (width, header_height), "black")
        draw = ImageDraw.Draw(header)

        # Use the class font
        font = self.font

        # Create info lines using shared method
        info_lines = generate_info_text(
            video_info, include_frame_rate=True, include_codec=False
        )

        # Draw each line
        y_position = 10
        for line in info_lines:
            draw.text((10, y_position), line, font=font, fill="white")
            y_position += line_height

        return header

    def _calculate_dimensions(self, video_info: dict, max_width: int) -> dict:
        """Calculate dimensions for the thumbnail grid."""
        padding = 30  # padding around the entire image
        spacing = 20  # spacing between frames

        # Calculate thumbnail dimensions while maintaining aspect ratio
        available_width = (
            max_width - (2 * padding) - (spacing * (self.grid_size[1] - 1))
        )
        frame_width = available_width // self.grid_size[1]
        aspect_ratio = video_info["width"] / video_info["height"]
        frame_height = int(frame_width / aspect_ratio)

        # Calculate total dimensions including padding and spacing
        grid_width = (
            (frame_width * self.grid_size[1])
            + (spacing * (self.grid_size[1] - 1))
            + (2 * padding)
        )
        grid_height = (frame_height * self.grid_size[0]) + (
            spacing * (self.grid_size[0] - 1)
        )

        # Calculate info section height based on content
        info_font_size = int(grid_width * 0.015)  # 1.5% of width for media info
        line_spacing = info_font_size + 5
        num_info_lines = 5  # Number of info text lines
        info_height = (line_spacing * num_info_lines) + (
            padding * 2
        )  # Add padding top and bottom

        total_height = grid_height + info_height + (2 * padding)

        # Add extra height for watermark
        watermark_padding = padding * 2  # Double padding for watermark section
        total_height = total_height + watermark_padding

        return {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "info_height": info_height,
            "total_height": total_height,
            "padding": padding,
            "spacing": spacing,
            "info_font_size": info_font_size,
            "line_spacing": line_spacing,
            "watermark_padding": watermark_padding,
        }

    def _create_info_section(self, video_info: dict, dimensions: dict) -> Image.Image:
        """Create the info section of the thumbnail."""
        grid_width = dimensions["grid_width"]
        info_height = dimensions["info_height"]
        padding = dimensions["padding"]
        info_font_size = dimensions["info_font_size"]
        line_spacing = dimensions["line_spacing"]

        # Create info section image
        info_section = Image.new("RGB", (grid_width, info_height), "white")
        draw = ImageDraw.Draw(info_section)

        info_font = self.font.font_variant(size=info_font_size)

        # Create info text using shared method
        info_text = generate_info_text(
            video_info, include_frame_rate=True, include_codec=False
        )

        y_position = padding
        for text in info_text:
            draw.text((padding, y_position), text, font=info_font, fill="black")
            y_position += line_spacing

        return info_section

    def _process_frame_with_timestamp(
        self, frame: np.ndarray, timestamp: float, dimensions: dict
    ) -> Image.Image:
        """Process a single frame with timestamp overlay."""
        frame_width = dimensions["frame_width"]
        frame_height = dimensions["frame_height"]

        # Convert frame to PIL Image
        pil_frame = Image.fromarray(frame)
        resized_frame = pil_frame.resize(
            (frame_width, frame_height), Image.Resampling.LANCZOS
        )

        # Calculate timestamp
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        timestamp_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create timestamp font
        timestamp_font_size = int(frame_width * 0.06)
        timestamp_font = self.font.font_variant(size=timestamp_font_size)

        # Create a new image for the frame with timestamp
        frame_with_timestamp = resized_frame.copy()
        frame_draw = ImageDraw.Draw(frame_with_timestamp)

        # Calculate timestamp text dimensions
        timestamp_bbox = frame_draw.textbbox(
            (0, 0), timestamp_text, font=timestamp_font
        )
        timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
        timestamp_height = timestamp_bbox[3] - timestamp_bbox[1]

        # Get font metrics for better vertical centering
        font_metrics = timestamp_font.getmetrics()
        ascent, descent = font_metrics

        # Adjust timestamp background and text positioning
        margin = int(frame_width * 0.02)  # 2% of frame width for margin
        padding_inside_timestamp = int(
            frame_width * 0.015
        )  # 1.5% of frame width for inner padding

        # Calculate background rectangle position
        bg_left = (
            frame_width
            - timestamp_width
            - (margin * 2)
            - (padding_inside_timestamp * 2)
        )
        bg_top = (
            frame_height
            - timestamp_height
            - margin
            - (padding_inside_timestamp * 2)
            - descent
        )
        bg_right = frame_width - margin
        bg_bottom = frame_height - margin

        # Draw semi-transparent background
        overlay = Image.new("RGBA", resized_frame.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom], fill=(0, 0, 0, 180)
        )
        frame_with_timestamp = Image.alpha_composite(
            frame_with_timestamp.convert("RGBA"), overlay
        ).convert("RGB")

        # Add timestamp text
        frame_draw = ImageDraw.Draw(frame_with_timestamp)
        bg_width = bg_right - bg_left
        bg_height = bg_bottom - bg_top
        text_x = bg_left + (bg_width - timestamp_width) // 2
        text_y = bg_top + (bg_height - (ascent + descent)) // 2
        frame_draw.text(
            (text_x, text_y), timestamp_text, font=timestamp_font, fill="white"
        )

        return frame_with_timestamp

    def _add_watermark(self, image: Image.Image, dimensions: dict) -> Image.Image:
        """Add watermark to the image."""
        grid_width = dimensions["grid_width"]
        total_height = dimensions["total_height"]
        watermark_padding = dimensions["watermark_padding"]

        watermark_text = "Thumbr — Video Thumbnail Generator"
        watermark_font_size = int(grid_width * 0.015)  # 1.5% of width for small text

        # Create copy of italic font with desired size
        watermark_font = self.font_italic.font_variant(size=watermark_font_size)

        # Create watermark overlay
        watermark_overlay = Image.new("RGBA", (grid_width, total_height), (0, 0, 0, 0))
        watermark_draw = ImageDraw.Draw(watermark_overlay)

        # Calculate watermark position
        watermark_bbox = watermark_draw.textbbox(
            (0, 0), watermark_text, font=watermark_font
        )
        watermark_width = watermark_bbox[2] - watermark_bbox[0]
        watermark_height = watermark_bbox[3] - watermark_bbox[1]

        watermark_x = (grid_width - watermark_width) // 2  # Center horizontally
        watermark_y = (
            total_height - watermark_height - watermark_padding
        )  # Bottom position

        # Draw watermark with 40% opacity
        watermark_draw.text(
            (watermark_x, watermark_y),
            watermark_text,
            font=watermark_font,
            fill=(0, 0, 0, 153),  # Black with 40% opacity
        )

        # Composite watermark onto final image
        final_image = Image.alpha_composite(
            image.convert("RGBA"), watermark_overlay
        ).convert("RGB")

        return final_image

    def generate_thumbnail(
        self,
        video_path: str,
        output_path: str = None,
        max_width: int = None,
        use_combined_capture: bool = True,
    ) -> str:
        """Generate a thumbnail grid from a video file.

        Args:
            video_path: Path to the video file.
            output_path: Optional output path. If None, will use video filename.
            max_width: Maximum width of the output image. Default from instance.
            use_combined_capture: If True, open video once for info and frames.

        Returns:
            Path to the generated thumbnail.
        """
        max_width = max_width or self.max_width

        # Define the steps for progress tracking
        steps = [
            "Getting video information",
            "Capturing frames",
            "Calculating dimensions",
            "Creating info section",
            "Processing frames with timestamps",
            "Adding watermark",
            "Saving thumbnail",
        ]

        with tqdm(total=len(steps), desc="Generating thumbnail", unit="step") as pbar:
            # Get video info and frames (combined for performance)
            pbar.set_postfix_str(steps[0])
            if use_combined_capture:
                video_info, frames = self.get_video_info_and_frames(video_path)
            else:
                video_info = self.get_video_info(video_path)
                frames = self.capture_frames(video_path)
            pbar.update(2)  # Update by 2 since we did info + frames

            if not frames:
                raise ValueError("No frames could be captured from the video")

            # If no output path specified, generate one in output/ directory
            if output_path is None:
                output_dir = Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)
                video_filename = Path(video_path).stem
                # Replace spaces with underscores and remove any special characters
                safe_filename = "".join(
                    c if c.isalnum() or c == "_" else "_"
                    for c in video_filename.replace(" ", "_")
                )
                output_path = str(output_dir / f"{safe_filename}_thumbnail.jpg")

            pbar.set_postfix_str(steps[2])
            dimensions = self._calculate_dimensions(video_info, max_width)
            pbar.update(1)

            pbar.set_postfix_str(steps[3])
            info_section = self._create_info_section(video_info, dimensions)
            pbar.update(1)

            # Create the final image with white background
            grid_width = dimensions["grid_width"]
            total_height = dimensions["total_height"]
            final_image = Image.new("RGB", (grid_width, total_height), "white")

            # Paste info section at the top
            final_image.paste(info_section, (0, 0))

            # Calculate starting y position for frames
            frames_start_y = dimensions["info_height"]

            # Process frames with timestamps
            total_duration = video_info["duration"]
            frame_interval = total_duration / (self.total_frames + 1)

            pbar.set_postfix_str(steps[4])
            for idx, frame in enumerate(frames):
                # Calculate timestamp for this frame
                timestamp = (idx + 1) * frame_interval

                # Process frame with timestamp
                frame_with_timestamp = self._process_frame_with_timestamp(
                    frame, timestamp, dimensions
                )

                # Calculate position and paste frame
                row = idx // self.grid_size[1]
                col = idx % self.grid_size[1]
                x = dimensions["padding"] + (
                    col * (dimensions["frame_width"] + dimensions["spacing"])
                )
                y = frames_start_y + (
                    row * (dimensions["frame_height"] + dimensions["spacing"])
                )
                final_image.paste(frame_with_timestamp, (x, y))

                # Explicitly delete processed frame to free memory
                del frame_with_timestamp

            # Clear frames list to free memory
            del frames
            pbar.update(1)

            pbar.set_postfix_str(steps[5])
            final_image = self._add_watermark(final_image, dimensions)
            pbar.update(1)

            pbar.set_postfix_str(steps[6])
            # Apply JPEG compression quality only for JPEG files
            output_lower = output_path.lower()
            if output_lower.endswith((".jpg", ".jpeg")):
                final_image.save(output_path, quality=50)
            else:
                final_image.save(output_path)

            # Explicitly clean up
            del final_image
            pbar.update(1)

        self.logger.info(f"Thumbnail saved: {output_path}")
        return output_path

    def generate_thumbnail_safe(
        self,
        video_path: str,
        output_path: str = None,
        max_width: int = None,
        skip_existing: bool = False,
    ) -> ThumbnailResult:
        """Generate thumbnail with error handling and retry logic.

        Args:
            video_path: Path to the video file.
            output_path: Optional output path.
            max_width: Maximum width of the output image.
            skip_existing: If True, skip files where output already exists.

        Returns:
            ThumbnailResult with success status and details.
        """
        # Validate input file
        is_valid, error_msg = validate_video_file(video_path, self.logger)
        if not is_valid:
            return ThumbnailResult(
                video_path=video_path, output_path=None, success=False, error=error_msg
            )

        # Determine output path - default to output/ directory
        if output_path is None:
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_filename = Path(video_path).stem
            safe_filename = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in video_filename.replace(" ", "_")
            )
            output_path = str(output_dir / f"{safe_filename}_thumbnail.jpg")

        # Check for existing output
        if skip_existing and Path(output_path).exists():
            self.logger.info(f"Skipping {video_path} - thumbnail already exists")
            return ThumbnailResult(
                video_path=video_path,
                output_path=output_path,
                success=True,
                skipped=True,
            )

        # Generate with retry logic
        try:
            self._retry_on_failure(
                self.generate_thumbnail, video_path, output_path, max_width
            )
            return ThumbnailResult(
                video_path=video_path, output_path=output_path, success=True, error=None
            )
        except Exception as e:
            self.logger.error(f"Failed to generate thumbnail for {video_path}: {e}")
            return ThumbnailResult(
                video_path=video_path, output_path=None, success=False, error=str(e)
            )


def _process_single_video(args: Tuple) -> ThumbnailResult:
    """Worker function for parallel processing.

    This is a module-level function to allow pickling for ProcessPoolExecutor.

    Args:
        args: Tuple of (video_path, output_path, grid_size, max_width,
                       skip_existing, logger_level, verbose)

    Returns:
        ThumbnailResult for the processed video.
    """
    (
        video_path,
        output_path,
        grid_size,
        max_width,
        skip_existing,
        logger_level,
        verbose,
    ) = args

    # Suppress FFmpeg/OpenCV warnings in workers unless verbose mode
    import os

    if not verbose:
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "fatal"
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    # Create a fresh logger for this process
    import logging

    logger = logging.getLogger("thumbr")
    logger.setLevel(logger_level)

    # Create handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    # Create Thumbr instance
    thumbr = Thumbr(grid_size=grid_size, max_width=max_width, logger=logger)

    # Check for shutdown
    if is_shutdown_requested():
        return ThumbnailResult(
            video_path=video_path,
            output_path=None,
            success=False,
            error="Shutdown requested",
        )

    return thumbr.generate_thumbnail_safe(
        video_path, output_path, max_width, skip_existing
    )


class BatchProcessor:
    """Batch processor for multiple videos with parallel execution."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (3, 3),
        max_width: int = 1920,
        workers: Optional[int] = None,
        skip_existing: bool = False,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        """Initialize batch processor.

        Args:
            grid_size: Grid size for thumbnails.
            max_width: Maximum width of output images.
            workers: Number of parallel workers. If None, uses CPU count.
            skip_existing: Skip videos that already have thumbnails.
            logger: Logger instance.
            verbose: If True, show FFmpeg/OpenCV warnings.
        """
        import multiprocessing
        import logging

        self.grid_size = grid_size
        self.max_width = max_width
        self.skip_existing = skip_existing
        self.logger = logger or get_logger("thumbr")
        self.verbose = verbose

        # Determine worker count (max 5 concurrent items)
        if workers is None:
            self.workers = min(multiprocessing.cpu_count(), 5)
        else:
            self.workers = min(workers, 5)

        self.logger.info(f"Using {self.workers} worker(s) for parallel processing")

    def process_path(
        self,
        path: str,
        recursive: bool = False,
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> List[ThumbnailResult]:
        """Process a single file or directory of videos.

        Args:
            path: Path to video file or directory.
            recursive: If True, search subdirectories.
            formats: Optional list of formats to filter.
            output_dir: Optional output directory. If None, uses same as input.

        Returns:
            List of ThumbnailResult for each processed video.
        """
        # Discover videos
        self.logger.info(f"Discovering videos in: {path}")
        videos = discover_videos(path, recursive, formats)

        if not videos:
            self.logger.warning("No video files found")
            return []

        self.logger.info(f"Found {len(videos)} video(s)")

        # Process each video
        results = []

        if self.workers == 1:
            # Single-threaded processing
            results = self._process_sequential(videos, output_dir)
        else:
            # Multi-threaded processing
            results = self._process_parallel(videos, output_dir, self.verbose)

        return results

    def _process_sequential(
        self, videos: List[str], output_dir: Optional[str]
    ) -> List[ThumbnailResult]:
        """Process videos sequentially."""
        import logging

        thumbr = Thumbr(
            grid_size=self.grid_size, max_width=self.max_width, logger=self.logger
        )

        results = []

        with tqdm(total=len(videos), desc="Processing videos", unit="file") as pbar:
            for video_path in videos:
                if is_shutdown_requested():
                    self.logger.info("Shutdown requested, stopping processing")
                    break

                # Determine output path
                output_path = self._get_output_path(video_path, output_dir)

                result = thumbr.generate_thumbnail_safe(
                    video_path, output_path, self.max_width, self.skip_existing
                )

                results.append(result)

                # Log result
                if result.skipped:
                    pbar.write(f"Skipped: {video_path}")
                elif result.success:
                    pbar.write(f"✓ {video_path}")
                else:
                    pbar.write(f"✗ {video_path}: {result.error}")

                pbar.update(1)

        return results

    def _process_parallel(
        self, videos: List[str], output_dir: Optional[str], verbose: bool = False
    ) -> List[ThumbnailResult]:
        """Process videos in parallel."""
        import logging
        import multiprocessing

        # Get logger level for passing to workers
        logger_level = self.logger.level

        # Prepare arguments for workers
        work_items = []
        for video_path in videos:
            output_path = self._get_output_path(video_path, output_dir)
            work_items.append(
                (
                    video_path,
                    output_path,
                    self.grid_size,
                    self.max_width,
                    self.skip_existing,
                    logger_level,
                    verbose,
                )
            )

        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(_process_single_video, item): item[0]
                for item in work_items
            }

            with tqdm(
                total=len(work_items), desc="Processing videos", unit="file"
            ) as pbar:
                for future in as_completed(future_to_video):
                    if is_shutdown_requested():
                        self.logger.info(
                            "Shutdown requested, canceling remaining tasks"
                        )
                        # Cancel pending futures
                        for f in future_to_video:
                            f.cancel()
                        break

                    video_path = future_to_video[future]

                    try:
                        result = future.result()
                        results.append(result)

                        # Log result
                        if result.skipped:
                            pbar.write(f"Skipped: {video_path}")
                        elif result.success:
                            pbar.write(f"✓ {video_path}")
                        else:
                            pbar.write(f"✗ {video_path}: {result.error}")

                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing {video_path}: {e}"
                        )
                        results.append(
                            ThumbnailResult(
                                video_path=video_path,
                                output_path=None,
                                success=False,
                                error=str(e),
                            )
                        )

                    pbar.update(1)

        return results

    def _get_output_path(
        self, video_path: str, output_dir: Optional[str]
    ) -> Optional[str]:
        """Determine output path for a video."""
        if output_dir is None:
            # Default to output/ directory
            output_dir = "output"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        video_filename = Path(video_path).stem
        safe_filename = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in video_filename.replace(" ", "_")
        )

        return str(Path(output_dir) / f"{safe_filename}_thumbnail.jpg")


def process_batch(
    path: str,
    grid_size: Tuple[int, int] = (3, 3),
    max_width: int = 1920,
    workers: Optional[int] = None,
    recursive: bool = False,
    formats: Optional[List[str]] = None,
    skip_existing: bool = False,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convenience function to process a batch of videos.

    Args:
        path: Path to video file or directory.
        grid_size: Grid size for thumbnails.
        max_width: Maximum width of output images.
        workers: Number of parallel workers.
        recursive: Search subdirectories.
        formats: Video formats to filter.
        skip_existing: Skip existing thumbnails.
        output_dir: Output directory.
        logger: Logger instance.
        verbose: If True, show FFmpeg/OpenCV warnings.

    Returns:
        Dictionary with processing statistics.
    """
    logger = logger or get_logger("thumbr")

    processor = BatchProcessor(
        grid_size=grid_size,
        max_width=max_width,
        workers=workers,
        skip_existing=skip_existing,
        logger=logger,
        verbose=verbose,
    )

    results = processor.process_path(
        path, recursive=recursive, formats=formats, output_dir=output_dir
    )

    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.success)

    return {
        "total": total,
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }
