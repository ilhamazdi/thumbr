import logging
import os
from pathlib import Path
from typing import List, Optional, Set
from PIL import ImageFont

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# All video formats supported by FFmpeg (comprehensive list)
# Container formats and raw video codecs
# Sorted alphabetically: 0-9, a-z
VIDEO_EXTENSIONS: Set[str] = {
    ".3gp",
    ".3g2",
    ".amv",
    ".avi",
    ".asf",
    ".av1",
    ".avc",
    ".bik",
    ".bk2",
    ".divx",
    ".dv",
    ".dif",
    ".dvr-ms",
    ".f4v",
    ".flv",
    ".gxf",
    ".h264",
    ".h265",
    ".hevc",
    ".m1v",
    ".m2t",
    ".m2ts",
    ".m2v",
    ".m4a",
    ".m4p",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp2",
    ".mp4",
    ".mp4v",
    ".mpc",
    ".mpeg",
    ".mpg",
    ".mpe",
    ".mpv",
    ".mts",
    ".mxf",
    ".nsv",
    ".nut",
    ".ogm",
    ".ogg",
    ".ogv",
    ".qt",
    ".ram",
    ".raw",
    ".rm",
    ".rmvb",
    ".roq",
    ".sdp",
    ".smk",
    ".swf",
    ".ts",
    ".vivo",
    ".vob",
    ".webm",
    ".wtv",
    ".wmv",
    ".xmv",
    ".yuv",
}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return a logger for the application.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("thumbr")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Create console handler with appropriate format
    handler = logging.StreamHandler()
    if verbose:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(handler)
    return logger


def get_logger(name: str = "thumbr") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def is_video_file(file_path: str, probe_with_opencv: bool = False) -> bool:
    """Check if a file is a supported video format.

    This function performs a fast extension check first. If the extension is not
    in the known VIDEO_EXTENSIONS set and probe_with_opencv is True, it will
    attempt to open the file with OpenCV to determine if it's a valid video.

    Args:
        file_path: Path to the file.
        probe_with_opencv: If True and extension is not in VIDEO_EXTENSIONS,
                          try to open with OpenCV to verify it's a valid video.
                          This is slower but allows for ANY video format supported
                          by FFmpeg/OpenCV.

    Returns:
        True if the file is a supported video format.
    """
    ext = Path(file_path).suffix.lower()

    # Fast path: check if extension is in known video extensions
    if ext in VIDEO_EXTENSIONS:
        return True

    # If extension not found and probing is enabled, try OpenCV
    if probe_with_opencv:
        return _probe_video_with_opencv(file_path)

    return False


def _probe_video_with_opencv(file_path: str) -> bool:
    """Probe if a file is a valid video using OpenCV.

    This uses OpenCV's video capture to determine if the file
    is a valid video that FFmpeg can read.

    Args:
        file_path: Path to the file to probe.

    Returns:
        True if OpenCV can successfully open the file as video.
    """
    if not HAS_CV2:
        return False

    try:
        # Try to open the file with OpenCV
        cap = cv2.VideoCapture(file_path)
        is_opened = cap.isOpened()
        cap.release()
        return is_opened
    except Exception:
        return False


def discover_videos(
    path: str, recursive: bool = False, formats: Optional[List[str]] = None
) -> List[str]:
    """Discover video files in a directory.

    Args:
        path: Path to directory or file.
        recursive: If True, search subdirectories.
        formats: Optional list of specific formats to filter (e.g., ['.mp4', '.avi']).
                 If None, uses all supported formats.

    Returns:
        List of video file paths.
    """
    # Normalize formats to include dot prefix
    target_formats = set()
    if formats:
        for f in formats:
            if f.startswith("."):
                target_formats.add(f.lower())
            else:
                target_formats.add(f".{f.lower()}")
    else:
        target_formats = VIDEO_EXTENSIONS.copy()
    videos = []

    path_obj = Path(path)

    if path_obj.is_file():
        # Single file - validate it's a video
        if is_video_file(str(path_obj)):
            videos.append(str(path_obj.absolute()))
    elif path_obj.is_dir():
        # Directory - discover videos
        if recursive:
            # Recursive search
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in target_formats:
                        videos.append(str(file_path.absolute()))
        else:
            # Top-level only
            for file_path in path_obj.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in target_formats:
                    videos.append(str(file_path.absolute()))

    return sorted(videos)


def validate_video_file(
    file_path: str, logger: Optional[logging.Logger] = None
) -> tuple[bool, str]:
    """Validate that a file exists, is readable, and is a valid video.

    Args:
        file_path: Path to the video file.
        logger: Optional logger for warnings.

    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        msg = f"File does not exist: {file_path}"
        if logger:
            logger.warning(msg)
        return False, msg

    # Check if it's a file (not directory)
    if not path.is_file():
        msg = f"Path is not a file: {file_path}"
        if logger:
            logger.warning(msg)
        return False, msg

    # Check file extension
    if not is_video_file(str(path)):
        msg = f"Unsupported video format: {path.suffix}"
        if logger:
            logger.warning(msg)
        return False, msg

    # Check if readable
    if not os.access(file_path, os.R_OK):
        msg = f"File is not readable: {file_path}"
        if logger:
            logger.warning(msg)
        return False, msg

    # Check file size (not empty)
    file_size = path.stat().st_size
    if file_size == 0:
        msg = f"File is empty: {file_path}"
        if logger:
            logger.warning(msg)
        return False, msg

    return True, ""


def load_font() -> ImageFont.FreeTypeFont:
    """Load Arial font with fallback to system sans-serif fonts."""
    font_names = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, 16)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def load_italic_font() -> ImageFont.FreeTypeFont:
    """Load Arial Italic font with fallback to system sans-serif fonts."""
    font_names = [
        "ariali.ttf",
        "Arial-Italic.ttf",
        "DejaVuSans-Oblique.ttf",
        "LiberationSans-Italic.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, 16)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format (e.g., 1h 2m 1s)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def generate_info_text(
    media_info: dict,
    include_frame_rate: bool = True,
    include_codec: bool = False,
) -> List[str]:
    """Generate standardized info text for media.

    Args:
        media_info: Dictionary containing media information.
        include_frame_rate: Whether to include frame rate info (for video).
        include_codec: Whether to include codec info (for contact sheet).

    Returns:
        List of formatted info strings.
    """
    info_lines = []

    # Always include filename
    info_lines.append(f"Filename: {media_info.get('filename', 'Unknown')}")

    # Format duration if available
    if "duration" in media_info:
        if isinstance(media_info["duration"], str):
            info_lines.append(f"Duration: {media_info['duration']}")
        else:
            duration = format_duration(media_info["duration"])
            info_lines.append(f"Duration: {duration}")

    # Include resolution if available
    if "width" in media_info and "height" in media_info:
        info_lines.append(f"Resolution: {media_info['width']}x{media_info['height']}")

    # Include frame rate for video thumbnails
    if include_frame_rate and "fps" in media_info:
        info_lines.append(f"Frame Rate: {media_info['fps']:.1f} fps")

    # Include file size if available
    if "file_size" in media_info:
        file_size = format_file_size(media_info["file_size"])
        info_lines.append(f"File Size: {file_size}")

    # Include codec for contact sheets
    if include_codec and "codec_name" in media_info:
        info_lines.append(f"Codec: {media_info['codec_name']}")

    # Include bitrate for contact sheets
    if include_codec and "bit_rate" in media_info:
        info_lines.append(f"Bitrate: {media_info['bit_rate']}")

    return info_lines
