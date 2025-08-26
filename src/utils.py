from typing import List
from PIL import ImageFont


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
