import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
from src.utils import (
    load_font,
    load_italic_font,
    generate_info_text,
)


class Thumbr:
    def __init__(self, grid_size: Tuple[int, int] = (3, 3)):
        """Initialize the thumbnailer with grid configuration.

        Args:
            grid_size: Tuple of (rows, columns) for the thumbnail grid.
                       Default is (3, 3).
        """
        self.grid_size = grid_size
        self.total_frames = grid_size[0] * grid_size[1]

        # Initialize fonts with cross-platform fallback
        self.font = load_font()
        self.font_italic = load_italic_font()

    def get_video_info(self, video_path: str) -> dict:
        """Get video information including resolution, duration, and file size.

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
        self, video_path: str, output_path: str = None, max_width: int = 1920
    ) -> str:
        """Generate a thumbnail grid from a video file.

        Args:
            video_path: Path to the video file.
            output_path: Optional output path. If None, will use video filename.
            max_width: Maximum width of the output image. Default 1920.

        Returns:
            Path to the generated thumbnail.
        """
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
            pbar.set_postfix_str(steps[0])
            video_info = self.get_video_info(video_path)
            pbar.update(1)

            pbar.set_postfix_str(steps[1])
            frames = self.capture_frames(video_path)
            pbar.update(1)

            if not frames:
                raise ValueError("No frames could be captured from the video")

            # If no output path specified, generate one from video filename
            if output_path is None:
                video_filename = Path(video_path).stem
                # Replace spaces with underscores and remove any special characters
                safe_filename = "".join(
                    c if c.isalnum() or c == "_" else "_"
                    for c in video_filename.replace(" ", "_")
                )
                output_path = f"{safe_filename}_thumbnail.jpg"

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
            pbar.update(1)

            pbar.set_postfix_str(steps[5])
            final_image = self._add_watermark(final_image, dimensions)
            pbar.update(1)

            pbar.set_postfix_str(steps[6])
            final_image.save(output_path, quality=95)
            pbar.update(1)

        print(f"Thumbnail saved: {output_path}")
        return output_path
