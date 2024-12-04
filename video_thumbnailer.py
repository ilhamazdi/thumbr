import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import math

class Thumbr:
    def __init__(self, grid_size: Tuple[int, int] = (3, 3)):
        """Initialize the thumbnailer with grid configuration.
        
        Args:
            grid_size: Tuple of (rows, columns) for the thumbnail grid. Default is (3, 3).
        """
        self.grid_size = grid_size
        self.total_frames = grid_size[0] * grid_size[1]
        
        # Initialize fonts - regular and italic
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
            self.font_italic = ImageFont.truetype("ariali.ttf", 16)  # Arial Italic
        except:
            self.font = ImageFont.load_default()
            self.font_italic = self.font
    
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
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'file_size': file_size,
            'filename': os.path.basename(video_path)
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
        header_height = 60
        header = Image.new('RGB', (width, header_height), color='black')
        draw = ImageDraw.Draw(header)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        info_text = (
            f"File: {video_info['filename']} | "
            f"Size: {video_info['file_size'] / (1024*1024):.1f}MB | "
            f"Resolution: {video_info['width']}x{video_info['height']} | "
            f"Duration: {video_info['duration']:.1f}s"
        )
        
        text_bbox = draw.textbbox((0, 0), info_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (header_height - text_height) // 2
        draw.text((x, y), info_text, font=font, fill='white')
        
        return header
    
    def generate_thumbnail(self, video_path: str, output_path: str = None, max_width: int = 1920) -> str:
        """Generate a thumbnail grid from a video file.
        
        Args:
            video_path: Path to the video file.
            output_path: Optional output path. If None, will use video filename.
            max_width: Maximum width of the output image. Default 1920.
            
        Returns:
            Path to the generated thumbnail.
        """
        video_info = self.get_video_info(video_path)
        frames = self.capture_frames(video_path)
        
        if not frames:
            raise ValueError("No frames could be captured from the video")
            
        # If no output path specified, generate one from video filename
        if output_path is None:
            video_filename = Path(video_path).stem
            # Replace spaces with underscores and remove any special characters
            safe_filename = "".join(c if c.isalnum() or c == '_' else '_' 
                                  for c in video_filename.replace(' ', '_'))
            output_path = f"{safe_filename}_thumbnail.jpg"

        # Add padding and spacing
        padding = 30  # padding around the entire image
        spacing = 20  # spacing between frames
        
        # Calculate thumbnail dimensions while maintaining aspect ratio
        available_width = max_width - (2 * padding) - (spacing * (self.grid_size[1] - 1))
        frame_width = available_width // self.grid_size[1]
        aspect_ratio = video_info['width'] / video_info['height']
        frame_height = int(frame_width / aspect_ratio)
        
        # Calculate total dimensions including padding and spacing
        grid_width = (frame_width * self.grid_size[1]) + (spacing * (self.grid_size[1] - 1)) + (2 * padding)
        grid_height = (frame_height * self.grid_size[0]) + (spacing * (self.grid_size[0] - 1))
        
        # Calculate info section height based on content
        info_font_size = int(grid_width * 0.015)  # 1.5% of width for media info
        line_spacing = info_font_size + 5
        num_info_lines = 5  # Number of info text lines
        info_height = (line_spacing * num_info_lines) + (padding * 2)  # Add padding top and bottom
        
        total_height = grid_height + info_height + (2 * padding)
        
        # Add extra height for watermark
        watermark_padding = padding * 2  # Double padding for watermark section
        total_height = total_height + watermark_padding
        
        # Create the final image with white background
        final_image = Image.new('RGB', (grid_width, total_height), 'white')
        draw = ImageDraw.Draw(final_image)
        
        # Calculate font sizes
        info_font_size = int(grid_width * 0.015)  # 1.5% of width for media info
        timestamp_font_size = int(frame_width * 0.06)  # Increased from 0.04 to 0.06
        
        try:
            info_font = ImageFont.truetype("arial.ttf", info_font_size)
            timestamp_font = ImageFont.truetype("arial.ttf", timestamp_font_size)
        except:
            info_font = ImageFont.load_default()
            timestamp_font = ImageFont.load_default()
        
        # Reset y_position to properly space info text
        y_position = padding
        
        # Format duration to show appropriate units
        def format_duration(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = int(seconds % 60)
            
            if hours > 0:
                return f"Duration: {hours}h {minutes}m {remaining_seconds}s"
            elif minutes > 0:
                return f"Duration: {minutes}m {remaining_seconds}s"
            else:
                return f"Duration: {remaining_seconds}s"
        
        info_text = [
            f"Filename: {video_info['filename']}",
            format_duration(video_info['duration']),
            f"Resolution: {video_info['width']}x{video_info['height']}",
            f"Frame Rate: {video_info['fps']:.1f} fps",
            f"File Size: {video_info['file_size'] / (1024*1024):.1f} MB"
        ]
        
        for text in info_text:
            draw.text((padding, y_position), text, font=info_font, fill='black')
            y_position += line_spacing
        
        # Calculate starting y position for frames right after the info section
        frames_start_y = info_height
        
        # Resize frames and add timestamps
        total_duration = video_info['duration']
        frame_interval = total_duration / (self.total_frames + 1)
        
        for idx, frame in enumerate(frames):
            # Convert frame to PIL Image
            pil_frame = Image.fromarray(frame)
            resized_frame = pil_frame.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # Calculate timestamp for this frame
            timestamp = (idx + 1) * frame_interval
            hours = int(timestamp // 3600)
            minutes = int((timestamp % 3600) // 60)
            seconds = int(timestamp % 60)
            timestamp_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create a new image for the frame with timestamp
            frame_with_timestamp = resized_frame.copy()
            frame_draw = ImageDraw.Draw(frame_with_timestamp)
            
            # Calculate timestamp text dimensions
            timestamp_bbox = frame_draw.textbbox((0, 0), timestamp_text, font=timestamp_font)
            timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
            timestamp_height = timestamp_bbox[3] - timestamp_bbox[1]
            
            # Get font metrics for better vertical centering
            font_metrics = timestamp_font.getmetrics()
            ascent, descent = font_metrics
            
            # Adjust timestamp background and text positioning
            margin = int(frame_width * 0.02)  # 2% of frame width for margin
            padding_inside_timestamp = int(frame_width * 0.015)  # 1.5% of frame width for inner padding
            
            # Calculate background rectangle position with adjusted vertical spacing
            bg_left = frame_width - timestamp_width - (margin * 2) - (padding_inside_timestamp * 2)
            bg_top = frame_height - timestamp_height - margin - (padding_inside_timestamp * 2) - descent
            bg_right = frame_width - margin
            bg_bottom = frame_height - margin
            
            # Calculate background dimensions
            bg_width = bg_right - bg_left
            bg_height = bg_bottom - bg_top
            
            # Draw semi-transparent background
            overlay = Image.new('RGBA', resized_frame.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [bg_left, bg_top, bg_right, bg_bottom],
                fill=(0, 0, 0, 180)
            )
            frame_with_timestamp = Image.alpha_composite(
                frame_with_timestamp.convert('RGBA'),
                overlay
            ).convert('RGB')
            
            # Add timestamp text with centered positioning
            frame_draw = ImageDraw.Draw(frame_with_timestamp)
            
            # Calculate text position to center it within the background
            text_x = bg_left + (bg_width - timestamp_width) // 2
            # Adjust vertical position considering font metrics
            text_y = bg_top + (bg_height - (ascent + descent)) // 2
            
            frame_draw.text(
                (text_x, text_y),
                timestamp_text,
                font=timestamp_font,
                fill='white'
            )
            
            # Calculate position and paste frame
            row = idx // self.grid_size[1]
            col = idx % self.grid_size[1]
            x = padding + (col * (frame_width + spacing))
            y = frames_start_y + (row * (frame_height + spacing))
            final_image.paste(frame_with_timestamp, (x, y))
        
        # Calculate info section layout
        info_section_width = grid_width * 0.7  # Use 70% of width for info text
        watermark_section_width = grid_width - info_section_width - (padding * 2)

        # Add footer watermark text
        watermark_text = "Generated using : Thumbr — Video Thumbnail Generator"
        watermark_font_size = int(grid_width * 0.015)  # 1.5% of width for small text
        
        # Create copy of italic font with desired size
        watermark_font = self.font_italic.font_variant(size=watermark_font_size)

        # Create watermark overlay
        watermark_overlay = Image.new('RGBA', (grid_width, total_height), (0, 0, 0, 0))
        watermark_draw = ImageDraw.Draw(watermark_overlay)

        # Calculate watermark position (centered at bottom)
        watermark_bbox = watermark_draw.textbbox((0, 0), watermark_text, font=watermark_font)
        watermark_width = watermark_bbox[2] - watermark_bbox[0]
        watermark_height = watermark_bbox[3] - watermark_bbox[1]
        
        # Position at bottom with extra padding
        watermark_x = (grid_width - watermark_width) // 2  # Center horizontally
        watermark_y = total_height - watermark_height - watermark_padding  # Bottom position

        # Draw watermark with 40% opacity
        watermark_draw.text(
            (watermark_x, watermark_y),
            watermark_text,
            font=watermark_font,
            fill=(0, 0, 0, 102),  # Black with 40% opacity (102 is 40% of 255)
        )

        # Composite watermark onto final image
        final_image = Image.alpha_composite(
            final_image.convert('RGBA'),
            watermark_overlay
        ).convert('RGB')

        # Save the thumbnail
        final_image.save(output_path, quality=95)
        return output_path
    
    def create_contact_sheet(
        self, screenshots: List[Image.Image], media_info: Dict[str, Any]
    ) -> Image.Image:
        # Add spacing between screenshots
        spacing = 20  # pixels between screenshots
        
        # Calculate new dimensions with spacing
        total_width = (self.thumbnail_width * self.grid_size[1]) + (spacing * (self.grid_size[1] - 1))
        total_height = (self.thumbnail_height * self.grid_size[0]) + (spacing * (self.grid_size[0] - 1))
        
        # Add extra height for media info
        info_height = int(total_height * 0.15)  # 15% of total height for info
        contact_sheet = Image.new('RGB', (total_width, total_height + info_height), 'white')
        
        # Paste screenshots with spacing
        for idx, screenshot in enumerate(screenshots):
            row = idx // self.grid_size[1]
            col = idx % self.grid_size[1]
            
            x = col * (self.thumbnail_width + spacing)
            y = row * (self.thumbnail_height + spacing)
            
            contact_sheet.paste(screenshot, (x, y))

        # Add media info with improved formatting
        draw = ImageDraw.Draw(contact_sheet)
        
        # Calculate font sizes based on image dimensions
        title_font_size = int(total_width * 0.02)  # 2% of width
        info_font_size = int(total_width * 0.015)  # 1.5% of width
        
        title_font = ImageFont.truetype("Arial.ttf", title_font_size)
        info_font = ImageFont.truetype("Arial.ttf", info_font_size)
        
        # Position text at the bottom with proper spacing
        y_position = total_height + 0  # Start 20 pixels below the screenshots
        line_spacing = info_font_size + 10
        
        # Add media info with proper formatting
        info_text = [
            f"Filename: {media_info.get('filename', 'Unknown')}",
            f"Duration: {media_info.get('duration', 'Unknown')}",
            f"Resolution: {media_info.get('width', '?')}x{media_info.get('height', '?')}",
            f"Codec: {media_info.get('codec_name', 'Unknown')}",
            f"Bitrate: {media_info.get('bit_rate', 'Unknown')}"
        ]
        
        for text in info_text:
            draw.text((20, y_position), text, font=info_font, fill='black')
            y_position += line_spacing

        return contact_sheet 