#!/usr/bin/env python3
"""
Thumbr - Video Thumbnail Generator

Generate thumbnail grids from video files. Supports single file processing
or batch processing of multiple videos with parallel execution.
"""

from src.thumbr import Thumbr, process_batch, BatchProcessor
from src.utils import setup_logging, get_logger, discover_videos, validate_video_file
import os
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate thumbnail grids from video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video file
  %(prog)s video.mp4

  # Process a single video with custom output
  %(prog)s video.mp4 -o output/thumbnail.jpg

  # Process all videos in a directory (non-recursive)
  %(prog)s /path/to/videos/

  # Process all videos recursively with 4 workers
  %(prog)s /path/to/videos/ --recursive --workers 4

  # Only process specific formats
  %(prog)s /path/to/videos/ --format mp4 avi

  # Skip existing thumbnails
  %(prog)s /path/to/videos/ --skip-existing

  # Custom grid size
  %(prog)s video.mp4 -g 4x4
""",
    )

    # Positional argument - now accepts file or directory
    parser.add_argument(
        "input", help="Input video file or directory containing video files."
    )

    # Output path
    parser.add_argument(
        "-o",
        "--output",
        help="Output thumbnail file path (for single file) or output directory (for batch).",
    )

    # Grid size
    parser.add_argument(
        "-g",
        "--grid",
        default="3x3",
        help="Grid size in ROWSxCOLS format (e.g., 3x3, 4x4). Default: 3x3",
    )

    # Max width
    parser.add_argument(
        "-w",
        "--max-width",
        type=int,
        default=1920,
        help="Maximum width of output thumbnail. Default: 1920",
    )

    # Recursive search
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories for video files when input is a directory.",
    )

    # Workers
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: auto (CPU count).",
    )

    # Format filter
    parser.add_argument(
        "--format",
        nargs="+",
        default=None,
        help="Filter by video format(s): mp4 avi mkv mov wmv flv webm m4v mpeg mpg",
    )

    # Skip existing
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have a thumbnail.",
    )

    # Verbose
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (debug) logging."
    )

    args = parser.parse_args()

    # Suppress FFmpeg/OpenCV warnings unless verbose mode
    import os

    if not args.verbose:
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = (
            "fatal"  # Suppress warnings, only show fatal errors
        )
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Also suppress OpenCV logs

    # Set up logging
    logger = setup_logging(verbose=args.verbose)

    # Validate input path
    input_path = args.input
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        return 1

    # Parse grid size
    try:
        grid_rows, grid_cols = map(int, args.grid.split("x"))
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError()
    except ValueError:
        logger.error("Invalid grid format. Please use ROWSxCOLS (e.g., 3x3, 4x4).")
        return 1

    # Determine if input is a file or directory
    is_directory = os.path.isdir(input_path)
    is_file = os.path.isfile(input_path)

    if is_directory:
        # Batch processing mode
        return run_batch_processing(
            input_path=input_path,
            output_dir=args.output,
            grid_size=(grid_rows, grid_cols),
            max_width=args.max_width,
            workers=args.workers,
            recursive=args.recursive,
            formats=args.format,
            skip_existing=args.skip_existing,
            logger=logger,
            verbose=args.verbose,
        )
    elif is_file:
        # Single file mode (backward compatible)
        return run_single_file(
            input_path=input_path,
            output_path=args.output,
            grid_size=(grid_rows, grid_cols),
            max_width=args.max_width,
            logger=logger,
            verbose=args.verbose,
        )
    else:
        logger.error(f"Input path is neither a file nor directory: {input_path}")
        return 1


def run_single_file(
    input_path: str,
    output_path: str,
    grid_size: tuple,
    max_width: int,
    logger,
    verbose: bool = False,
) -> int:
    """Process a single video file."""
    # Validate file
    is_valid, error_msg = validate_video_file(input_path, logger)
    if not is_valid:
        logger.error(error_msg)
        return 1

    # Determine output path - default to output/ directory
    if not output_path:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        p = Path(input_path)
        output_path = str(output_dir / f"{p.stem}_thumbnail.jpg")

    # Generate thumbnail
    logger.info(f"Generating thumbnail for: {input_path}")

    thumbr = Thumbr(grid_size=grid_size, max_width=max_width, logger=logger)

    try:
        result = thumbr.generate_thumbnail_safe(input_path, output_path, max_width)

        if result.success:
            logger.info(f"Thumbnail generated successfully: {result.output_path}")
            return 0
        else:
            logger.error(f"Failed to generate thumbnail: {result.error}")
            return 1

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def run_batch_processing(
    input_path: str,
    output_dir: str,
    grid_size: tuple,
    max_width: int,
    workers: int,
    recursive: bool,
    formats: list,
    skip_existing: bool,
    logger,
    verbose: bool = False,
) -> int:
    """Process multiple videos in batch mode."""
    # Validate output directory
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            return 1

    # Discover videos
    logger.info(f"Searching for videos in: {input_path}")
    if recursive:
        logger.info("Recursive search enabled")

    videos = discover_videos(input_path, recursive, formats)

    if not videos:
        logger.warning("No video files found")
        return 0

    logger.info(f"Found {len(videos)} video(s) to process")

    if workers:
        logger.info(f"Using {workers} worker(s)")

    try:
        # Process batch
        stats = process_batch(
            path=input_path,
            grid_size=grid_size,
            max_width=max_width,
            workers=workers,
            recursive=recursive,
            formats=formats,
            skip_existing=skip_existing,
            output_dir=output_dir,
            logger=logger,
            verbose=verbose,
        )

        # Print final status (summary already shown by BatchProgressDisplay)
        if stats["failed"] > 0:
            logger.warning(f"{stats['failed']} video(s) failed processing")
            return 1
        else:
            logger.info("All videos processed successfully")
            return 0

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
