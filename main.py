from src.thumbr import Thumbr
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate a thumbnail from a video.")
    parser.add_argument("input", help="Input video file path.")
    parser.add_argument("-o", "--output", help="Output thumbnail file path.")
    parser.add_argument(
        "-g",
        "--grid",
        default="3x3",
        help="Grid size in ROWSxCOLS format (e.g., 3x3).",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    output_path = args.output
    if not output_path:
        p = Path(input_path)
        output_path = str(p.with_name(f"{p.stem}_thumbr.jpg"))

    try:
        grid_rows, grid_cols = map(int, args.grid.split("x"))
    except ValueError:
        print("Error: Invalid grid format. Please use ROWSxCOLS (e.g., 4x4).")
        return

    thumbr = Thumbr(grid_size=(grid_rows, grid_cols))
    try:
        print(f"Generating thumbnail for {input_path}...")
        thumbr.generate_thumbnail(input_path, output_path)
        print(f"Thumbnail generated successfully: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
