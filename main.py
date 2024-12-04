from pathlib import Path
from video_thumbnailer import Thumbr

def main():
    # Create a thumbnailer with default 3x3 grid
    thumbnailer = Thumbr(grid_size=(3, 4))

    # Define input path
    video_path = Path(r"path/to/your/video.mp4")
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output path in the output directory
    output_path = output_dir / f"{video_path.stem}_thumbnail.jpg"

    try:
        # Generate thumbnail with specified output path
        result_path = thumbnailer.generate_thumbnail(str(video_path), str(output_path))
        print(f"Thumbnail generated successfully: {result_path}")
    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")

if __name__ == "__main__":
    main()