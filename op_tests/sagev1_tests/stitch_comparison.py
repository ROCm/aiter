#!/usr/bin/env python3
"""
Stitch multiple CogVideoX attention comparison videos into a grid layout.

This script combines videos rendered with different attention types into a single
comparison video with text annotations showing which attention type was used.
"""

import argparse
import glob
import logging
import math
import os
import re
import sys

# Check for moviepy installation before importing
# MoviePy 2.x changed import structure: use `from moviepy import ...` instead of `from moviepy.editor import ...`
try:
    # Try MoviePy 2.x imports first
    from moviepy import (
        VideoFileClip,
        TextClip,
        CompositeVideoClip,
        clips_array,
        ColorClip,
    )
except ImportError:
    try:
        # Fall back to MoviePy 1.x imports
        from moviepy.editor import (
            VideoFileClip,
            TextClip,
            CompositeVideoClip,
            clips_array,
            ColorClip,
        )
    except ImportError as e:
        print("\n" + "=" * 60)
        print("ERROR: MoviePy is not installed!")
        print("=" * 60)
        print("\nTo install MoviePy, run one of the following:")
        print("\n  pip install moviepy")
        print("  # or")
        print("  pip install moviepy[optional]  # for additional codecs")
        print("\nMoviePy also requires ImageMagick for text rendering.")
        print("Install ImageMagick:")
        print("\n  Ubuntu/Debian: sudo apt-get install imagemagick")
        print("  macOS:         brew install imagemagick")
        print("  Fedora/RHEL:   sudo dnf install ImageMagick")
        print("\nAfter installing ImageMagick, you may need to edit the policy file")
        print("to allow MoviePy to use it. See MoviePy documentation for details.")
        print("=" * 60 + "\n")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_attention_type(filename: str) -> str:
    """
    Extract kernel name from filename.
    
    Supports multiple naming formats:
    - WAN format: wan_result_{task}_{options}_attn_{kernel_name}_{resolution}.mp4
      Example: wan_result_i2v_ulysses8_ringNone_compileTrue_attn_fav3_fp8_720x1280.mp4 -> fav3_fp8
    - CogVideoX format: cogvideox-{model}_{attention_type}.mp4
      Example: cogvideox-2b_sagev1.mp4 -> sagev1
    """
    basename = os.path.basename(filename)
    
    # Try WAN format: extract kernel name between _attn_ and _<resolution>
    # Resolution pattern: _\d+x\d+ (e.g., _720x1280)
    wan_match = re.search(r'_attn_(.+?)_\d+x\d+\.mp4$', basename)
    if wan_match:
        return wan_match.group(1)
    
    # Try CogVideoX format: cogvideox-{model}_{attention_type}.mp4
    cogvideo_match = re.search(r'cogvideox-[^_]+_(.+)\.mp4$', basename)
    if cogvideo_match:
        return cogvideo_match.group(1)
    
    # Fallback: use filename without extension
    return os.path.splitext(basename)[0]


def create_text_clip(label: str, font_size: int, duration: float, styled: bool = True):
    """
    Create a TextClip with MoviePy version compatibility.
    MoviePy 2.x changed the API: fontsize -> font_size, positional text -> keyword text
    """
    import moviepy
    major_version = int(moviepy.__version__.split('.')[0])
    
    if major_version >= 2:
        # MoviePy 2.x API
        if styled:
            return TextClip(
                text=label,
                font_size=font_size,
                color='white',
                stroke_color='black',
                stroke_width=2,
                duration=duration,
            )
        else:
            return TextClip(
                text=label,
                font_size=font_size,
                color='white',
                duration=duration,
            )
    else:
        # MoviePy 1.x API
        if styled:
            return TextClip(
                label,
                fontsize=font_size,
                color='white',
                font='DejaVu-Sans-Bold',
                stroke_color='black',
                stroke_width=2,
            ).set_duration(duration)
        else:
            return TextClip(
                label,
                fontsize=font_size,
                color='white',
            ).set_duration(duration)


def create_labeled_clip(video_path: str, label: str, font_size: int = 36) -> CompositeVideoClip:
    """
    Create a video clip with a text label overlay.
    
    Args:
        video_path: Path to the video file
        label: Text label to overlay
        font_size: Size of the label font
        
    Returns:
        CompositeVideoClip with the label overlaid
    """
    video = VideoFileClip(video_path)
    
    # Create text label with background for readability
    try:
        # Try to create styled TextClip
        txt_clip = create_text_clip(label, font_size, video.duration, styled=True)
    except Exception as e:
        logger.warning(f"Could not create styled text: {e}")
        logger.warning("Falling back to simple text...")
        try:
            txt_clip = create_text_clip(label, font_size, video.duration, styled=False)
        except Exception as e2:
            logger.error(f"Failed to create text clip: {e2}")
            logger.error("Make sure ImageMagick is installed and configured for MoviePy")
            raise
    
    # Create semi-transparent background for the label
    txt_w, txt_h = txt_clip.size
    padding = 10
    
    # MoviePy 2.x uses with_* methods, 1.x uses set_* methods
    import moviepy
    major_version = int(moviepy.__version__.split('.')[0])
    
    bg_clip = ColorClip(
        size=(txt_w + padding * 2, txt_h + padding * 2),
        color=(0, 0, 0)
    )
    
    # Position label at top-left with some margin
    margin = 15
    
    if major_version >= 2:
        # MoviePy 2.x API
        bg_clip = bg_clip.with_opacity(0.6).with_duration(video.duration)
        bg_clip = bg_clip.with_position((margin, margin))
        txt_clip = txt_clip.with_position((margin + padding, margin + padding))
    else:
        # MoviePy 1.x API
        bg_clip = bg_clip.set_opacity(0.6).set_duration(video.duration)
        bg_clip = bg_clip.set_position((margin, margin))
        txt_clip = txt_clip.set_position((margin + padding, margin + padding))
    
    # Composite the video with background and text
    return CompositeVideoClip([video, bg_clip, txt_clip])


def calculate_grid_dimensions(n_videos: int) -> tuple[int, int]:
    """
    Calculate optimal grid dimensions for n videos.
    Prefers wider layouts (more columns than rows).
    
    Args:
        n_videos: Number of videos to arrange
        
    Returns:
        Tuple of (rows, cols)
    """
    if n_videos <= 0:
        raise ValueError("Need at least 1 video")
    if n_videos == 1:
        return (1, 1)
    if n_videos == 2:
        return (1, 2)
    if n_videos == 3:
        return (1, 3)
    if n_videos == 4:
        return (2, 2)
    if n_videos <= 6:
        return (2, 3)
    if n_videos <= 9:
        return (3, 3)
    
    # For larger numbers, calculate based on sqrt
    cols = math.ceil(math.sqrt(n_videos))
    rows = math.ceil(n_videos / cols)
    return (rows, cols)


def stitch_videos(
    input_dir: str,
    output_path: str,
    video_pattern: str = "cogvideox-*.mp4",
    font_size: int = 36,
    fps: int = None,
) -> None:
    """
    Stitch multiple videos into a grid comparison layout.
    
    Args:
        input_dir: Directory containing input videos
        output_path: Path for the output comparison video
        video_pattern: Glob pattern to match video files
        font_size: Font size for labels
        fps: Output FPS (None to use source FPS)
    """
    # Find all matching videos
    pattern = os.path.join(input_dir, video_pattern)
    video_files = sorted(glob.glob(pattern))
    
    if not video_files:
        logger.error(f"No videos found matching pattern: {pattern}")
        logger.error(f"Looking in directory: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} videos to stitch:")
    for vf in video_files:
        logger.info(f"  - {os.path.basename(vf)}")
    
    if len(video_files) < 2:
        logger.error("Need at least 2 videos to create a comparison grid")
        sys.exit(1)
    
    # Create labeled clips
    logger.info("Creating labeled clips...")
    labeled_clips = []
    for video_path in video_files:
        attn_type = extract_attention_type(video_path)
        logger.info(f"  Processing: {attn_type}")
        clip = create_labeled_clip(video_path, attn_type, font_size)
        labeled_clips.append(clip)
    
    # Get target dimensions from first clip
    target_size = labeled_clips[0].size
    logger.info(f"Target clip size: {target_size}")
    
    # Resize all clips to match (in case of slight differences)
    resized_clips = []
    for clip in labeled_clips:
        if clip.size != target_size:
            clip = clip.resize(target_size)
        resized_clips.append(clip)
    
    # Calculate grid layout
    rows, cols = calculate_grid_dimensions(len(resized_clips))
    logger.info(f"Grid layout: {rows} rows x {cols} columns")
    
    # Pad with None if needed to fill the grid
    total_cells = rows * cols
    import moviepy
    major_version = int(moviepy.__version__.split('.')[0])
    
    while len(resized_clips) < total_cells:
        # Create a black placeholder clip
        placeholder = ColorClip(
            size=target_size,
            color=(0, 0, 0)
        )
        if major_version >= 2:
            placeholder = placeholder.with_duration(resized_clips[0].duration)
        else:
            placeholder = placeholder.set_duration(resized_clips[0].duration)
        resized_clips.append(placeholder)
    
    # Arrange into grid
    grid = []
    idx = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(resized_clips[idx])
            idx += 1
        grid.append(row)
    
    # Create the final grid video
    logger.info("Creating grid layout...")
    final_clip = clips_array(grid)
    
    # Determine output FPS
    if fps is None:
        fps = labeled_clips[0].fps or 8  # Default to 8 fps if not set
    
    # Write output
    logger.info(f"Writing output to: {output_path}")
    logger.info(f"Output FPS: {fps}")
    
    final_clip.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        audio=False,  # CogVideoX outputs don't have audio
        logger=None,  # Suppress moviepy's verbose output
    )
    
    # Cleanup
    logger.info("Cleaning up...")
    final_clip.close()
    for clip in labeled_clips:
        clip.close()
    
    logger.info(f"✓ Comparison video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch attention comparison videos into a grid layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stitch_comparison.py --input_dir ./rendered_videos
  python stitch_comparison.py --input_dir ./videos --output my_comparison.mp4
  python stitch_comparison.py --input_dir . --pattern "*.mp4" --font_size 48
        """
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./rendered_videos',
        help='Directory containing rendered videos (default: ./rendered_videos)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_grid.mp4',
        help='Output video filename (default: comparison_grid.mp4)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='cogvideox-*.mp4',
        help='Glob pattern to match video files (default: cogvideox-*.mp4)'
    )
    parser.add_argument(
        '--font_size',
        type=int,
        default=36,
        help='Font size for attention type labels (default: 36)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Output FPS (default: same as source)'
    )
    
    args = parser.parse_args()
    
    # Resolve output path
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.input_dir, output_path)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    stitch_videos(
        input_dir=args.input_dir,
        output_path=output_path,
        video_pattern=args.pattern,
        font_size=args.font_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

