"""
Lane Line Extractor using OpenCV (Classical Computer Vision)
============================================================
A modular, classical computer vision pipeline to extract highway lane markings
from static road images without machine learning or deep learning models.

Pipeline stages:
    1. Load image (cv2.imread with BGR check)
    2. HSV color filtering (Yellow + White line detection via cv2.inRange and bitwise_or)
    3. Gaussian blur (5x5 kernel smoothing to suppress high-frequency noise)
    4. Canny edge detection (Intensity gradient calculation with hysteresis thresholding)
    5. Region of Interest (ROI) masking (Adaptive fractional trapezoid polygon)
"""

import os
import sys
import argparse
from typing import Tuple, List, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PIPELINE CONFIGURATION CONSTANTS (Tweak these parameters to tune detection)
# ==============================================================================

# Step 2: HSV Color Filtering Thresholds
# Note: In OpenCV, Hue values range from 0 to 179, Saturation from 0 to 255, Value from 0 to 255.

# Yellow lane line threshold (Hue ~15-35 captures typical yellow highway paint)
YELLOW_HSV_LOWER = np.array([15, 80, 100], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# White lane line threshold (Low saturation and high brightness/value)
WHITE_HSV_LOWER = np.array([0, 0, 190], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 40, 255], dtype=np.uint8)

# Step 3: Gaussian Blur Kernel Size (must be an odd integer, e.g., 3, 5, 7)
GAUSSIAN_KERNEL_SIZE = 5

# Step 4: Canny Edge Detection Thresholds
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Step 5: Region of Interest (ROI) Trapezoid Vertices defined as FRACTIONS (x, y)
# Coordinate system: (0.0, 0.0) is top-left, (1.0, 1.0) is bottom-right.
# Polygon order: [Bottom-Left, Top-Left, Top-Right, Bottom-Right]
ROI_VERTICES_FRACTIONS = np.array([
    [0.08, 0.95],  # Bottom-Left: Near the bottom-left edge of the road
    [0.43, 0.60],  # Top-Left: Horizon/vanishing point boundary
    [0.57, 0.60],  # Top-Right: Horizon/vanishing point boundary
    [0.95, 0.95]   # Bottom-Right: Near the bottom-right edge of the road
], dtype=np.float32)


# ==============================================================================
# PIPELINE FUNCTIONS (Implemented in strict pipeline order)
# ==============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Step 1: Load an image from disk using OpenCV.

    What it does:
        Reads the image file at `image_path` into an in-memory NumPy ndarray.

    Why it is needed:
        OpenCV's cv2.imread() returns None if the path is invalid, the file is
        missing, or the file format is corrupted. Explicit validation ensures
        the program fails gracefully with an actionable error message rather
        than crashing downstream.

    Important note:
        OpenCV loads images in BGR (Blue, Green, Red) channel order by default,
        unlike standard RGB formats.

    Args:
        image_path: Filesystem path to the input image.

    Returns:
        np.ndarray: The loaded image in BGR format.

    Raises:
        FileNotFoundError: If the image cannot be read or does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"[ERROR] Image file not found at: '{image_path}'. "
            f"Please verify the path and file name."
        )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"[ERROR] OpenCV failed to decode the image at: '{image_path}'. "
            f"Check that the file is a valid, uncorrupted image (JPG, PNG, etc.)."
        )

    return image


def apply_hsv_color_filter(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Filter yellow and white road markings using the HSV color space.

    What it does:
        1. Converts the BGR image to HSV (Hue, Saturation, Value).
        2. Creates a binary mask for yellow lane markings using cv2.inRange().
        3. Creates a binary mask for white lane markings using cv2.inRange().
        4. Combines both masks using cv2.bitwise_or().
        5. Applies the combined mask to the original image using cv2.bitwise_and().

    Why it is needed:
        Standard RGB color space blends color and illumination into every channel,
        making it extremely sensitive to shadows, sun glare, and pavement color shifts.
        The HSV color space separates chromaticity (Hue) from brightness (Value),
        making yellow and white line identification much more robust under varying
        lighting conditions.

    Args:
        image: Original BGR input image.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - filtered_image: Color image with only yellow/white pixels retained (rest is black).
            - combined_mask: 1-channel binary mask (255 for lane colors, 0 elsewhere).
    """
    # 1. Convert BGR color space to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Threshold yellow colors
    yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    # 3. Threshold white colors
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

    # 4. Combine masks: Keep any pixel that is either yellow OR white
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # 5. Apply combined mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

    return filtered_image, combined_mask


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = GAUSSIAN_KERNEL_SIZE) -> np.ndarray:
    """
    Step 3: Suppress high-frequency noise using Gaussian Blur smoothing.

    What it does:
        Convolves the image with a 2D Gaussian kernel of size (kernel_size, kernel_size).

    Why it is needed:
        Raw asphalt surfaces contain high-frequency noise such as gravel texture,
        cracks, and speckled reflections. If passed directly to an edge detector,
        these micro-textures would produce hundreds of false-positive edges.
        Gaussian blurring smoothly averages neighboring pixels, preserving prominent
        lane edges while blurring out tiny surface noise.

    Args:
        image: Image to smooth (typically the color-filtered result from Step 2).
        kernel_size: Odd integer specifying the filter dimension (default: 5).

    Returns:
        np.ndarray: Gaussian-smoothed image.
    """
    # Ensure kernel size is positive and odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def detect_edges(
    image: np.ndarray,
    low_threshold: int = CANNY_LOW_THRESHOLD,
    high_threshold: int = CANNY_HIGH_THRESHOLD
) -> np.ndarray:
    """
    Step 4: Detect boundary edges using the Canny Edge Detection algorithm.

    What it does:
        1. Converts the image to single-channel grayscale if it has 3 channels.
        2. Applies the Canny algorithm (computes intensity gradients, performs
           non-maximum suppression, and applies hysteresis thresholding).

    Why it is needed:
        Lane lines appear as sharp intensity transitions between the painted marking
        and the dark asphalt road. Canny edge detection calculates the gradient
        magnitude and direction at every pixel, isolating thin 1-pixel-wide line
        boundaries. Hysteresis thresholding retains strong edges (> high_threshold)
        and preserves weak edges (> low_threshold) only if connected to strong ones.

    Args:
        image: Blurred image from Step 3.
        low_threshold: Lower bound for hysteresis edge linking (default: 50).
        high_threshold: Upper bound for strong edge detection (default: 150).

    Returns:
        np.ndarray: 1-channel binary edge map (255 for edge pixels, 0 for background).
    """
    # Convert to grayscale first if the input is a 3-channel color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def apply_roi_mask(
    edges: np.ndarray,
    fractions: np.ndarray = ROI_VERTICES_FRACTIONS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 5: Mask the Region of Interest (ROI) using an adaptive trapezoid.

    What it does:
        1. Calculates absolute pixel coordinates from fractional vertices based on
           the input image width and height.
        2. Creates an all-black mask with the same dimensions as the edge map.
        3. Fills the trapezoid polygon with white (255) using cv2.fillPoly().
        4. Applies the mask using cv2.bitwise_and() so only edges inside the
           trapezoid are retained.

    Why it is needed:
        Dashboard and windshield cameras capture the entire forward environment:
        trees, oncoming cars, horizon, guardrails, and sky. Lane lines exist only
        on the drivable road surface directly in front of the vehicle. Masking out
        the irrelevant regions eliminates non-road edges and focuses processing on
        the driving lane.

    Why fractional coordinates:
        Hardcoded pixel coordinates (e.g., [400, 300]) break when images have
        different resolutions (e.g., 720p vs 1080p). Defining vertices as
        fractions of image width and height ensures the mask automatically adapts
        to any image dimensions.

    Args:
        edges: Binary edge map from Step 4.
        fractions: Array of (x_fraction, y_fraction) defining the trapezoid corners.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - roi_edges: Binary edge map containing only edges within the ROI.
            - mask: The binary trapezoid mask (useful for visualization).
    """
    height, width = edges.shape[:2]

    # Convert fractional coordinates to absolute pixel vertices
    pixel_vertices = np.array([
        [int(fx * width), int(fy * height)] for fx, fy in fractions
    ], dtype=np.int32)

    # Create a single-channel black mask matching the image dimensions
    mask = np.zeros_like(edges)

    # Fill the polygon defined by the vertices with white (value 255)
    cv2.fillPoly(mask, [pixel_vertices], 255)

    # Perform bitwise AND: keep edge pixels only where the mask is white
    roi_edges = cv2.bitwise_and(edges, mask)

    return roi_edges, mask


# ==============================================================================
# VISUALIZATION & OUTPUT SAVING HELPERS
# ==============================================================================

def save_pipeline_stages(
    output_dir: str,
    original: np.ndarray,
    color_mask: np.ndarray,
    color_filtered: np.ndarray,
    blurred: np.ndarray,
    edges: np.ndarray,
    roi_edges: np.ndarray,
    roi_mask: np.ndarray
) -> str:
    """
    Saves individual images for each stage of the pipeline into the output directory,
    plus an all-in-one side-by-side comparison figure.

    Args:
        output_dir: Directory to store the output images.
        original: Original BGR image.
        color_mask: Binary yellow+white color mask.
        color_filtered: BGR image masked with yellow+white filter.
        blurred: Blurred color image.
        edges: Canny edge detector result.
        roi_edges: Edges filtered by the ROI trapezoid.
        roi_mask: Binary ROI trapezoid mask.

    Returns:
        str: Filepath of the generated comparison image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Original Image
    cv2.imwrite(os.path.join(output_dir, "01_original.jpg"), original)

    # Stage 2: HSV Color Mask and Filtered Image
    cv2.imwrite(os.path.join(output_dir, "02_color_mask.jpg"), color_mask)
    cv2.imwrite(os.path.join(output_dir, "02_color_filtered.jpg"), color_filtered)

    # Stage 3: Gaussian Blurred
    cv2.imwrite(os.path.join(output_dir, "03_blurred.jpg"), blurred)

    # Stage 4: Canny Edges
    cv2.imwrite(os.path.join(output_dir, "04_edges.jpg"), edges)

    # Stage 5: ROI Isolated Edges
    cv2.imwrite(os.path.join(output_dir, "05_roi_result.jpg"), roi_edges)

    # Build Overlay: Highlight detected lane edges in vibrant green on the original image
    overlay = original.copy()
    overlay[roi_edges > 0] = [0, 255, 0]  # Green in BGR
    cv2.imwrite(os.path.join(output_dir, "05_lane_overlay.jpg"), overlay)

    # Generate a comprehensive 2x3 comparison figure using Matplotlib
    comparison_path = os.path.join(output_dir, "06_comparison.jpg")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Lane Line Extraction Pipeline (OpenCV Classical CV)", fontsize=18, fontweight="bold", y=0.98)

    # Panel 1: Original Image (convert BGR to RGB for correct Matplotlib display)
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image (Input)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: HSV Color Filtered
    axes[0, 1].imshow(cv2.cvtColor(color_filtered, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. HSV Color Filter (Yellow & White)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Gaussian Blur
    axes[0, 2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"3. Gaussian Blur ({GAUSSIAN_KERNEL_SIZE}x{GAUSSIAN_KERNEL_SIZE} Kernel)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Panel 4: Canny Edge Detection
    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title(f"4. Canny Edges (Thresholds: {CANNY_LOW_THRESHOLD}, {CANNY_HIGH_THRESHOLD})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 5: ROI Mask Trapezoid
    # Show ROI mask overlaid on edges for clarity
    roi_viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.polylines(
        roi_viz,
        [np.array([[int(fx * original.shape[1]), int(fy * original.shape[0])] for fx, fy in ROI_VERTICES_FRACTIONS], dtype=np.int32)],
        isClosed=True,
        color=(0, 0, 255),  # Red polygon border in BGR
        thickness=3
    )
    axes[1, 1].imshow(cv2.cvtColor(roi_viz, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Region of Interest (Trapezoid)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # Panel 6: Final Extracted Lane Lines overlaid on original
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Extracted Lane Lines Overlay", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")

    return comparison_path


def process_image(image_path: str, output_base_dir: str = "output", show: bool = False) -> None:
    """
    Executes the complete 5-stage lane line extraction pipeline on a single image.

    Args:
        image_path: Path to the input image file.
        output_base_dir: Root directory for saving results.
        show: If True, displays the comparison figure in an interactive window.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_base_dir, image_name)

    print(f"\n{'=' * 60}")
    print(f"Processing: {image_path}")
    print(f"{'=' * 60}")

    # Stage 1: Load image
    print("[1/5] Loading image...")
    original = load_image(image_path)
    height, width, channels = original.shape
    print(f"      Dimensions: {width}x{height} pixels, {channels} color channels")

    # Stage 2: HSV Color Filtering
    print("[2/5] Applying HSV yellow and white color filtering...")
    color_filtered, color_mask = apply_hsv_color_filter(original)
    detected_color_pixels = np.count_nonzero(color_mask)
    print(f"      Identified {detected_color_pixels:,} lane color pixels")

    # Stage 3: Gaussian Blur
    print(f"[3/5] Applying Gaussian blur (kernel size: {GAUSSIAN_KERNEL_SIZE}x{GAUSSIAN_KERNEL_SIZE})...")
    blurred = apply_gaussian_blur(color_filtered, GAUSSIAN_KERNEL_SIZE)

    # Stage 4: Canny Edge Detection
    print(f"[4/5] Running Canny edge detection (thresholds: {CANNY_LOW_THRESHOLD}, {CANNY_HIGH_THRESHOLD})...")
    edges = detect_edges(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    total_edge_pixels = np.count_nonzero(edges)
    print(f"      Detected {total_edge_pixels:,} total edge pixels")

    # Stage 5: ROI Masking
    print("[5/5] Applying Region of Interest (ROI) trapezoid mask...")
    roi_edges, roi_mask = apply_roi_mask(edges, ROI_VERTICES_FRACTIONS)
    lane_edge_pixels = np.count_nonzero(roi_edges)
    print(f"      Retained {lane_edge_pixels:,} lane line edge pixels inside ROI")

    # Save all stages to disk
    comparison_file = save_pipeline_stages(
        output_dir=image_output_dir,
        original=original,
        color_mask=color_mask,
        color_filtered=color_filtered,
        blurred=blurred,
        edges=edges,
        roi_edges=roi_edges,
        roi_mask=roi_mask
    )
    print(f"[SUCCESS] All stage outputs saved to: '{image_output_dir}/'")
    print(f"          Comparison sheet saved to:  '{comparison_file}'")

    # Optional interactive display
    if show:
        print("[DISPLAY] Showing comparison window. Close the window to continue...")
        plt.show()
    else:
        plt.close("all")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Parses command-line arguments and coordinates image processing.
    """
    parser = argparse.ArgumentParser(
        description="Lane Line Extractor using OpenCV (Classical Computer Vision)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to an input road image or directory of images. "
             "If omitted, processes all images in 'test_images/'."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Directory where intermediate stages and comparison images are saved."
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the side-by-side comparison figure in an interactive GUI window."
    )
    parser.add_argument(
        "--drag", "-d",
        action="store_true",
        help="Interactive Drag & Drop mode: prompts you to drag and drop any image into the terminal."
    )

    args = parser.parse_args()

    # Interactive drag-and-drop prompt
    if args.drag:
        print("\n" + "=" * 60)
        print(" [DRAG & DROP MODE] Drag any road image into this terminal")
        print(" and press ENTER to process it.")
        print("=" * 60 + "\n")
        try:
            dropped_path = input("Drag & drop image file here: ").strip()
            if not dropped_path:
                print("[INFO] No file dropped. Exiting.")
                sys.exit(0)
            args.input = dropped_path
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Cancelled by user.")
            sys.exit(0)

    # Clean any quotes Windows terminal adds when dropping files
    if args.input:
        args.input = args.input.strip().strip('\'"')

    # Determine which images to process
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths: List[str] = []

    if args.input is None:
        # Default behavior: process all images in test_images/ folder
        default_folder = "test_images"
        if not os.path.isdir(default_folder):
            print(f"[ERROR] Default test images folder '{default_folder}' does not exist.")
            print("Please create 'test_images/' and add road images, or use --input <path>.")
            sys.exit(1)

        candidates = [
            os.path.join(default_folder, fname)
            for fname in sorted(os.listdir(default_folder))
            if os.path.splitext(fname)[1].lower() in valid_extensions
        ]
        if not candidates:
            print(f"[ERROR] No valid images found in '{default_folder}'.")
            print(f"Supported formats: {', '.join(valid_extensions)}")
            sys.exit(1)

        image_paths = candidates
        print(f"Found {len(image_paths)} image(s) in '{default_folder}'.")

    elif os.path.isdir(args.input):
        # User specified a directory
        candidates = [
            os.path.join(args.input, fname)
            for fname in sorted(os.listdir(args.input))
            if os.path.splitext(fname)[1].lower() in valid_extensions
        ]
        if not candidates:
            print(f"[ERROR] No valid images found in directory: '{args.input}'")
            sys.exit(1)
        image_paths = candidates
        print(f"Found {len(image_paths)} image(s) in directory '{args.input}'.")

    elif os.path.isfile(args.input):
        # User specified a single file
        image_paths = [args.input]

    else:
        print(f"[ERROR] The input path '{args.input}' was not found as a file or directory.")
        sys.exit(1)

    # Process all collected image paths
    for idx, path in enumerate(image_paths, start=1):
        try:
            process_image(
                image_path=path,
                output_base_dir=args.output_dir,
                show=args.show
            )
        except Exception as exc:
            print(f"[ERROR] Failed to process '{path}': {exc}", file=sys.stderr)

    print(f"\n[DONE] Finished processing {len(image_paths)} image(s).")
    print(f"Outputs are available in '{os.path.abspath(args.output_dir)}'.\n")


if __name__ == "__main__":
    main()
