# 🛣️ Lane Line Extractor using OpenCV

A clean, modular computer vision project that extracts and isolates highway lane markings from static road images using **classical computer vision techniques** (no machine learning or deep neural networks).

Built with **Python**, **OpenCV**, and **NumPy**, this project implements an end-to-end 5-stage image processing pipeline designed to be transparent, mathematically intuitive, and easy to explain during a viva or academic project defense.

---

## 📁 Project Structure

```text
lane-line-extractor/
├── lane_extractor.py     # Main Python script containing the 5-stage pipeline
├── requirements.txt      # Project dependencies (opencv-python, numpy, matplotlib)
├── README.md             # Complete project documentation and results
├── .gitignore            # Git exclusion rules for clean commits
├── test_images/          # Input road images (straight, curved, shadowed)
│   ├── road1.jpg         # Straight highway road with solid/dashed white lines
│   ├── road2.jpg         # Curved road with solid yellow line
│   └── road3.jpg         # Road with dynamic tree shadows and lighting shifts
└── output/               # Saved pipeline outputs (one subfolder per image)
    ├── road1/
    ├── road2/
    └── road3/
```

---

## ⚙️ How It Works (The 5-Stage Pipeline)

The pipeline processes input images in strict chronological order. Each stage performs a dedicated computer vision operation:

```
[1. Load Image] ──> [2. HSV Color Filter] ──> [3. Gaussian Blur] ──> [4. Canny Edges] ──> [5. ROI Mask]
```

### 1. Load Image (`cv2.imread`)
- **What it does:** Reads the raw image from the filesystem into memory as a NumPy matrix. It also checks whether the file exists and is uncorrupted.
- **Why it is needed:** If an image fails to load, `cv2.imread()` silently returns `None` instead of throwing an error. Checking for `None` prevents obscure downstream crashes and provides clear error feedback.
- **Key Concept:** OpenCV loads images in **BGR (Blue, Green, Red)** color order by default, rather than the standard RGB format.

### 2. HSV Color Filtering (`cv2.cvtColor`, `cv2.inRange`, `cv2.bitwise_or`, `cv2.bitwise_and`)
- **What it does:** Converts the BGR image to the **HSV (Hue, Saturation, Value)** color space. It isolates yellow lines (via hue thresholds) and white lines (via low saturation + high brightness thresholds), merges both masks using `cv2.bitwise_or()`, and applies the mask to the image using `cv2.bitwise_and()`.
- **Why it is needed:** In standard RGB, shadows and changing sunlight alter all three color channels simultaneously. HSV separates chromatic color information (**Hue**) from illumination intensity (**Value**), allowing consistent detection of yellow and white lane paint across varied asphalt colors and light conditions.

### 3. Gaussian Blur (`cv2.GaussianBlur`)
- **What it does:** Smooths the image by convolving it with a $5 \times 5$ Gaussian kernel.
- **Why it is needed:** Asphalt surfaces contain granular textures, small pebbles, cracks, and light reflections. If passed directly to an edge detector, these micro-textures produce hundreds of false-positive edges. Gaussian blurring averages neighboring pixels using a bell-shaped weight curve, suppressing high-frequency noise while preserving prominent lane boundaries.

### 4. Canny Edge Detection (`cv2.Canny`)
- **What it does:** Converts the blurred image to single-channel grayscale, computes intensity gradients using the Sobel operator, suppresses non-maximum pixels, and applies **hysteresis thresholding** with low threshold 50 and high threshold 150.
- **Why it is needed:** Lane lines are characterized by steep intensity transitions between the painted marking and the dark pavement. Canny identifies these sharp boundaries and thins them down to 1-pixel-wide edges. Pixels with gradients above 150 are classified as strong edges, while pixels between 50 and 150 are kept only if connected to a strong edge.

### 5. Region of Interest (ROI) Masking (`cv2.fillPoly`, `cv2.bitwise_and`)
- **What it does:** Constructs a trapezoidal polygon mask covering the ego-lane on the road surface and applies it with `cv2.bitwise_and()`. Vertices are defined as **fractions of image width and height** rather than hardcoded pixel counts.
- **Why it is needed:** The camera views the entire forward scene, including the sky, mountains, trees, oncoming traffic, and roadside guardrails. Lane markers exist strictly on the triangular/trapezoidal road area ahead of the vehicle. Masking discards irrelevant peripheral edges.
- **Why fractional coordinates:** Using relative fractions (e.g., $0.08 \times \text{width}$, $0.60 \times \text{height}$) ensures the mask automatically scales and functions correctly across any image resolution (480p, 720p, 1080p, 4K) without manual reconfiguration.

---

## 📊 Pipeline Parameters

All pipeline parameters are configured as constants at the top of [`lane_extractor.py`](lane_extractor.py):

| Parameter | Value | Description & Selection Rationale |
| :--- | :--- | :--- |
| **Yellow HSV Lower** | `[15, 80, 100]` | Hue $\ge 15$ captures golden/yellow shades; Saturation $\ge 80$ avoids washed-out gray asphalt; Value $\ge 100$ filters dark shadows. |
| **Yellow HSV Upper** | `[35, 255, 255]` | Hue $\le 35$ bounds yellow before it turns green; allows full saturation and brightness. |
| **White HSV Lower** | `[0, 0, 190]` | Hue spans full range ($0$ to $180$); Saturation $\le 40$ ensures white lacks color tint; Value $\ge 190$ captures bright white paint. |
| **White HSV Upper** | `[180, 40, 255]` | Allows maximum brightness for white markings while excluding off-white road concrete. |
| **Gaussian Blur Kernel** | `(5, 5)` | $5 \times 5$ odd matrix; optimal compromise between smoothing out asphalt texture without washing out thin dashed lane lines. |
| **Canny Low Threshold** | `50` | Lower hysteresis threshold; accepts weak edges that are connected to prominent lane markings. |
| **Canny High Threshold** | `150` | Upper hysteresis threshold (standard 1:3 Canny ratio with low threshold 50); confirms strong lane edge candidates. |
| **ROI Trapezoid Vertices** | `[[0.08, 0.95], [0.43, 0.60], [0.57, 0.60], [0.95, 0.95]]` | Bottom corners spread across the hood; top corners converge near the road horizon ($60\%$ height) to fit road perspective. |

---

## 🚀 Setup and Usage

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/lane-line-extractor.git
cd lane-line-extractor
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

#### Process all test images (default):
```bash
python lane_extractor.py
```

#### Process a single image:
```bash
python lane_extractor.py --input test_images/road1.jpg
```

#### Process and display results in an interactive window:
```bash
python lane_extractor.py --input test_images/road1.jpg --show
```

#### Specify a custom output folder:
```bash
python lane_extractor.py --input test_images/ --output-dir my_results
```

---

## 📸 Results

For every processed image, the script generates individual stage outputs in `output/<image_name>/` as well as a side-by-side comparison sheet:

### 1. Straight Highway Road (`road1.jpg`)
*Features clean pavement, a solid white right line, and a dashed white center line.*

| Stage | Image |
| :--- | :--- |
| **1. Original Image** | ![Original](output/road1/01_original.jpg) |
| **2. HSV Color Mask** | ![Color Mask](output/road1/02_color_mask.jpg) |
| **3. Gaussian Blurred** | ![Blurred](output/road1/03_blurred.jpg) |
| **4. Canny Edges** | ![Edges](output/road1/04_edges.jpg) |
| **5. ROI Isolated Edges** | ![ROI Result](output/road1/05_roi_result.jpg) |
| **Final Overlay** | ![Lane Overlay](output/road1/05_lane_overlay.jpg) |

#### Full Stage Comparison Sheet:
![road1 Comparison](output/road1/06_comparison.jpg)

---

### 2. Curved Road with Yellow Line (`road2.jpg`)
*Features a solid yellow lane on the left and dashed white on the right entering a bend.*

| Stage | Image |
| :--- | :--- |
| **1. Original Image** | ![Original](output/road2/01_original.jpg) |
| **2. HSV Color Mask** | ![Color Mask](output/road2/02_color_mask.jpg) |
| **3. Gaussian Blurred** | ![Blurred](output/road2/03_blurred.jpg) |
| **4. Canny Edges** | ![Edges](output/road2/04_edges.jpg) |
| **5. ROI Isolated Edges** | ![ROI Result](output/road2/05_roi_result.jpg) |
| **Final Overlay** | ![Lane Overlay](output/road2/05_lane_overlay.jpg) |

#### Full Stage Comparison Sheet:
![road2 Comparison](output/road2/06_comparison.jpg)

---

### 3. Road with Tree Shadows (`road3.jpg`)
*Tests robustness against heavy tree shadows casting dark patterns across asphalt and lane lines.*

| Stage | Image |
| :--- | :--- |
| **1. Original Image** | ![Original](output/road3/01_original.jpg) |
| **2. HSV Color Mask** | ![Color Mask](output/road3/02_color_mask.jpg) |
| **3. Gaussian Blurred** | ![Blurred](output/road3/03_blurred.jpg) |
| **4. Canny Edges** | ![Edges](output/road3/04_edges.jpg) |
| **5. ROI Isolated Edges** | ![ROI Result](output/road3/05_roi_result.jpg) |
| **Final Overlay** | ![Lane Overlay](output/road3/05_lane_overlay.jpg) |

#### Full Stage Comparison Sheet:
![road3 Comparison](output/road3/06_comparison.jpg)

---

## 🖼️ Test Images

The three sample road images are included in the `test_images/` directory:

1. **`road1.jpg`** — Straight highway with solid white line on the right and dashed lane markings.
2. **`road2.jpg`** — Gentle curve displaying a solid yellow divider on the left.
3. **`road3.jpg`** — Variable pavement lighting with distinct tree shadows crossing the lane.

### Where to get more test images:
If you want to test on additional images, you can obtain free, open-source road footage from:
- **Udacity Self-Driving Car Nanodegree Dataset** (MIT License): [GitHub Repository](https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_images)
  - `solidWhiteCurve.jpg`
  - `whiteCarLaneSwitch.jpg`
  - `solidYellowLeft.jpg`
- **Kaggle TuSimple Lane Detection Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/manideep1108/tusimple)
- **Unsplash & Pexels (Free commercial use)**: Search for `"highway road dashcam"` or `"empty asphalt highway"`. Save images into `test_images/<name>.jpg`.

---

## ⚠️ Limitations of Classical Computer Vision

While classical CV is fast, interpretable, and computationally lightweight, it has inherent challenges:

1. **Extreme Shadows and Direct Glare:** Dark tree shadows or sun glare reflecting off asphalt can cause intensity spikes that trigger false edges or obscure faint markings.
2. **Worn or Faded Paint:** If road markings have deteriorated, their color saturation and brightness fall below the HSV threshold, leading to fragmented edges.
3. **Sharp Hairpin Curves:** The fixed trapezoidal ROI assumes the lane remains ahead within the polygon. Sharp turns cause lane lines to exit the side of the trapezoid before reaching the horizon.
4. **Nighttime & Rain:** Wet asphalt creates mirror-like headlight reflections, creating dozens of spurious vertical lines that confuse Canny edge detection.
5. **Concrete vs. Asphalt:** Light-colored concrete highways have low contrast against white lane paint, making threshold separation difficult without adaptive histogram equalization.

---

## 🔮 Future Improvements

1. **Hough Line Transform (`cv2.HoughLinesP`):** Convert detected edge pixels into mathematical line segments and extrapolate continuous left/right lane lines using linear regression.
2. **Perspective Transform (Bird's Eye View):** Warp the road perspective into an overhead top-down view to measure curvature and vehicle offset from lane center.
3. **Polynomial Lane Fitting ($2^{\text{nd}}$ Order):** Fit quadratic curves ($x = Ay^2 + By + C$) to support sharp curves and highway exits.
4. **Video Stream Processing:** Implement temporal smoothing (exponential moving average or Kalman filtering) across consecutive video frames to prevent lane flicker.
5. **Adaptive Thresholding:** Dynamically adjust HSV values based on overall frame luminance to handle tunnel entrances and sudden lighting changes.

---

## 🎓 Viva / Project Defense Tips

When explaining this project to an examiner, keep these concise points in mind:
- **Why not RGB?** RGB mixes light intensity and color in every channel. HSV separates them, making it immune to pure brightness changes.
- **Why 5x5 Gaussian?** Smaller kernels ($3 \times 3$) leave too much asphalt noise; larger kernels ($9 \times 9$) blur and weaken thin dashed lines.
- **Why Canny thresholds 50 and 150?** A $1:3$ ratio (50 low, 150 high) is the recommended standard by John F. Canny to balance noise rejection and continuous edge tracing.
- **Why fractional ROI?** Relative coordinates ($0.08$ to $0.95$) make the code resolution-agnostic, working identically on 720p or 4K dashcam frames.

---

## 📜 License

This project is licensed under the **MIT License** — free for academic and personal use.
