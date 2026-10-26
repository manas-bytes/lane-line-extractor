"""
Web Dashboard for Lane Line Extractor using OpenCV
=================================================
A modern Flask web application providing an interactive Drag & Drop interface
for extracting lane lines from any highway road image.

How to run:
    python web_app.py
    (or double-click run_web_ui.bat)
"""

import os
import sys
import uuid
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import cv2
import numpy as np

# Import our exact classical pipeline functions from lane_extractor
from lane_extractor import (
    load_image,
    apply_hsv_color_filter,
    apply_gaussian_blur,
    detect_edges,
    apply_roi_mask,
    save_pipeline_stages
)

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedded Single-Page Web UI with Modern Glassmorphism & Drag-and-Drop
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lane Line Extractor - Interactive Drag & Drop</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-secondary: #131b2e;
            --bg-card: rgba(22, 33, 58, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0a0f1d 75%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 1.5rem 4rem;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.9rem;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 640px;
            margin: 0 auto;
        }

        /* Drag and Drop Zone */
        .dropzone-container {
            background: var(--bg-card);
            border: 2px dashed rgba(96, 165, 250, 0.4);
            border-radius: 1.25rem;
            padding: 3.5rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(12px);
            margin-bottom: 2rem;
            position: relative;
        }

        .dropzone-container:hover, .dropzone-container.dragover {
            border-color: var(--accent-cyan);
            background: rgba(30, 58, 102, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 15px 35px -5px rgba(6, 182, 212, 0.2);
        }

        .drop-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 1.25rem;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(6, 182, 212, 0.2));
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
        }

        .drop-icon svg {
            width: 32px;
            height: 32px;
        }

        .drop-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .drop-desc {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }

        .btn-browse {
            display: inline-block;
            background: linear-gradient(135deg, #2563eb, #0891b2);
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.75rem;
            border-radius: 0.75rem;
            font-size: 0.95rem;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
            transition: all 0.2s;
        }

        .btn-browse:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        }

        input[type="file"] {
            display: none;
        }

        /* Preset sample pills */
        .samples-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.25rem;
        }

        .sample-label {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .sample-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: #e2e8f0;
            padding: 0.4rem 0.9rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-btn:hover {
            background: rgba(59, 130, 246, 0.2);
            border-color: rgba(96, 165, 250, 0.5);
            color: #60a5fa;
        }

        /* Loader */
        .loader-card {
            display: none;
            text-align: center;
            padding: 3rem 1rem;
            background: var(--bg-card);
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--accent-cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1.25rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Results Gallery */
        #results-area {
            display: none;
        }

        .results-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .results-header h2 {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .status-badge {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.4rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .stage-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            transition: transform 0.2s;
        }

        .stage-card:hover {
            transform: translateY(-4px);
            border-color: rgba(96, 165, 250, 0.3);
        }

        .stage-img-wrapper {
            position: relative;
            background: #000;
            width: 100%;
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .stage-img-wrapper img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .stage-info {
            padding: 1.1rem;
        }

        .stage-number {
            font-size: 0.75rem;
            font-weight: 700;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .stage-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .stage-desc {
            color: var(--text-muted);
            font-size: 0.85rem;
            line-height: 1.4;
        }

        /* Full Comparison Hero Card */
        .comparison-card {
            background: var(--bg-card);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 1.25rem;
            padding: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .comparison-card h3 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #60a5fa;
        }

        .comparison-card img {
            width: 100%;
            border-radius: 0.75rem;
            border: 1px solid var(--border-color);
        }

        .folder-callout {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            font-size: 0.9rem;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-top: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">Classical Computer Vision • OpenCV</div>
            <h1>Lane Line Extractor Dashboard</h1>
            <p class="subtitle">Drag & drop your own road image below to run the 5-stage classical lane detection pipeline and inspect intermediate stages.</p>
        </header>

        <!-- Drag & Drop Zone -->
        <div class="dropzone-container" id="dropzone">
            <input type="file" id="file-input" accept="image/jpeg,image/png,image/jpg,image/bmp">
            <div class="drop-icon">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
            </div>
            <div class="drop-title">Drag & Drop Your Road Image Here</div>
            <div class="drop-desc">Supports JPG, PNG, BMP (Dashcam photos, highway roads, curved lanes)</div>
            <span class="btn-browse">Or Click to Browse Files</span>

            <div class="samples-bar" onclick="event.stopPropagation()">
                <span class="sample-label">Or test with presets:</span>
                <button class="sample-btn" onclick="processPreset('road1.jpg')">🛣️ Straight (road1)</button>
                <button class="sample-btn" onclick="processPreset('road2.jpg')">🔄 Curved (road2)</button>
                <button class="sample-btn" onclick="processPreset('road3.jpg')">🌳 Shadows (road3)</button>
            </div>
        </div>

        <!-- Processing Loader -->
        <div class="loader-card" id="loader">
            <div class="spinner"></div>
            <h3 style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Processing Pipeline...</h3>
            <p style="color: var(--text-muted); font-size: 0.9rem;">Executing HSV Filtering ➔ Gaussian Blur ➔ Canny Edges ➔ ROI Masking</p>
        </div>

        <!-- Results Gallery -->
        <div id="results-area">
            <div class="results-header">
                <div>
                    <h2 id="result-filename">Processing Results</h2>
                    <p style="color: var(--text-muted); font-size: 0.9rem;" id="result-stats">Analysis details</p>
                </div>
                <div class="status-badge">✓ Completed & Saved</div>
            </div>

            <!-- Full Stage Comparison Card -->
            <div class="comparison-card">
                <h3>Stage-by-Stage Comparison Grid</h3>
                <img id="img-comparison" src="" alt="Comparison Grid">
                <div class="folder-callout">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z"></path></svg>
                    <span>All high-resolution outputs automatically stored in: <strong id="output-disk-path">output/</strong></span>
                </div>
            </div>

            <!-- Individual Stages Grid -->
            <div class="gallery-grid">
                <!-- 1. Original -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-01" src="" alt="Original">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Stage 1</div>
                        <div class="stage-title">Original Road Image</div>
                        <div class="stage-desc">Input frame loaded in BGR space via OpenCV.</div>
                    </div>
                </div>

                <!-- 2. Color Mask -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-02" src="" alt="Color Mask">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Stage 2</div>
                        <div class="stage-title">HSV Color Filter</div>
                        <div class="stage-desc">Isolated yellow and white paint pixels using HSV thresholding.</div>
                    </div>
                </div>

                <!-- 3. Blurred -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-03" src="" alt="Blurred">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Stage 3</div>
                        <div class="stage-title">Gaussian Blur (5x5)</div>
                        <div class="stage-desc">Smoothed asphalt gravel noise and micro-texture.</div>
                    </div>
                </div>

                <!-- 4. Canny Edges -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-04" src="" alt="Edges">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Stage 4</div>
                        <div class="stage-title">Canny Edge Detection</div>
                        <div class="stage-desc">Sharp intensity gradient boundaries (Thresholds: 50, 150).</div>
                    </div>
                </div>

                <!-- 5. ROI Edges -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-05" src="" alt="ROI Edges">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Stage 5</div>
                        <div class="stage-title">ROI Trapezoid Masking</div>
                        <div class="stage-desc">Isolated lane markings within the driving corridor.</div>
                    </div>
                </div>

                <!-- 6. Lane Overlay -->
                <div class="stage-card">
                    <div class="stage-img-wrapper">
                        <img id="img-overlay" src="" alt="Lane Overlay">
                    </div>
                    <div class="stage-info">
                        <div class="stage-number">Final Output</div>
                        <div class="stage-title">Detected Lane Lines Overlay</div>
                        <div class="stage-desc">Extracted lane lines highlighted in vibrant green on the road.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('file-input');
        const loader = document.getElementById('loader');
        const resultsArea = document.getElementById('results-area');

        // Drag & Drop event handlers
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        function handleFileUpload(file) {
            const formData = new FormData();
            formData.append('image', file);

            showLoader();

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                hideLoader();
                if (data.success) {
                    renderResults(data);
                } else {
                    alert('Error processing image: ' + data.error);
                }
            })
            .catch(err => {
                hideLoader();
                alert('Network error: ' + err);
            });
        }

        function processPreset(filename) {
            showLoader();
            fetch('/process_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: filename })
            })
            .then(res => res.json())
            .then(data => {
                hideLoader();
                if (data.success) {
                    renderResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(err => {
                hideLoader();
                alert('Network error: ' + err);
            });
        }

        function showLoader() {
            loader.style.display = 'block';
            resultsArea.style.display = 'none';
        }

        function hideLoader() {
            loader.style.display = 'none';
        }

        function renderResults(data) {
            resultsArea.style.display = 'block';
            document.getElementById('result-filename').innerText = 'Processed: ' + data.image_name;
            document.getElementById('result-stats').innerText = 
                `Resolution: ${data.width}x${data.height}px | Lane pixels: ${data.lane_pixels} | Output directory: output/${data.image_stem}/`;

            document.getElementById('output-disk-path').innerText = data.output_folder;

            const t = new Date().getTime(); // Prevent browser image caching
            document.getElementById('img-01').src = `${data.urls.original}?t=${t}`;
            document.getElementById('img-02').src = `${data.urls.color_filtered}?t=${t}`;
            document.getElementById('img-03').src = `${data.urls.blurred}?t=${t}`;
            document.getElementById('img-04').src = `${data.urls.edges}?t=${t}`;
            document.getElementById('img-05').src = `${data.urls.roi_result}?t=${t}`;
            document.getElementById('img-overlay').src = `${data.urls.overlay}?t=${t}`;
            document.getElementById('img-comparison').src = `${data.urls.comparison}?t=${t}`;

            // Smooth scroll to results
            resultsArea.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/output/<image_stem>/<filename>")
def serve_output(image_stem, filename):
    folder = os.path.join(OUTPUT_DIR, image_stem)
    return send_from_directory(folder, filename)


@app.route("/process", methods=["POST"])
def process_uploaded_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # Save to uploads folder
    safe_name = os.path.basename(file.filename)
    image_stem = os.path.splitext(safe_name)[0]
    upload_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(upload_path)

    # Process through pipeline
    return run_pipeline(upload_path, image_stem)


@app.route("/process_preset", methods=["POST"])
def process_preset():
    data = request.get_json() or {}
    filename = data.get("filename", "road1.jpg")
    preset_path = os.path.join(TEST_IMAGES_DIR, filename)

    if not os.path.exists(preset_path):
        return jsonify({"success": False, "error": f"Preset '{filename}' not found"}), 404

    image_stem = os.path.splitext(filename)[0]
    return run_pipeline(preset_path, image_stem)


def run_pipeline(image_path: str, image_stem: str):
    try:
        # 1. Load image
        original = load_image(image_path)
        height, width = original.shape[:2]

        # 2. HSV Color Filtering
        color_filtered, color_mask = apply_hsv_color_filter(original)

        # 3. Gaussian Blur
        blurred = apply_gaussian_blur(color_filtered)

        # 4. Canny Edge Detection
        edges = detect_edges(blurred)

        # 5. ROI Masking
        roi_edges, roi_mask = apply_roi_mask(edges)

        # Save all stages to output/<image_stem>/
        save_folder = os.path.join(OUTPUT_DIR, image_stem)
        save_pipeline_stages(
            output_dir=save_folder,
            original=original,
            color_mask=color_mask,
            color_filtered=color_filtered,
            blurred=blurred,
            edges=edges,
            roi_edges=roi_edges,
            roi_mask=roi_mask
        )

        lane_pixels = int(np.count_nonzero(roi_edges))

        return jsonify({
            "success": True,
            "image_name": os.path.basename(image_path),
            "image_stem": image_stem,
            "width": width,
            "height": height,
            "lane_pixels": lane_pixels,
            "output_folder": os.path.abspath(save_folder),
            "urls": {
                "original": f"/output/{image_stem}/01_original.jpg",
                "color_mask": f"/output/{image_stem}/02_color_mask.jpg",
                "color_filtered": f"/output/{image_stem}/02_color_filtered.jpg",
                "blurred": f"/output/{image_stem}/03_blurred.jpg",
                "edges": f"/output/{image_stem}/04_edges.jpg",
                "roi_result": f"/output/{image_stem}/05_roi_result.jpg",
                "overlay": f"/output/{image_stem}/05_lane_overlay.jpg",
                "comparison": f"/output/{image_stem}/06_comparison.jpg",
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    port = 5000
    print("\n" + "=" * 65)
    print("  🚀 Lane Line Extractor Web Dashboard is Starting!")
    print(f"  👉 Opening: http://127.0.0.1:{port}")
    print("  👉 Drag and drop any road image into the browser window.")
    print("=" * 65 + "\n")

    # Automatically open default browser after 1 second
    Timer(1.2, open_browser).start()
    app.run(host="127.0.0.1", port=port, debug=False)
