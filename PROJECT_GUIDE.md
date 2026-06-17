# Crowd Density Analyser - Project Guide

## Overview
**Crowd Density Analyser** is a Streamlit web application that detects and counts people in images/videos using the **P2PNet deep learning model**, then calculates crowd density and risk levels.

---

## Architecture

### 1. **Frontend (app.py)**
- Streamlit web interface
- Two modes: **Image Analysis** & **Video Analysis**
- User inputs: image/video + visible area (m²)
- Displays: crowd count, density, risk level, annotated images, heatmaps

### 2. **Backend (crowd_analyzer.py)**
- **P2PNet Model**: Computer vision model trained to detect human heads in images
- **CrowdAnalyzer Class**: Main processor
- **Singleton Pattern**: Loads model once, reuses across requests (memory efficient)

### 3. **Models (models.py)**
- P2PNet architecture definition
- Uses VGG16-BN backbone
- Grid-based point detection (2×2 default grid)

---

## Density Calculation (Core Logic)

### Formula
```
Density = Count / Area
Where:
  - Count = Number of people detected
  - Area  = Visible real-world area (m²)
  - Density = persons per m²
```

### Example
- **Image shows**: 50 people detected
- **Visible area**: 200 m²
- **Density**: 50 / 200 = **0.25 persons/m²**

---

## Risk Classification

Risk levels are determined based on density thresholds:

| Density Range | Risk Level | Action |
|---|---|---|
| < 2 persons/m² | **LOW** | Safe, no action needed |
| 2 – 4 persons/m² | **MODERATE** | Monitor situation |
| 4 – 6 persons/m² | **HIGH** | Consider crowd management |
| > 6 persons/m² | **CRITICAL** | Immediate action required |

### Risk Logic (classify_risk function)
```python
def classify_risk(density):
    if density < 2.0: return "LOW"
    if density < 4.0: return "MODERATE"
    if density < 6.0: return "HIGH"
    return "CRITICAL"
```

---

## Processing Pipeline

### Image Analysis
1. User uploads image + enters visible area
2. Image resized to multiple of 128 (P2PNet requirement)
3. P2PNet detects head coordinates
4. Confidence filtering (threshold = 0.5)
5. **Density calculated**: count / area
6. Risk classified
7. Results: annotated image, heatmap, CSV report

### Video Analysis
1. User uploads video + sampling rate (frames per second to process)
2. Frame extracted at specified intervals
3. Each frame processed like image analysis
4. Results accumulated in DataFrame
5. Graphs: crowd count over time
6. CSV report: time-indexed frame data

---

## Key Components

### CrowdAnalyzer.predict(image_path, area)
**Input:**
- `image_path`: Path to image
- `area`: Real-world area in m²

**Output Dictionary:**
```python
{
    "count": 42,                      # People detected
    "points": [[x1, y1], [x2, y2]], # Head coordinates
    "density": 0.21,                  # persons/m²
    "risk": "LOW",                    # Risk level
    "inference_time": 0.85,           # Processing time (seconds)
    "annotated_image": ndarray,       # Image with dots at heads
    "original_image": PIL.Image,      # Resized original
    "image_resolution": (640, 480)    # Final dimensions
}
```

### Model Parameters
- **Backbone**: VGG16-BN (feature extractor)
- **Grid**: 2×2 division (detects heads in grid cells)
- **Threshold**: 0.5 confidence (filters weak detections)
- **Device**: GPU if available, else CPU

---

## Data Flow

```
User Input (Image + Area)
    ↓
Load Model (singleton)
    ↓
Preprocess Image (resize, normalize)
    ↓
P2PNet Inference (detect head points)
    ↓
Confidence Filtering (threshold 0.5)
    ↓
Count = Filtered Points
Density = Count / Area
Risk = Classify(Density)
    ↓
Generate Annotations (red dots on image)
    ↓
Save & Display Results
```

---

## Important Functions

| Function | Purpose |
|---|---|
| `get_analyzer()` | Get/create singleton analyzer instance |
| `load_model()` | Load P2PNet weights |
| `predict(image_path, area)` | Run detection and density analysis |
| `classify_risk(density)` | Map density to risk level |
| `generate_heatmap(points, shape)` | Create density heatmap (Gaussian blur) |
| `_resize_to_multiple(img, 128)` | Resize image (P2PNet requirement) |
| `_generate_annotated_image()` | Draw circles at detected head locations |

---

## How Heatmap Works

Heatmap visualizes crowd density spatially:
1. For each detected head point, apply Gaussian kernel (sigma=15 pixels)
2. Create 2D density map
3. Apply JET colormap (blue=sparse, red=dense)
4. Displayed as overlay on original image

---

## File Structure
```
Crowd-Analyser/
├── app.py                 # Streamlit interface
├── crowd_analyzer.py      # Core detector class
├── models.py              # P2PNet architecture
├── weights/
│   └── SHTechA.pth        # Pre-trained model weights
└── PROJECT_GUIDE.md       # This file
```

---

## Key Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| Confidence Threshold | 0.5 | Filter weak detections |
| Grid Size | 2×2 | P2PNet divides image into 4 cells |
| Dimension Multiple | 128 | Image height/width must be multiples |
| Heatmap Sigma | 15px | Gaussian spread in heatmap |
| Video Frame Skip | User defined | Process every Nth frame |

---

## Common Interview Questions

**Q: How is crowd density calculated?**
A: Density = Number of people detected / Visible area in m². For example, 100 people in 500 m² = 0.2 persons/m².

**Q: What is the P2PNet model?**
A: It's a deep learning model that detects human heads as point coordinates in images, using grid-based prediction with VGG16-BN backbone.

**Q: Why resize images to multiples of 128?**
A: P2PNet architecture requires input dimensions divisible by 128 due to its encoder-decoder structure with stride limitations.

**Q: How are risk levels determined?**
A: Based on density thresholds: LOW (<2), MODERATE (2-4), HIGH (4-6), CRITICAL (>6) persons/m².

**Q: What does the confidence threshold (0.5) do?**
A: Filters weak detections. Only head points with confidence >0.5 are counted to reduce false positives.

**Q: Why use a singleton pattern for the model?**
A: Avoids reloading the large model weights on every Streamlit rerun, improving performance.

---

## Technologies Used
- **Frontend**: Streamlit (Python web framework)
- **ML Model**: P2PNet (PyTorch)
- **Image Processing**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Hardware**: CUDA (GPU) or CPU fallback

---

*Last Updated: June 2026*
