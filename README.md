# 🍬 Candy AI – Intelligent Candy Identification System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-FF6B35)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B)
![License](https://img.shields.io/badge/License-Proprietary-0F172A)

Candy AI is an enterprise-grade Computer Vision and Artificial Intelligence web application powered by **Ultralytics YOLOv8** and **Streamlit**. It provides automated multi-object localization, brand categorization, real-time quantity counting, and audit reporting across **20 distinct confectionery categories** using an existing trained custom model (`Model/best.pt`).

---

## 📌 Key Capabilities

- **🎯 Multi-Object Candy Detection:** Detects, classifies, and bounds multiple candies in a single image with bounding boxes, distinct color codes, and confidence scores.
- **🔢 Automated Candy Counting Report:** Dynamic category aggregation with formatted ASCII reports and interactive Plotly distribution charts.
- **📊 Real-Time Executive Dashboard:** Live session analytics, total scans, most frequent candy, class coverage, and activity timelines without hardcoded or fake statistics.
- **📹 Real-Time Webcam Recognition:** Dual-mode video processing supporting continuous live OpenCV streams (with Start/Stop controls and FPS counters) and browser snapshot recognition.
- **🍬 20-Category Product Directory:** Comprehensive catalog of all 20 classes mapped directly from `model.names` with live session detection counters.
- **📜 Session Detection History & Audit:** Persistent session-based history tracking with timestamps, thumbnails, full image drill-downs, and CSV data export.
- **⚡ Hardware Acceleration & CPU Fallback:** Automatic CUDA GPU detection with fallback to CPU.

---

## 🏗️ Model Specifications

- **Architecture:** YOLOv8n (Nano)
- **Task:** Custom Object Detection
- **Trained Weights:** `Model/best.pt`
- **Number of Categories:** 20 classes
- **Input Dimensions:** 640 × 640 px (configurable via UI)
- **Inference Defaults:**
  - Confidence Threshold: `0.25`
  - NMS IoU Threshold: `0.45`
  - Input Resolution (`imgsz`): `640`

### Supported Candy Classes (Class ID 0 – 19)

| ID | Class Label | ID | Class Label |
|:--:|:------------|:--:|:------------|
| `0` | **Bounty** | `10` | **eugenia** |
| `1` | **Dare_caramel** | `11` | **gummy_worms** |
| `2` | **Dare_cocoa** | `12` | **milky_way** |
| `3` | **MMs_peanut** | `13` | **nerds** |
| `4` | **MMs_regular** | `14` | **skittles** |
| `5` | **Mars** | `15` | **snickers** |
| `6` | **Twix_caramel** | `16` | **snickers_creamy** |
| `7` | **Twix_clasic** | `17` | **starbust** |
| `8` | **Twix_white** | `18` | **three_musketeers** |
| `9` | **airheads** | `19` | **twizzlers** |

---

## 🗂️ Project Architecture

```text
Candy-AI/
│
├── app.py                      # Main entrypoint and navigation router
│
├── Model/
│   └── best.pt                 # Existing trained YOLOv8 model weights
│
├── utils/
│   ├── __init__.py             # Public package exports
│   ├── detector.py             # Model loading, YOLO inference, and bounding boxes
│   ├── image_processing.py     # Image validation, conversions, and thumbnails
│   ├── statistics.py           # Metrics calculation and Plotly visualization
│   └── history.py              # Session history manager and CSV export
│
├── pages/
│   ├── dashboard.py            # Module A: Executive Dashboard & live analytics
│   ├── candy_detection.py      # Module B, C, D: Image upload, inference, & counting
│   ├── realtime_detection.py   # Module E: Continuous webcam and snapshot detection
│   ├── candy_categories.py     # Module 7: 20-category directory & scan tallies
│   ├── detection_history.py    # Module F: Session audit logs and detail drill-down
│   └── about_model.py          # Module 8: YOLOv8 technical specifications
│
├── assets/
│   ├── logo.png                # Candy AI high-resolution brand badge
│   └── style.css               # Modern AI dashboard custom styling
│
├── tests/
│   └── test_candy_ai.py        # Automated test suite (10 unit/integration tests)
│
├── requirements.txt            # Python package dependencies
└── README.md                   # Complete documentation
```

---

## 💻 Windows Installation & Execution

### 1. Prerequisites

- Windows 10 or Windows 11 (64-bit)
- Python 3.10+ installed and added to `PATH`
- Web browser (Chrome, Edge, Firefox)
- Webcam (optional, for real-time video detection)

### 2. Environment Setup

Open PowerShell or Command Prompt in the project folder:

```powershell
# Navigate to project directory
cd "c:\Users\Lenovo\Documents\Work Hub\OpenCV\CandyDetection"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch the Application

```powershell
streamlit run app.py
```

The application will start and open automatically in your default browser at:

```text
http://localhost:8501
```

---

## 🧪 Testing Instructions

Verify model performance across various test cases:

### Test Case 1: Single Candy Image
- Upload an image containing a single candy (e.g., one Snickers bar).
- **Expected Result:** Exactly 1 bounding box with high confidence (`> 80%`), Category = `snickers`, Total Candies = 1.

### Test Case 2: Multi-Category Scene
- Upload an image containing multiple candy varieties (e.g., Bounty, Mars, Skittles).
- **Expected Result:** Each distinct candy gets a distinct color-coded bounding box. The counting report lists breakdown per brand.

### Test Case 3: Multiple Candies of the Same Category
- Upload an image with 3–5 packages of MMs or Twix.
- **Expected Result:** Separate bounding boxes for each package. Category count accurately reflects the total quantity.

### Test Case 4: Non-Candy / Negative Image
- Upload a scene without any confectionery (e.g., empty table, office desk, landscape).
- **Expected Result:** 0 objects detected. The application cleanly informs: *"Zero Objects Detected: Try lowering the Confidence Threshold slider in the sidebar"*. No crashes or phantom detections.

### Test Case 5: Visually Similar Candy Packages
- Upload images with similar variants (e.g., `Twix_caramel`, `Twix_clasic`, and `Twix_white` or `snickers` vs `snickers_creamy`).
- **Expected Result:** YOLOv8 differentiates between packaging variants based on distinct class weights.

### Test Case 6: Real-Time Webcam
- Navigate to **📹 Real-Time Detection** and click **▶️ Start Camera**.
- Hold a candy in front of the lens.
- **Expected Result:** Live bounding box tracks the candy with live FPS display and dynamic count updates. Click **⏹️ Stop Camera** to safely release the device.

### Running Automated Test Suite

To run the automated verification suite:

```powershell
.\venv\Scripts\python.exe -m unittest tests/test_candy_ai.py
```

---

## 🛡️ Robustness & Error Handling

- **Missing Model Protection:** If `Model/best.pt` is missing or moved, the app displays a prominent error banner with the exact expected path.
- **Cached Singleton Model:** Model weights are loaded once via `@st.cache_resource` and cached across all pages, preventing memory leaks and redundant loading.
- **Format & Corrupt Image Validation:** Uploads are checked for format (`.jpg`, `.jpeg`, `.png`, `.webp`), size (max 25MB), and valid image header integrity before inference.
- **Graceful Camera Handling:** If the camera is busy or unavailable, clear troubleshooting guidance is provided without application termination.
- **Device Agnostic:** Seamlessly utilizes NVIDIA CUDA if available; defaults to optimized CPU execution otherwise.

---

## 📜 Candy Detection Report Sample Output

When processing an image with multiple candies, the system generates:

```text
========================================
        CANDY DETECTION REPORT          
========================================

Bounty             : 2
Mars               : 1
Snickers           : 3
Skittles           : 4
Twix_clasic        : 2

----------------------------------------
Total Candies      : 12
Unique Categories  : 5
========================================
```

---

## 📄 License & Attribution

Designed and engineered for Candy AI Object Detection applications using Ultralytics YOLOv8 and Streamlit.
All rights reserved.

#   c a n d y - a i - i d e n t i f i c a t i o n  
 