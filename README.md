# 🍬 Candy AI – Intelligent Candy Identification System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-FF6B35)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-0F172A)

Candy AI is an enterprise-grade Computer Vision and Artificial Intelligence web application powered by **Ultralytics YOLOv8** and **Streamlit**. It provides automated multi-object localization, brand categorization, real-time quantity counting, and audit reporting across **20 distinct confectionery categories** using your existing trained custom model (`Model/best.pt`).

---

## 📸 Application Visual Walkthrough

### 1. Candy Image Detection & Counting Interface
🔗 **Direct Reference:** [Detect.png - Google Drive](https://drive.google.com/file/d/1tNPqEq-_Odg7aLU5CVdyNejdtXamsJrL/view?usp=drive_open)

*The Candy Detection module delivers high-accuracy multi-object inference, automated brand identification, bounding box localization, and inventory counting reports.*

- **Interactive Inference Controls:** Dynamic sidebar adjustment for Confidence Threshold (default `0.25`), IoU Threshold (default `0.45`), and Input Dimension (`640px`).
- **High-Contrast Bounding Boxes:** 20 distinct accessible colors mapped to classes with label pills (`#ID CategoryName (Confidence %)`).
- **Tabbed Visualizations:**
  - *Annotated Detections:* Localized objects with bounding boxes and labels.
  - *Side-by-Side Comparison:* Direct side-by-side view of raw input versus model annotations.
  - *Original Image:* Full resolution unprocessed input.
- **Dynamic Candy Counting Report:** Dynamic category tally with formatted ASCII summaries and horizontal frequency bar charts.
- **Object Localization Table:** Comprehensive coordinate table recording `Detection ID`, `Category`, `Class ID`, `Confidence Score`, `Box (x1, y1)`, `Box (x2, y2)`, and `Box Dimensions (W×H)` with one-click CSV export.

---

### 2. Executive Analytics Dashboard
🔗 **Direct Reference:** [Dashboard.png - Google Drive](https://drive.google.com/file/d/1OPKcI8Pc2g8_gHudDHMhw_f7TCaXScm8/view)

*The Executive Dashboard computes real-time business and session statistics dynamically derived from live model detections.*

- **Metric KPI Cards:**
  - **Images Processed:** Total scans analyzed during the session.
  - **Candies Detected:** Cumulative count of all localized candy items.
  - **Unique Categories:** Active category breadth discovered (out of 20 possible classes).
  - **Most Frequent Candy:** Most prevalent confectionery brand detected with exact instance count.
- **Interactive Distribution Visualizations:**
  - *Category Distribution Donut:* Interactive Plotly donut chart highlighting class proportions.
  - *Category Frequency Breakdown:* Horizontal bar chart ranking detected brands.
  - *Activity Timeline:* Chronological volume curve plotting detection count per scan.
- **Recent Detection Activity Table:** Real-time audit log with Scan IDs, timestamps, image filenames, identified items, and inference latency in milliseconds.

---

## 🏛️ Refactored Architecture & Separation of Concerns

The codebase is engineered with strict separation of business logic, presentation components, data modeling, and utility layers:

```text
CandyDetection/
│
├── app.py                      # Main entrypoint, CSS injection, and multi-page routing
│
├── Model/
│   └── best.pt                 # Existing custom YOLOv8n weights (20 classes)
│
├── config/
│   ├── __init__.py             # Configuration package exports
│   └── settings.py             # Centralized constants, paths, thresholds, and color palettes
│
├── core/
│   ├── __init__.py             # Core package exports
│   ├── model_loader.py         # Framework-agnostic model loading & device discovery
│   ├── detector.py             # Dataclasses (DetectionItem, DetectionResult) & bounding boxes
│   ├── image_processor.py      # Input normalization, format conversions, thumbnails
│   └── candy_counter.py        # Tallying, ranking, and ASCII report formatting
│
├── services/
│   ├── __init__.py             # Services package exports
│   ├── detection_service.py    # Detection orchestration lifecycle
│   ├── statistics_service.py   # High-level session metrics and time-series extraction
│   ├── history_service.py      # Session state audit logging, caching, and CSV export
│   └── camera_service.py       # OpenCV webcam management and rolling FPS tracking
│
├── components/
│   ├── __init__.py             # Reusable UI component exports
│   ├── header.py               # Reusable gradient page header
│   ├── sidebar.py              # Centralized branding, controls, and session snapshot
│   ├── metric_cards.py         # Styled responsive KPI card containers
│   ├── detection_viewer.py     # Image tabs, report boxes, and coordinates tables
│   └── charts.py               # Interactive Plotly donut, bar, and timeline charts
│
├── pages/
│   ├── dashboard.py            # Module A: Executive Dashboard & live analytics
│   ├── candy_detection.py      # Module B, C, D: Image upload, inference, & counting
│   ├── realtime_detection.py   # Module E: Continuous webcam and snapshot detection
│   ├── candy_categories.py     # Module 7: 20-category directory & scan tallies
│   ├── detection_history.py    # Module F: Session audit logs and detail drill-down
│   └── about_model.py          # Module 8: YOLOv8 technical specifications
│
├── utils/
│   ├── __init__.py             # Utility package exports
│   ├── file_handler.py         # Path resolution, model verification, size helpers
│   ├── image_utils.py          # PIL/OpenCV array conversions & encoding
│   ├── validators.py           # Threshold boundaries, image dimensions, byte validation
│   └── helpers.py              # Formats, UUID generators, timestamps
│
├── assets/
│   ├── logo.png                # Candy AI high-resolution brand badge
│   ├── styles.css              # Modern AI dashboard custom styling
│   └── style.css               # Legacy alias stylesheet
│
├── tests/
│   ├── __init__.py
│   ├── test_detector.py        # Dataclass, mock prediction, and model integration tests
│   ├── test_image_processor.py # Array, format, byte, and thumbnail tests
│   ├── test_candy_counter.py   # Counting, ranking, and ASCII report tests
│   └── test_validators.py      # File format, dimension, byte, and threshold tests
│
├── .streamlit/
│   └── config.toml             # Streamlit server and primary orange/yellow theme config
│
├── .gitignore                  # Comprehensive Python & Streamlit gitignore
├── LICENSE                     # MIT License
├── requirements.txt            # Dependency specification
└── main.py                     # Convenience launcher script
```

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

## 💻 Windows Installation & Execution

### 1. Prerequisites

- Windows 10 or Windows 11 (64-bit)
- Python 3.10+ installed and added to `PATH`
- Web browser (Chrome, Edge, Firefox)

### 2. Environment Setup

```powershell
# Open terminal in project directory
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
*(Or alternatively run `python main.py`)*

The application will start on:

```text
http://localhost:8501
```

---

## 🧪 Automated Testing Suite

The project includes 33 modular unit and integration tests covering validators, image normalization, counting logic, data classes, and YOLO model inference:

```powershell
# Run all unit and integration tests
.\venv\Scripts\python.exe -m unittest discover tests
```

---

## 🛡️ Robustness & Error Handling

- **Missing Model Protection:** If `Model/best.pt` is missing or moved, the app displays a prominent error banner with the exact expected path.
- **Cached Singleton Model:** Model weights are loaded once via `@st.cache_resource` and cached across all pages, preventing memory leaks and redundant loading.
- **Format & Corrupt Image Validation:** Uploads are checked for format (`.jpg`, `.jpeg`, `.png`, `.webp`), size (max 25MB), and valid image header integrity before inference.
- **Graceful Camera Handling:** If the camera is busy or unavailable, clear troubleshooting guidance is provided without application termination.
- **Device Agnostic:** Seamlessly utilizes NVIDIA CUDA if available; defaults to optimized CPU execution otherwise.

---

## 📄 License & Attribution

Designed and engineered for Candy AI Object Detection applications using Ultralytics YOLOv8 and Streamlit.
Licensed under the [MIT License](LICENSE).
