# One-shot Logo Recognition

This project implements a one-shot logo recognition system for videos and images. The system combines object detection and segmentation using YOLO with feature extraction and matching using ArcFace (based on the EfficientNet architecture). This allows the system to recognize new logos from a single reference image (one-shot) without the need to retrain the entire model.

---

## Pipeline Architecture (Multi-threading)

The video data stream is processed through a high-performance multi-threaded pipeline. The system utilizes independent Workers communicating via thread-safe queues (`CircularQueue`) to ensure concurrent execution and maximize processing speed:

1. **Input Worker**: Reads the input video/image stream and extracts frames sequentially.
2. **YOLO Detect Worker (Detection & Segmentation)**:
   - Uses the YOLO model (`best.pt`) to detect bounding boxes containing logos.
   - Applies segmentation masks to remove background noise, improving the accuracy of logo feature extraction.
3. **ArcFace Recognition Worker (Feature Extraction & Matching)**:
   - The cropped and masked logo regions are passed through the ArcFace neural network (`arcface_logo_model_best_b4_64_06.pth`).
   - ArcFace converts the logo image into a 1D feature vector (Embedding of size 512).
   - The system computes the Cosine similarity between this embedding and a database of known logos (`embedding_db.pkl`). If the similarity exceeds a threshold, the logo is labeled accordingly.
4. **Post-process & Output Worker**:
   - Aggregates recognition results. Draws bounding boxes, labels, and confidence scores onto the original frames.
   - Writes the processed frames to the output video file.

---

## Repository Structure (OOP Architecture)

```text
one-shot-logo-recognition/
├── scripts/               # Quick run entrypoint scripts.
├── src/                   # Main source code directory.
│   ├── oslr/              # Core CLI pipeline package (Multi-threading Workers & utils).
│   └── web/               # Visual Web Demo Application (OOP/MVC Architecture).
│       ├── app.py         # Application factory & SocketIO entrypoint.
│       ├── config.py      # Global configuration manager.
│       ├── utils.py       # Shared utility functions.
│       ├── extensions.py  # Extension initializations (e.g., SocketIO).
│       ├── events.py      # WebSocket real-time event management.
│       ├── routes/        # API & View routing.
│       │   ├── api.py     # Backend REST APIs.
│       │   └── views.py   # HTML template rendering.
│       ├── services/      # Business Logic Layer.
│       │   ├── video_service.py    # Video inference logic (YOLO + ArcFace).
│       │   └── registry_service.py # Logo registration & embedding storage.
│       ├── static/        # Static files (CSS, JS).
│       └── templates/     # HTML templates.
├── training/              # Model training scripts.
├── weights/               # Directory for model weights (YOLO, ArcFace).
├── dataset/               # Dataset directory.
├── requirements.txt       # Python dependencies.
└── README.md              # This documentation file.
```

---

## Environment Setup

**Python 3.8+** is required. It is recommended to use a virtual environment (virtualenv or conda).

```bash
# Install required libraries
pip install -r requirements.txt
```

---

## Model Weights

The system requires two weight files to operate. Please place these files in the `weights/` directory:
1. `best.pt`: YOLO model weights for object detection and segmentation.
2. `arcface_logo_model_best_b4_64_06.pth`: ArcFace model weights for feature extraction.

---

## Usage (Command Line Interface - CLI)

You can run the pipeline directly to process videos via the provided script helper.

Basic syntax (ensure you are in the root directory of the project):
```bash
python scripts/run_pipeline.py \
  --video "output/query.mp4" \
  --yolo-weights "weights/best.pt" \
  --recog-weights "weights/arcface_logo_model_best_b4_64_06.pth" \
  --embed-db "output/embedding_db.pkl" \
  --output "output/result.mp4" \
  --conf-threshold 0.7 \
  --recog-threshold 0.4
```

Alternatively, run the module directly (requires navigating to the `src` directory):
```bash
cd src
python -m oslr --help
```

### Main Parameters:
- `--video`: Path to the input video file.
- `--yolo-weights`: Path to the YOLO weights file.
- `--recog-weights`: Path to the ArcFace weights file.
- `--embed-db`: Path to the database storing logo vectors (Pickle file storing `(embedding, label)` tuples).
- `--output`: Path to export the resulting output video.
- `--conf-threshold`: YOLO confidence threshold (default: `0.7`).
- `--recog-threshold`: ArcFace similarity threshold for accepting a match (default: `0.4`).
- `--device`: Specify the device to run on (e.g., `cuda:0`, `cpu`). Defaults to auto-detection.

---

## Web App Demo

The project provides a lightweight Web interface (Flask + SocketIO) that allows users to upload videos, monitor processing progress in real-time, and preview visual results directly in the browser. The entire Web App codebase uses an OOP architecture and shares the EfficientNet-B4 model with the CLI.

```bash
cd src/web
python app.py
```
Once started, access the web interface via your browser at: `http://localhost:5000`

---

## Data Notes

- The `dataset/` directory is excluded via `.gitignore` by default to prevent uploading large datasets to GitHub.
- When testing locally, the system will automatically create an `output/` directory to store output videos and the database file.
