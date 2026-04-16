# Sign_Detection

Sign detection pipeline using **YOLO** for detection and **MobileNet** for classification.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python sign_detection.py \
  --source /absolute/path/to/image.jpg \
  --output /absolute/path/to/output.jpg \
  --yolo-model yolov8n.pt
```

Use a traffic-sign-trained YOLO model for best results.
