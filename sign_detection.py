from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LINE_THICKNESS = 2
TEXT_MIN_Y_OFFSET = 20
TEXT_Y_PADDING = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign detection with YOLO + MobileNet.")
    parser.add_argument("--source", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", type=Path, required=True, help="Output image path.")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or model name.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    return parser.parse_args()


def run_detection(source: Path, output: Path, yolo_model: str, confidence: float) -> None:
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    from ultralytics import YOLO

    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read image: {source}")

    detector = YOLO(yolo_model)
    classifier_weights = MobileNet_V2_Weights.DEFAULT
    classifier = mobilenet_v2(weights=classifier_weights).eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    categories = classifier_weights.meta["categories"]

    results = detector.predict(source=image, conf=confidence, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
        return

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        input_tensor = preprocess(pil_crop).unsqueeze(0)

        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = torch.softmax(logits[0], dim=0)
            cls_idx = torch.argmax(probs).item()
            cls_name = categories[cls_idx]
            cls_conf = float(probs[cls_idx].item())

        label = f"{cls_name} {cls_conf:.2f}"
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            DEFAULT_LINE_THICKNESS,
        )
        cv2.putText(
            image,
            label,
            (x1, max(TEXT_MIN_Y_OFFSET, y1 - TEXT_Y_PADDING)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            DEFAULT_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def main() -> None:
    args = parse_args()
    run_detection(args.source, args.output, args.yolo_model, args.confidence)


if __name__ == "__main__":
    main()
