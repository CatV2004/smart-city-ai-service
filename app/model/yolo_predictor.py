from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict
from app.model.labels import CLASS_NAMES
from collections import defaultdict


class UrbanEyePredictor:
    def __init__(self, model_path: str):
        self.class_names = CLASS_NAMES

        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_half = True
        else:
            self.device = "cpu"
            self.use_half = False

        self.model = YOLO(model_path)
        if self.device.startswith("cuda") and self.use_half:
            self.model.fuse()
            self.model.to(self.device)
        
        print(f"✅ YOLO Predictor loaded on {self.device}")
        print(f"   Classes: {self.class_names}")

    def predict(self, image: np.ndarray, top_k: int = 3) -> Dict:
        """Predict image with top-K aggregated labels"""

        results = self.model(
            image,
            device=self.device,
            half=self.use_half,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append({
                        "label": self.class_names[cls_id],
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }
                    })

        top_predictions = self.aggregate_predictions(
            detections,
            top_k=top_k
        )

        prediction = top_predictions[0] if top_predictions else None

        return {
            "prediction": prediction,
            "top_predictions": top_predictions,
            "detections": detections
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Dự đoán cho batch ảnh"""
        results = self.model(images, device=self.device, half=self.use_half, verbose=False)
        
        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "label": self.class_names[cls_id],
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })
            all_detections.append(detections)
        
        return all_detections
    
    def aggregate_predictions(
        self,
        detections: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:

        label_scores = defaultdict(float)

        for det in detections:
            label = det["label"]
            conf = det["confidence"]

            # max confidence per label
            label_scores[label] = max(label_scores[label], conf)

        sorted_preds = sorted(
            [
                {
                    "label": label,
                    "label_vi": label,
                    "confidence": score
                }
                for label, score in label_scores.items()
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )

        return sorted_preds[:top_k]