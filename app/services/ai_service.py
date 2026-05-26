import requests
import cv2
import numpy as np
import logging
from typing import Dict, List

from app.config import settings
from app.model.yolo_predictor import UrbanEyePredictor
from app.model.text_classifier import UrbanTextClassifier
from app.model.labels import CLASS_NAMES, LABEL_VI

logger = logging.getLogger(__name__)


class AIService:
    """Hybrid AI Service: YOLO (image) + Text Classifier"""
    
    def __init__(self):
        # YOLO Predictor
        self.yolo_predictor = UrbanEyePredictor(settings.MODEL_PATH)
        
        # Text Classifier
        self.text_classifier = UrbanTextClassifier(
            model_path=settings.TEXT_MODEL_PATH,
            tokenizer_path=settings.TEXT_TOKENIZER_PATH
        )
        
        # Class names
        self.class_names = CLASS_NAMES
        self.label_vi = LABEL_VI
        
        # Trọng số fusion
        self.yolo_weight = settings.YOLO_WEIGHT
        self.text_weight = settings.TEXT_WEIGHT
        
        logger.info("✅ AI Service initialized (YOLO + Text Classifier)")
    
    def download_image(self, image_url: str) -> np.ndarray:
        """Tải ảnh từ URL"""
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    
    def fuse_predictions(self, yolo_detections, text_result):

        class_scores = {cls: 0.0 for cls in self.class_names}

        # ----------------------
        # YOLO → MAX aggregation
        # ----------------------
        yolo_max_scores = {}

        for det in yolo_detections:
            label = det.get("label")
            conf = det.get("confidence", 0)

            if label not in yolo_max_scores:
                yolo_max_scores[label] = conf
            else:
                yolo_max_scores[label] = max(yolo_max_scores[label], conf)

        for label, conf in yolo_max_scores.items():
            if label in class_scores:
                class_scores[label] = conf * self.yolo_weight

        # ----------------------
        # TEXT → top-K fusion
        # ----------------------
        text_predictions = text_result.get("top_predictions", [])

        for pred in text_predictions:
            label = pred["label"]
            conf = pred["confidence"]

            if label in class_scores:
                class_scores[label] += conf * self.text_weight

        # ----------------------
        # BEST LABEL
        # ----------------------
        best_label = max(class_scores, key=class_scores.get)
        best_conf = class_scores[best_label]

        # ----------------------
        # NORMALIZATION (optional)
        # ----------------------
        best_conf = min(best_conf, 1.0)

        # ----------------------
        # TOP-K
        # ----------------------
        sorted_scores = sorted(
            class_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_predictions = [
            {
                "label": label,
                "label_vi": self.label_vi.get(label, label),
                "confidence": score
            }
            for label, score in sorted_scores[:3]
            if score > 0
        ]

        return {
            "label": best_label,
            "label_vi": self.label_vi.get(best_label, best_label),
            "confidence": best_conf,
            "top_predictions": top_predictions,
            "scores": class_scores
        }
    
    def analyze_image_only(self, image_url: str) -> Dict:
        """Phân tích chỉ ảnh (không có text)"""

        image = self.download_image(image_url)

        image_result = self.yolo_predictor.predict(image)

        detections = image_result["detections"]
        top_predictions = image_result["top_predictions"]
        prediction = image_result["prediction"]

        if not detections:
            return {
                "status": "no_detections",
                "detections": [],
                "top_predictions": [],
                "message": "Không phát hiện vấn đề trong ảnh"
            }

        return {
            "status": "success",

            # detection objects từ YOLO
            "detections": detections,

            # top-K labels để phục vụ fusion
            "top_predictions": [
                {
                    "label": pred["label"],
                    "label_vi": self.label_vi.get(
                        pred["label"],
                        pred["label"]
                    ),
                    "confidence": pred["confidence"]
                }
                for pred in top_predictions
            ],

            # prediction tốt nhất
            "best_detection": {
                "label": prediction["label"],
                "label_vi": self.label_vi.get(
                    prediction["label"],
                    prediction["label"]
                ),
                "confidence": prediction["confidence"]
            }
        }
    
    def analyze_text_only(self, title: str) -> Dict:
        """Phân tích chỉ text (không có ảnh)"""
        result = self.text_classifier.predict(title, return_all_probs=True)
        
        return {
            "status": "success",
            "text": title,
            "prediction": {
                "label": result["label"],
                "label_vi": result["label_vi"],
                "confidence": result["confidence"]
            },
            "top_predictions": result.get("top_predictions", [])
        }
    
    def analyze(self, image_url: str, title: str) -> Dict:
        """
        Phân tích đầy đủ: Ảnh + Text (Multimodal Fusion)
        """
        try:
            # 1. Tải ảnh
            image = self.download_image(image_url)

            # 2. YOLO Prediction (UPDATED CONTRACT)
            image_result = self.yolo_predictor.predict(image)

            yolo_detections = image_result["detections"]
            yolo_top_predictions = image_result["top_predictions"]

            logger.info(
                f"YOLO detected {len(yolo_detections)} objects"
            )

            # 3. Text Classification (PhoBERT)
            text_result = self.text_classifier.predict(
                title,
                return_all_probs=True
            )

            logger.info(
                f"Text classified as: {text_result['label_vi']} "
                f"(conf: {text_result['confidence']:.3f})"
            )

            # 4. Fusion (UPDATED INPUT)
            if yolo_top_predictions:
                fusion_result = self.fuse_predictions(
                    yolo_top_predictions,
                    text_result
                )
            else:
                fusion_result = {
                    "label": text_result["label"],
                    "label_vi": text_result["label_vi"],
                    "confidence": text_result["confidence"] * self.text_weight,
                    "top_predictions": text_result.get("top_predictions", []),
                    "scores": {
                        text_result["label"]: text_result["confidence"]
                    }
                }

            # 5. Return unified response
            return {
                "status": "success",
                "image_url": image_url,
                "title": title,

                # raw YOLO outputs (for explainability)
                "yolo_detections": yolo_detections,

                # image-level predictions (IMPORTANT for fusion)
                "yolo_top_predictions": yolo_top_predictions,

                # text model output
                "text_classification": {
                    "label": text_result["label"],
                    "label_vi": text_result["label_vi"],
                    "confidence": text_result["confidence"],
                    "top_predictions": text_result.get("top_predictions", [])
                },

                # final multimodal decision
                "final_prediction": fusion_result
            }

        except Exception as e:
            logger.error(f"Error analyzing: {e}")
            return {
                "status": "error",
                "message": str(e),
                "image_url": image_url,
                "title": title
            }