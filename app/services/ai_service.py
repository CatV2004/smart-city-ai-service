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
    
    def fuse_predictions(
        self, 
        yolo_detections: List[Dict], 
        text_result: Dict
    ) -> Dict:
        """Kết hợp kết quả từ YOLO và Text Classifier"""
        
        # Khởi tạo scores
        class_scores = {cls: 0.0 for cls in self.class_names}
        
        # Từ YOLO
        for det in yolo_detections:
            label = det.get("label")
            conf = det.get("confidence", 0)
            if label in class_scores:
                class_scores[label] += conf * self.yolo_weight
        
        # Từ Text
        text_label = text_result["label"]
        text_conf = text_result["confidence"]
        if text_label in class_scores:
            class_scores[text_label] += text_conf * self.text_weight
        
        # Tìm label tốt nhất
        best_label = max(class_scores, key=class_scores.get)
        best_conf = class_scores[best_label]
        
        # Chuẩn hóa confidence
        max_possible = self.yolo_weight * max(1, len(yolo_detections)) + self.text_weight
        best_conf = min(best_conf / max_possible, 1.0)
        
        # Top predictions
        sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {"label": label, "label_vi": self.label_vi.get(label, label), "confidence": min(score, 1.0)}
            for label, score in sorted_scores[:3] if score > 0
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
        detections = self.yolo_predictor.predict(image)
        
        if not detections:
            return {
                "status": "no_detections",
                "detections": [],
                "message": "Không phát hiện vấn đề trong ảnh"
            }
        
        best_det = max(detections, key=lambda x: x["confidence"])
        
        return {
            "status": "success",
            "detections": detections,
            "best_detection": {
                "label": best_det["label"],
                "label_vi": self.label_vi.get(best_det["label"], best_det["label"]),
                "confidence": best_det["confidence"],
                "bbox": best_det["bbox"]
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
        Phân tích đầy đủ: Ảnh + Text
        """
        try:
            # 1. Tải ảnh
            image = self.download_image(image_url)
            
            # 2. YOLO Detection
            yolo_detections = self.yolo_predictor.predict(image)
            logger.info(f"YOLO detected {len(yolo_detections)} objects")
            
            # 3. Text Classification
            text_result = self.text_classifier.predict(title, return_all_probs=True)
            logger.info(f"Text classified as: {text_result['label_vi']} (conf: {text_result['confidence']:.3f})")
            
            # 4. Fusion
            if yolo_detections:
                fusion_result = self.fuse_predictions(yolo_detections, text_result)
            else:
                fusion_result = {
                    "label": text_result["label"],
                    "label_vi": text_result["label_vi"],
                    "confidence": text_result["confidence"] * self.text_weight,
                    "top_predictions": text_result.get("top_predictions", []),
                    "scores": {text_result["label"]: text_result["confidence"]}
                }
            
            return {
                "status": "success",
                "image_url": image_url,
                "title": title,
                "yolo_detections": yolo_detections,
                "text_classification": {
                    "label": text_result["label"],
                    "label_vi": text_result["label_vi"],
                    "confidence": text_result["confidence"],
                    "top_predictions": text_result.get("top_predictions", [])
                },
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