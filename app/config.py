from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ========================
    # YOLO MODEL
    # ========================
    MODEL_PATH: str = "models/yolo/best_YoLov8m.pt"  
    
    # ========================
    # TEXT MODEL
    # ========================
    TEXT_MODEL_PATH: str = "models/text/best_model.pt"
    TEXT_TOKENIZER_PATH: str = "models/text/tokenizer"
    TEXT_MAX_LENGTH: int = 128
    TEXT_CONFIDENCE_THRESHOLD: float = 0.5
    
    # ========================
    # FUSION WEIGHTS 
    # ========================
    YOLO_WEIGHT: float = 0.6
    TEXT_WEIGHT: float = 0.4
    
    # ========================
    # REDIS 
    # ========================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_PENDING_TTL: int = 3600
    REDIS_PENDING_PREFIX: str = "report:pending:"
    
    # ========================
    # KAFKA
    # ========================
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    
    # ========================
    # THRESHOLDS
    # ========================
    CONFIDENCE_THRESHOLD: float = 0.2  # YOLO confidence threshold
    
    # ========================
    # KAFKA TOPICS
    # ========================
    REPORT_CREATED_TOPIC: str = "report.created"
    REPORT_ATTACHMENTS_ADDED_TOPIC: str = "report.attachments.added"
    REPORT_AI_ANALYZED_TOPIC: str = "report-ai-analyzed"
    
    # ========================
    # API SETTINGS
    # ========================
    IMAGE_DOWNLOAD_TIMEOUT: int = 10
    MAX_IMAGE_SIZE: int = 640
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Bỏ qua các biến môi trường không được định nghĩa


settings = Settings()