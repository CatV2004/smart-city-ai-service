# app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading
import logging

from app.services.ai_service import AIService
from app.kafka.consumer import start_consumer, stop_consumer
from app.kafka.producer import close_producer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

# global AI service
ai_service: AIService | None = None
consumer_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service, consumer_thread

    logger.info("🚀 Initializing AI service...")
    ai_service = AIService()

    logger.info("📡 Starting Kafka consumer thread...")
    consumer_thread = threading.Thread(target=start_consumer, daemon=True)
    consumer_thread.start()
    logger.info("✅ Kafka consumer thread started")

    yield

    logger.info("🛑 Shutting down AI service...")
    
    # Dừng consumer
    stop_consumer()
    if consumer_thread:
        consumer_thread.join(timeout=10)
        logger.info("✅ Consumer thread stopped")
    
    # Đóng producer
    close_producer()
    
    logger.info("✅ AI service shutdown complete")


app = FastAPI(
    title="City Report AI Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-service"}


@app.get("/")
def root():
    return {"message": "City Report AI Service is running"}


@app.post("/detect")
async def detect(image_url: str):
    """Phân tích ảnh (không có text)"""
    result = await ai_service.analyze_image_only(image_url)
    return {"detections": result}


@app.post("/analyze")
async def analyze(image_url: str, title: str = None):
    """Phân tích ảnh và text"""
    if title:
        result = ai_service.analyze(image_url, title)
    else:
        result = await ai_service.analyze_image_only(image_url)
    return result


@app.post("/analyze/text")
async def analyze_text(title: str):
    """Phân tích chỉ text"""
    result = ai_service.analyze_text_only(title)
    return result