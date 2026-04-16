# app/services/ai_retry.py

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from app.services.ai_service import AIService
import logging

logger = logging.getLogger(__name__)

ai_service = AIService()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def analyze_with_retry(image_url: str, title: str = None):
    if title:
        return ai_service.analyze(image_url, title)
    else:
        return ai_service.analyze_image_only(image_url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def analyze_text_with_retry(title: str):
    return ai_service.analyze_text_only(title)