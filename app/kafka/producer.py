import json
import logging
from confluent_kafka import Producer
from uuid import UUID

from app.config import settings

logger = logging.getLogger(__name__)

_producer: Producer = None


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def get_producer() -> Producer:
    global _producer
    if _producer is None:
        conf = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'acks': 'all',
            'retries': 3,
            'retry.backoff.ms': 1000,
        }
        _producer = Producer(conf)
        logger.info(f"✅ Kafka producer connected to {settings.KAFKA_BOOTSTRAP_SERVERS}")
    return _producer


def delivery_callback(err, msg):
    if err:
        logger.error(f"❌ Message delivery failed: {err}")
    else:
        logger.debug(f"📤 Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def publish_ai_result(message: dict) -> bool:
    try:
        producer = get_producer()
        
        value = json.dumps(message, ensure_ascii=False, cls=UUIDEncoder).encode('utf-8')
        
        producer.produce(
            topic=settings.REPORT_AI_ANALYZED_TOPIC,
            value=value,
            callback=delivery_callback
        )
        
        producer.flush(timeout=5)
        
        logger.info(f"✅ Published AI result to {settings.REPORT_AI_ANALYZED_TOPIC}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to publish message: {e}")
        return False


def close_producer():
    global _producer
    if _producer:
        _producer.flush(timeout=5)
        _producer = None
        logger.info("🔚 Kafka producer closed")