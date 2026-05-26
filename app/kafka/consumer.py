import json
import logging

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka import Message as KafkaMessage

from app.config import settings
from app.retry.retry_handler import analyze_with_retry
from app.kafka.producer import publish_ai_result
from app.services.redis_service import get_redis_service

from app.schema.messages import (
    ReportCreatedMessage,
    ReportAttachmentsAddedMessage,
    ReportAIAnalyzedMessage,
    Prediction
)

logger = logging.getLogger(__name__)

# Khởi tạo Redis service
redis_service = get_redis_service()

# Global flag để graceful shutdown
running = True


def create_consumer() -> Consumer:
    conf = {
        'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'ai-service',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'session.timeout.ms': 30000,
        'max.poll.interval.ms': 300000,
    }
    
    consumer = Consumer(conf)
    return consumer


def handle_report_created(event: dict):
    try:
        message = ReportCreatedMessage(**event)
    except Exception as e:
        logger.error(f"❌ Failed to parse ReportCreatedMessage: {e}")
        return
    
    logger.info(f"📝 Received report_created: {message.reportId}")
    logger.info(f"   Title: {message.title}")
    logger.info(f"   Description: {message.description[:100] if message.description else 'None'}...")
    
    report_data = {
        "title": message.title,
        "description": message.description,
        "category": message.category,
        "address": message.address,
        # "created_at": message.createdAt.isoformat() if message.createdAt else None
    }
    
    success = redis_service.save_pending_report(str(message.reportId), report_data)
    if success:
        logger.info(f"   ✅ Saved to Redis with TTL={settings.REDIS_PENDING_TTL}s")
    else:
        logger.error(f"   ❌ Failed to save to Redis")


def handle_report_attachments(event: dict):
    try:
        message = ReportAttachmentsAddedMessage(**event)
    except Exception as e:
        logger.error(f"❌ Failed to parse ReportAttachmentsAddedMessage: {e}")
        return

    logger.info(
        f"🖼️ Processing attachments for report {message.reportId}"
    )
    logger.info(
        f"   Attachments: {len(message.attachmentUrls)} images"
    )

    # --------------------------------------------------
    # Get pending report info from Redis
    # --------------------------------------------------
    report_id_str = str(message.reportId)
    pending_info = redis_service.get_pending_report(report_id_str)

    title = ""

    if pending_info:
        title = pending_info.get("title", "")
        logger.info(
            f"   📥 Retrieved from Redis: "
            f"title='{title[:50] if title else ''}...'"
        )
    else:
        logger.warning(
            f"   ⚠️ No pending data found in Redis "
            f"for report {message.reportId}"
        )

    # --------------------------------------------------
    # Collect image-level predictions only
    # --------------------------------------------------
    all_predictions = []

    for idx, image_url in enumerate(message.attachmentUrls):

        try:
            logger.info(
                f"   🖼️ Analyzing image "
                f"{idx+1}/{len(message.attachmentUrls)}"
            )

            # ==================================================
            # CASE 1: Multimodal Fusion (Image + Text)
            # ==================================================
            if title:

                result = analyze_with_retry(image_url, title)
                logger.info(
                    f"   🔍 Multimodal analysis result: {result}")

                if result.get("status") != "success":
                    continue

                final_pred = result.get("final_prediction")
                logger.info(f"   🔍 Multimodal analysis final_pred: {final_pred}")

                if final_pred:

                    all_predictions.append(
                        Prediction(
                            label=final_pred.get("label"),
                            confidence=final_pred.get("confidence")
                        )
                    )

                    logger.info(
                        f"✅ Fusion Final: "
                        f"{final_pred.get('label_vi')} "
                        f"({final_pred.get('confidence'):.3f})"
                    )

            # ==================================================
            # CASE 2: Image Only
            # ==================================================
            else:

                result = analyze_with_retry(image_url)

                if result.get("status") != "success":
                    continue

                yolo_top_predictions = result.get(
                    "top_predictions",
                    []
                )

                for pred in yolo_top_predictions:

                    all_predictions.append(
                        Prediction(
                            label=pred.get("label"),
                            confidence=pred.get("confidence")
                        )
                    )

                    logger.info(
                        f"✅ YOLO Image-Level: "
                        f"{pred.get('label_vi', pred.get('label'))} "
                        f"({pred.get('confidence'):.3f})"
                    )

        except Exception as e:
            logger.error(
                f"   ❌ Failed to analyze {image_url}: {e}"
            )

    # --------------------------------------------------
    # Cleanup Redis
    # --------------------------------------------------
    redis_service.delete_pending_report(report_id_str)

    logger.info(
        f"   🗑️ Removed pending report from Redis"
    )

    # --------------------------------------------------
    # No prediction
    # --------------------------------------------------
    if not all_predictions:

        logger.info(
            f"📭 No predictions for report {message.reportId}"
        )

        ai_message = ReportAIAnalyzedMessage(
            type="ReportAIAnalyzedMessage",
            reportId=message.reportId,
            predictions=[]
        )

    # --------------------------------------------------
    # Aggregate report-level predictions
    # --------------------------------------------------
    else:

        best_by_label = {}

        for pred in all_predictions:

            label = pred.label

            if (
                label not in best_by_label
                or pred.confidence > best_by_label[label].confidence
            ):
                best_by_label[label] = pred

        # sort descending confidence
        sorted_preds = sorted(
            best_by_label.values(),
            key=lambda x: x.confidence,
            reverse=True
        )

        top_predictions = sorted_preds[:3]

        ai_message = ReportAIAnalyzedMessage(
            type="ReportAIAnalyzedMessage",
            reportId=message.reportId,
            predictions=top_predictions
        )

        logger.info(
            f"📊 Final predictions for report "
            f"{message.reportId}:"
        )

        for pred in top_predictions:
            logger.info(
                f"      - {pred.label}: "
                f"{pred.confidence:.3f}"
            )

    # --------------------------------------------------
    # Publish result
    # --------------------------------------------------
    publish_ai_result(ai_message.model_dump())

    logger.info(
        f"📤 Published AI result for report "
        f"{message.reportId}"
    )


def process_message(msg: KafkaMessage) -> bool:
    topic = msg.topic()
    
    try:
        value = json.loads(msg.value().decode('utf-8'))
        print(f"\n📨 Received message from {topic}")
        
        if topic == settings.REPORT_CREATED_TOPIC:
            handle_report_created(value)
            return True
            
        elif topic == settings.REPORT_ATTACHMENTS_ADDED_TOPIC:
            handle_report_attachments(value)
            return True
            
        else:
            logger.warning(f"⚠️ Unknown topic: {topic}")
            return True
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to decode JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        return False


def start_consumer():
    global running
    
    consumer = None
    
    try:
        print("🚀 Starting Kafka consumer...")
        
        # Kiểm tra Redis connection
        if not redis_service.health_check():
            raise Exception("Redis is not available")
        
        consumer = create_consumer()
        
        # Subscribe to topics
        topics = [settings.REPORT_CREATED_TOPIC, settings.REPORT_ATTACHMENTS_ADDED_TOPIC]
        consumer.subscribe(topics)
        
        print("✅ Kafka consumer connected")
        print(f"   Bootstrap servers: {settings.KAFKA_BOOTSTRAP_SERVERS}")
        print(f"   Listening to: {', '.join(topics)}")
        print(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        print("\n⏳ Waiting for messages...\n")
        
        while running:
            # Poll for messages
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"❌ Kafka error: {msg.error()}")
                    continue
            
            # Process message
            success = process_message(msg)
            
            # Commit offset nếu xử lý thành công
            if success:
                consumer.commit(msg)
                logger.debug(f"✅ Committed offset for {msg.topic()} [{msg.partition()}:{msg.offset()}]")
            else:
                logger.warning(f"⚠️ Not committing offset due to processing error")
                
    except KafkaException as e:
        print(f"❌ Kafka error: {e}")
    except Exception as e:
        print(f"❌ Consumer error: {e}")
    finally:
        if consumer:
            print("\n🔚 Closing Kafka consumer...")
            consumer.close()
            print("✅ Consumer closed")


def stop_consumer():
    """Dừng consumer"""
    global running
    running = False
    print("🛑 Stop signal sent to consumer")