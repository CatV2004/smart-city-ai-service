import redis
import json
import logging
from typing import Optional, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """Service quản lý kết nối và thao tác với Redis"""
    
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
            decode_responses=True  # Tự động decode về string
        )
        self.pending_prefix = settings.REDIS_PENDING_PREFIX
        self.pending_ttl = settings.REDIS_PENDING_TTL
        
        # Kiểm tra kết nối
        try:
            self.client.ping()
            logger.info(f"✅ Redis connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def _get_key(self, report_id: str) -> str:
        """Tạo key cho report"""
        return f"{self.pending_prefix}{report_id}"
    
    def save_pending_report(self, report_id: str, data: Dict[str, Any]) -> bool:
        """
        Lưu thông tin report đang chờ xử lý
        
        Args:
            report_id: ID của report
            data: Dữ liệu cần lưu (title, description, ...)
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            key = self._get_key(report_id)
            self.client.setex(
                key,
                self.pending_ttl,
                json.dumps(data, ensure_ascii=False)
            )
            logger.info(f"💾 Saved pending report to Redis: {report_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save to Redis: {e}")
            return False
    
    def get_pending_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin report đang chờ xử lý
        
        Args:
            report_id: ID của report
        
        Returns:
            Dữ liệu đã lưu hoặc None nếu không tìm thấy
        """
        try:
            key = self._get_key(report_id)
            data = self.client.get(key)
            if data:
                logger.info(f"📥 Retrieved pending report from Redis: {report_id}")
                return json.loads(data)
            else:
                logger.warning(f"⚠️ No pending report found in Redis: {report_id}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get from Redis: {e}")
            return None
    
    def delete_pending_report(self, report_id: str) -> bool:
        """
        Xóa thông tin report đã xử lý xong
        
        Args:
            report_id: ID của report
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            key = self._get_key(report_id)
            self.client.delete(key)
            logger.info(f"🗑️ Deleted pending report from Redis: {report_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete from Redis: {e}")
            return False
    
    def update_pending_report(self, report_id: str, data: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin report đang chờ
        
        Args:
            report_id: ID của report
            data: Dữ liệu cần cập nhật
        
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            key = self._get_key(report_id)
            existing = self.client.get(key)
            if existing:
                existing_data = json.loads(existing)
                existing_data.update(data)
                self.client.setex(
                    key,
                    self.pending_ttl,
                    json.dumps(existing_data, ensure_ascii=False)
                )
                logger.info(f"📝 Updated pending report in Redis: {report_id}")
                return True
            else:
                return self.save_pending_report(report_id, data)
        except Exception as e:
            logger.error(f"❌ Failed to update Redis: {e}")
            return False
    
    def health_check(self) -> bool:
        """Kiểm tra Redis có hoạt động không"""
        try:
            return self.client.ping()
        except:
            return False


# Singleton instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Lấy singleton instance của RedisService"""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service