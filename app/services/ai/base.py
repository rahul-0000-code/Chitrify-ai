from abc import ABC, abstractmethod
from typing import Optional

from app.utils.constants import ImageStyle

class AIService(ABC):
    """Abstract AI service interface"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI service is available"""
        pass
    
    @abstractmethod
    async def process_image(self, image_content: bytes, style: ImageStyle) -> Optional[bytes]:
        """Process image with AI model"""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of the AI service"""
        pass
